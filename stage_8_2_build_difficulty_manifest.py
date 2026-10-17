#!/usr/bin/env python3
"""
Stage 8.2 — Build Object & Image Difficulty Manifest (Post-OOF) (Hardened)
==========================================================================

Kontrak Metodologis:
1. Membaca prediksi OOF mentah dari stage_8_1_oof/oof_predictions_raw.csv (conf >= 0.001).
2. Membaca ground truth langsung dari file label YOLO (8.980 citra train).
3. Menggunakan fungsi evaluasi kanonis terpadu dari research_utils.detection_metrics:
   - greedy_bipartite_matching (Global 1-to-1 bipartite greedy matching pada IoU >= 0.10, zero reuse)
   - classify_object_size (Small < 32^2, Medium 32^2-96^2, Large >= 96^2)
   - calculate_object_difficulty (Matched+Correct, Matched+Wrong Class [penalty 0.35], Unmatched FN [0.80 + 0.20 S_i])
   - calculate_image_difficulty (0.5 * max + 0.5 * mean untuk citra berobjek, max(C_FP) untuk citra kosong)
   - calculate_mixture_weights (P(k) = (1 - lambda)/N + lambda * D_k / sum(D), lambda = 0.70)
4. Sanity Gates:
   - Hard ratio (D_image >= 0.80): <=60% PASS, 60-80% REVIEW, >80% FAIL
   - Dynamic Range: P90 - P10 >= 0.30
   - Effective Sample Size (ESS) dan ESS_ratio in [0.65, 0.95]
   - Clipping rate pada wrong-class cases
5. Mengekspor manifest kesulitan citra untuk digunakan pada Stage 8.3 & 8.4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from research_utils.detection_metrics import (
    calculate_image_difficulty,
    calculate_iou_xyxy,
    calculate_mixture_weights,
    calculate_object_difficulty,
    classify_object_size,
    get_image_dimensions,
    greedy_bipartite_matching,
    parse_yolo_labels,
)

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 8.2: Build Difficulty Manifest (Hardened)")
    parser.add_argument(
        "--oof-dir",
        default="d:/UNDIP/SKRIPSI/stage_8_1_oof",
        help="Folder output dari Stage 8.1",
    )
    parser.add_argument(
        "--data-yaml",
        default=None,
        help="Path ke data.yaml (Single Source of Truth). Jika diisi, direktori train diturunkan otomatis.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path alternatif root dataset (misal cassava-v7-standardized atau cassava-v7-research).",
    )
    parser.add_argument(
        "--research-dir",
        default="d:/UNDIP/SKRIPSI/cassava-v7-research",
        help="Path fallback dataset resmi cassava-v7-research",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/UNDIP/SKRIPSI/stage_8_2_difficulty",
        help="Folder output artefak manifest kesulitan",
    )
    parser.add_argument(
        "--iou-match-thresh",
        type=float,
        default=0.10,
        help="Ambang batas minimum IoU untuk matching kandidat (default: 0.10)",
    )
    parser.add_argument(
        "--confusion-penalty",
        type=float,
        default=0.35,
        help="Fixed design penalty untuk kasus salah kelas (default: 0.35)",
    )
    parser.add_argument(
        "--mixture-lambda",
        type=float,
        default=0.70,
        help="Bobot komponen adaptif pada mixture sampling (default: 0.70)",
    )
    return parser.parse_args()


def resolve_train_paths(args) -> tuple[Path, Path]:
    """Menentukan path direktori train/images dan train/labels secara Single Source of Truth."""
    if args.data_yaml:
        yaml_p = Path(args.data_yaml).resolve()
        if not yaml_p.exists():
            raise FileNotFoundError(f"data.yaml tidak ditemukan di {yaml_p}")
        with open(yaml_p, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        
        base_path = yaml_p.parent
        if "path" in data_cfg and data_cfg["path"]:
            base_path = Path(data_cfg["path"]).resolve()

        train_entry = data_cfg.get("train", "train/images")
        train_path = (base_path / train_entry).resolve()
        if train_path.name == "images":
            images_dir = train_path
            labels_dir = train_path.parent / "labels"
        else:
            images_dir = train_path / "images"
            labels_dir = train_path / "labels"
    elif args.dataset_dir:
        ds_p = Path(args.dataset_dir).resolve()
        images_dir = ds_p / "train" / "images"
        labels_dir = ds_p / "train" / "labels"
    else:
        r_p = Path(args.research_dir).resolve()
        images_dir = r_p / "train" / "images"
        labels_dir = r_p / "train" / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Direktori citra latih tidak ditemukan: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Direktori label latih tidak ditemukan: {labels_dir}")

    return images_dir, labels_dir


def load_ground_truth(images_dir: Path, labels_dir: Path) -> dict[str, list[dict]]:
    """Membaca seluruh ground truth objek menggunakan parse_yolo_labels kanonis dengan dimensi citra aktual."""
    image_files = sorted(os.listdir(images_dir))
    gt_by_image = {}
    total_gt_objects = 0

    for fname in image_files:
        stem = Path(fname).stem
        lbl_file = labels_dir / f"{stem}.txt"
        img_file = images_dir / fname

        img_w, img_h = get_image_dimensions(img_file)
        parsed_objs = parse_yolo_labels(lbl_file, img_width=img_w, img_height=img_h)
        objects = []
        for obj_id, po in enumerate(parsed_objs):
            box_xyxy = po["box"].tolist() if isinstance(po["box"], np.ndarray) else list(po["box"])
            area_px = float(po["area"])
            size_cat = po["size_category"]

            # Map size to difficulty score S_i
            if size_cat == "Small":
                size_score = 1.0
            elif size_cat == "Medium":
                size_score = 0.5
            else:
                size_score = 0.0

            objects.append({
                "gt_id": obj_id,
                "class_id": po["class_id"],
                "class_name": po["class_name"],
                "box_xyxy": box_xyxy,
                "area_px": area_px,
                "size_cat": size_cat,
                "size_score": size_score,
                "img_width": img_w,
                "img_height": img_h,
            })
            total_gt_objects += 1

        gt_by_image[fname] = objects

    print(f"[GROUND TRUTH] Memuat {len(gt_by_image):,} citra dengan total {total_gt_objects:,} objek ground truth.")
    return gt_by_image


def load_oof_predictions(raw_pred_file: Path) -> dict[str, list[dict]]:
    """Membaca prediksi OOF mentah hasil inferensi diagnostik @ conf = 0.001."""
    if not raw_pred_file.exists():
        raise FileNotFoundError(f"File prediksi OOF tidak ditemukan: {raw_pred_file}")

    preds_by_image = defaultdict(list)
    total_preds = 0

    with open(raw_pred_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["image_id"]
            box = [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ]
            preds_by_image[fname].append({
                "pred_id": int(row["prediction_id"]),
                "fold_id": int(row["fold_id"]),
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "box_xyxy": box,
            })
            total_preds += 1

    print(f"[OOF PREDICTIONS] Memuat {total_preds:,} prediksi mentah untuk {len(preds_by_image):,} citra.")
    return preds_by_image


def process_difficulty_manifest(
    gt_by_image: dict[str, list[dict]],
    preds_by_image: dict[str, list[dict]],
    output_dir: Path,
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    object_rows = []
    image_rows = []

    total_gt = 0
    taxonomy_counter = Counter()
    wrong_class_total = 0
    wrong_class_clipped = 0

    all_d_images_raw = []

    for fname, gts in sorted(gt_by_image.items()):
        preds = preds_by_image.get(fname, [])
        n_gt = len(gts)

        # Canonical greedy bipartite matching from research_utils.detection_metrics
        # Prepares candidate pairs with descending IoU
        gt_boxes = [g["box_xyxy"] for g in gts]
        pred_boxes = [p["box_xyxy"] for p in preds]
        matches, unassigned_preds = greedy_bipartite_matching(
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            iou_thresh=args.iou_match_thresh,
        )

        obj_difficulties = []
        target_small_candidates = []

        if n_gt > 0:
            for g_idx, g in enumerate(gts):
                total_gt += 1
                size_score = g["size_score"]
                gt_cls = g["class_id"]

                if g_idx in matches:
                    p_idx, iou_val = matches[g_idx]
                    p = preds[p_idx]
                    p_cls = p["class_id"]
                    conf = p["confidence"]

                    is_correct_class = (gt_cls == p_cls)

                    if is_correct_class:
                        if iou_val >= 0.50:
                            taxonomy = "True_Positive"
                        else:
                            taxonomy = "Localization_Error"
                    else:
                        if iou_val >= 0.50:
                            taxonomy = "Classification_Confusion"
                        else:
                            taxonomy = "Localization_and_Class_Error"
                        wrong_class_total += 1

                    # Canonical calculate_object_difficulty
                    d_i = calculate_object_difficulty(
                        matched=True,
                        conf=conf,
                        iou=iou_val,
                        size_score=size_score,
                        correct_class=is_correct_class,
                        confusion_penalty=args.confusion_penalty,
                    )

                    if not is_correct_class and (0.40 * (1.0 - conf) + 0.40 * (1.0 - iou_val) + 0.20 * size_score + args.confusion_penalty) >= 1.0:
                        wrong_class_clipped += 1

                    match_status = "Matched"
                    matched_conf = conf
                    matched_iou = iou_val
                else:
                    taxonomy = "False_Negative_Missed"
                    match_status = "Unmatched"
                    matched_conf = 0.0
                    matched_iou = 0.0
                    d_i = calculate_object_difficulty(
                        matched=False,
                        size_score=size_score,
                    )

                taxonomy_counter[taxonomy] += 1
                obj_difficulties.append(d_i)

                # Evaluasi lesi small yang sulit (target potensial context crop)
                if g["size_cat"] == "Small" and d_i >= 0.70:
                    target_small_candidates.append({
                        "gt_id": g["gt_id"],
                        "d_i": d_i,
                        "box": g["box_xyxy"],
                        "class_id": gt_cls,
                        "img_width": g.get("img_width", 640.0),
                        "img_height": g.get("img_height", 640.0),
                    })

                object_rows.append({
                    "image_id": fname,
                    "gt_id": g["gt_id"],
                    "class_name": g["class_name"],
                    "size_cat": g["size_cat"],
                    "match_status": match_status,
                    "taxonomy": taxonomy,
                    "confidence": f"{matched_conf:.4f}",
                    "iou": f"{matched_iou:.4f}",
                    "size_score": f"{size_score:.2f}",
                    "difficulty_score": f"{d_i:.4f}",
                })

            fp_confs = [preds[p_idx]["confidence"] for p_idx in unassigned_preds]
            d_image = calculate_image_difficulty(obj_difficulties=obj_difficulties, fp_confs=fp_confs)
            has_gt = 1
        else:
            has_gt = 0
            fp_confs = [preds[p_idx]["confidence"] for p_idx in unassigned_preds]
            d_image = calculate_image_difficulty(obj_difficulties=[], fp_confs=fp_confs)

        all_d_images_raw.append((fname, d_image, has_gt, target_small_candidates))

    # Mixture sampling weights using canonical calculate_mixture_weights
    d_list = [d for _, d, _, _ in all_d_images_raw]
    p_weights = calculate_mixture_weights(d_list, mixture_lambda=args.mixture_lambda)

    n_images = len(all_d_images_raw)
    for idx, (fname, d_image, has_gt, target_smalls) in enumerate(all_d_images_raw):
        p_mixture = p_weights[idx]
        gts_for_img = gt_by_image.get(fname, [])
        first_gt = gts_for_img[0] if gts_for_img else {}
        base_w = int(first_gt.get("img_width", 640))
        base_h = int(first_gt.get("img_height", 640))

        if target_smalls:
            best_small = max(target_smalls, key=lambda x: x["d_i"])
            target_small_exists = 1
            target_gt_id = best_small["gt_id"]
            im_w = float(best_small.get("img_width", base_w))
            im_h = float(best_small.get("img_height", base_h))
            bx1, by1, bx2, by2 = best_small["box"]
            nbx1 = max(0.0, min(1.0, bx1 / im_w))
            nby1 = max(0.0, min(1.0, by1 / im_h))
            nbx2 = max(0.0, min(1.0, bx2 / im_w))
            nby2 = max(0.0, min(1.0, by2 / im_h))
            target_box = f"{nbx1:.6f},{nby1:.6f},{nbx2:.6f},{nby2:.6f}"
            target_d_i = f"{best_small['d_i']:.4f}"
            box_fmt = "norm_xyxy"
            row_w, row_h = int(im_w), int(im_h)
        else:
            target_small_exists = 0
            target_gt_id = -1
            target_box = ""
            target_d_i = "0.0000"
            box_fmt = ""
            row_w, row_h = base_w, base_h

        image_rows.append({
            "image_id": fname,
            "d_image": f"{d_image:.6f}",
            "p_sampling": f"{p_mixture:.8e}",
            "has_gt": has_gt,
            "has_target_small": target_small_exists,
            "target_gt_id": target_gt_id,
            "target_small_di": target_d_i,
            "target_small_box": target_box,
            "box_format": box_fmt,
            "img_width": row_w,
            "img_height": row_h,
        })

    # Simpan train_object_difficulty.csv
    obj_csv_path = output_dir / "train_object_difficulty.csv"
    with open(obj_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "gt_id", "class_name", "size_cat", "match_status",
            "taxonomy", "confidence", "iou", "size_score", "difficulty_score"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)
    print(f"[OUTPUT] Tersimpan: {obj_csv_path} ({len(object_rows):,} baris objek)")

    # Simpan train_image_difficulty_manifest.csv
    img_csv_path = output_dir / "train_image_difficulty_manifest.csv"
    with open(img_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "d_image", "p_sampling", "has_gt", "has_target_small",
            "target_gt_id", "target_small_di", "target_small_box",
            "box_format", "img_width", "img_height"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)
    print(f"[OUTPUT] Tersimpan: {img_csv_path} ({len(image_rows):,} citra)")

    # 5. Hitung Statistik Diagnostik & Sanity Gate
    d_vals = sorted(float(r["d_image"]) for r in image_rows)
    p_vals = [float(r["p_sampling"]) for r in image_rows]

    # Effective Sample Size (ESS)
    sum_p2 = sum(p ** 2 for p in p_vals)
    ess = 1.0 / sum_p2 if sum_p2 > 0 else 0.0
    ess_ratio = ess / n_images

    # Percentiles
    def get_percentile(data, pct):
        k = (len(data) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1

    p10 = get_percentile(d_vals, 10)
    p25 = get_percentile(d_vals, 25)
    p50 = get_percentile(d_vals, 50)
    p75 = get_percentile(d_vals, 75)
    p90 = get_percentile(d_vals, 90)
    mean_d = sum(d_vals) / len(d_vals)
    dynamic_range = p90 - p10

    hard_count = sum(1 for d in d_vals if d >= 0.80)
    hard_ratio = hard_count / len(d_vals)

    clipping_rate = (wrong_class_clipped / wrong_class_total) if wrong_class_total > 0 else 0.0

    if hard_ratio <= 0.60:
        gate_status = "PASS"
    elif hard_ratio <= 0.80:
        gate_status = "REVIEW"
    else:
        gate_status = "FAIL"

    gate_results = {
        "gate_status": gate_status,
        "n_images": n_images,
        "total_gt_objects": total_gt,
        "taxonomy_breakdown": dict(taxonomy_counter),
        "d_image_stats": {
            "min": min(d_vals),
            "p10": p10,
            "p25": p25,
            "median_p50": p50,
            "p75": p75,
            "p90": p90,
            "max": max(d_vals),
            "mean": mean_d,
            "dynamic_range_p90_p10": dynamic_range,
        },
        "sampling_diagnostics": {
            "effective_sample_size_ess": ess,
            "ess_ratio": ess_ratio,
            "diagnostic_target_range": "[0.65, 0.95]",
        },
        "clipping_audit": {
            "wrong_class_total": wrong_class_total,
            "wrong_class_clipped_at_1_0": wrong_class_clipped,
            "clipping_rate": clipping_rate,
        },
        "hard_image_audit": {
            "hard_image_count_ge_0_8": hard_count,
            "hard_image_ratio": hard_ratio,
            "criteria": "<=60% PASS, 60-80% REVIEW, >80% FAIL",
        },
    }

    gate_json_path = output_dir / "difficulty_sanity_gate_report.json"
    with open(gate_json_path, "w", encoding="utf-8") as f:
        json.dump(gate_results, f, indent=2)
    print(f"[OUTPUT] Tersimpan: {gate_json_path}")

    # Laporan Markdown
    report_md_path = output_dir / "DIFFICULTY_PROFILING_REPORT.md"
    report_md = f"""# Laporan Hasil Stage 8.2 — Profil Kesulitan Objek & Citra OOF (Hardened)
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Status Sanity Gate
- **Status Akhir**: **{gate_status}**
- **Proporsi Citra Hard ($D_\\text{{image}} \\ge 0.80$)**: {hard_ratio * 100:.2f}% (Kriteria: $\\le 60\\%$ PASS, $60-80\\%$ REVIEW, $>80\\%$ FAIL)
- **Dynamic Range ($P90 - P10$)**: {dynamic_range:.4f} (Kriteria diagnostik: $\\ge 0.30$)
- **Effective Sample Size (ESS)**: {ess:.1f} / {n_images} ($ESS_\\text{{ratio}} = {ess_ratio:.4f}$)
- **Clipping Rate ($D_i = 1.0$ pada Wrong Class)**: {clipping_rate * 100:.2f}% ({wrong_class_clipped}/{wrong_class_total})

## 2. Taksonomi Error Ground Truth (OOF Validation Matching)
| Taksonomi Error | Jumlah Objek | Persentase |
| :--- | :---: | :---: |
| True Positive (TP) | {taxonomy_counter['True_Positive']:,} | {taxonomy_counter['True_Positive'] / total_gt * 100:.2f}% |
| Localization Error | {taxonomy_counter['Localization_Error']:,} | {taxonomy_counter['Localization_Error'] / total_gt * 100:.2f}% |
| Classification Confusion | {taxonomy_counter['Classification_Confusion']:,} | {taxonomy_counter['Classification_Confusion'] / total_gt * 100:.2f}% |
| Localization and Class Error | {taxonomy_counter['Localization_and_Class_Error']:,} | {taxonomy_counter['Localization_and_Class_Error'] / total_gt * 100:.2f}% |
| False Negative (Missed) | {taxonomy_counter['False_Negative_Missed']:,} | {taxonomy_counter['False_Negative_Missed'] / total_gt * 100:.2f}% |
| **Total Ground Truth** | **{total_gt:,}** | **100.00%** |

## 3. Distribusi Kuantil Kesulitan Citra ($D_\\text{{image}}$)
- **Min**: {min(d_vals):.4f}
- **P10**: {p10:.4f}
- **P25**: {p25:.4f}
- **Median (P50)**: {p50:.4f}
- **P75**: {p75:.4f}
- **P90**: {p90:.4f}
- **Max**: {max(d_vals):.4f}
- **Mean $\\pm$ Std**: {mean_d:.4f}

## 4. Kesiapan Menuju Stage 8.3 & 8.4
Manifest `train_image_difficulty_manifest.csv` telah memuat probabilitas sampling $P(k)$ terkalibrasi
dan identitas lesi target $i^*$ untuk augmentasi on-the-fly.
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OUTPUT] Tersimpan: {report_md_path}")

    print("\n" + "=" * 80)
    print(f"SANITY GATE RESULT: {gate_status} (Hard Ratio: {hard_ratio * 100:.2f}%, ESS Ratio: {ess_ratio:.4f})")
    print("=" * 80)


def main():
    args = parse_args()
    print("=" * 80)
    print("STAGE 8.2 — BUILD DIFFICULTY MANIFEST & SANITY GATE (HARDENED)")
    print("=" * 80)
    print(f"OOF Predictions : {Path(args.oof_dir) / 'oof_predictions_raw.csv'}")
    print(f"Data YAML       : {args.data_yaml}")
    print(f"Dataset Dir     : {args.dataset_dir or args.research_dir}")
    print(f"Output Dir      : {args.output_dir}")
    print(f"IoU Match Thresh: {args.iou_match_thresh}")
    print(f"Confusion Pen.  : {args.confusion_penalty}")
    print(f"Mixture Lambda  : {args.mixture_lambda}")
    print("=" * 80)

    raw_pred_file = Path(args.oof_dir) / "oof_predictions_raw.csv"
    if not raw_pred_file.exists():
        print(f"ERROR: File {raw_pred_file} belum ditemukan. Pastikan Stage 8.1 telah selesai!")
        sys.exit(1)

    images_dir, labels_dir = resolve_train_paths(args)
    gt_by_image = load_ground_truth(images_dir, labels_dir)
    preds_by_image = load_oof_predictions(raw_pred_file)

    process_difficulty_manifest(
        gt_by_image=gt_by_image,
        preds_by_image=preds_by_image,
        output_dir=Path(args.output_dir).resolve(),
        args=args,
    )


if __name__ == "__main__":
    main()

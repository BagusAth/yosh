#!/usr/bin/env python3
"""
Stage 8.2 — Build Object & Image Difficulty Manifest (Post-OOF)
==============================================================

Kontrak Metodologis:
1. Membaca prediksi OOF mentah dari stage_8_1_oof/oof_predictions_raw.csv (conf >= 0.001).
2. Membaca ground truth langsung dari file label YOLO (8.980 citra train).
3. Melakukan class-agnostic greedy one-to-one matching pada threshold IoU >= 0.10.
4. Menetapkan taksonomi error (TP, Localization, Confusion, Loc+Class, FN, FP).
5. Menghitung kesulitan objek (D_i):
   - Matched + Correct Class : 0.40(1 - C_i) + 0.40(1 - IoU_i) + 0.20 S_i
   - Matched + Wrong Class   : min(1.0, 0.40(1 - C_i) + 0.40(1 - IoU_i) + 0.20 S_i + 0.35)
   - Unmatched (FN)          : 0.80 + 0.20 S_i (design convention)
6. Menghitung kesulitan citra hybrid (D_image):
   - Citra berobjek: 0.5 * max(D_i) + 0.5 * mean(D_i)
   - Citra kosong: max(C_FP) jika ada FP, 0.0 jika bersih
7. Menghitung probabilitas sampling mixture:
   P(k) = 0.3 * (1/N) + 0.7 * (D_image / sum(D_image))
8. Mengevaluasi Sanity Gate:
   - Hard ratio (D_image >= 0.80): <=60% PASS, 60-80% REVIEW, >80% FAIL
   - Dynamic Range: P90 - P10 >= 0.30 (diagnostic criterion)
   - Effective Sample Size (ESS) dan ESS_ratio in [0.65, 0.95] (diagnostic range)
   - Clipping rate pada wrong-class cases
9. Mengekspor manifest kesulitan citra untuk digunakan pada Stage 8.3 & 8.4.
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

import numpy as np

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 8.2: Build Difficulty Manifest")
    parser.add_argument(
        "--oof-dir",
        default="d:/UNDIP/SKRIPSI/stage_8_1_oof",
        help="Folder output dari Stage 8.1",
    )
    parser.add_argument(
        "--research-dir",
        default="d:/UNDIP/SKRIPSI/cassava-v7-research",
        help="Path ke dataset resmi cassava-v7-research",
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


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Menghitung Intersection over Union (IoU) antara dua box format [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def load_ground_truth(train_dir: Path):
    """Membaca seluruh ground truth objek dari file label .txt (8.980 citra)."""
    labels_dir = train_dir / "labels"
    images_dir = train_dir / "images"
    image_files = sorted(os.listdir(images_dir))

    gt_by_image = {}
    total_gt_objects = 0

    for fname in image_files:
        stem = Path(fname).stem
        lbl_file = labels_dir / f"{stem}.txt"

        objects = []
        if lbl_file.exists() and os.path.getsize(lbl_file) > 0:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for obj_id, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if len(parts) == 5:
                            xc = float(parts[1]) * 640.0
                            yc = float(parts[2]) * 640.0
                            w = float(parts[3]) * 640.0
                            h = float(parts[4]) * 640.0
                        else:
                            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                            x_min = float(np.min(coords[:, 0])) * 640.0
                            y_min = float(np.min(coords[:, 1])) * 640.0
                            x_max = float(np.max(coords[:, 0])) * 640.0
                            y_max = float(np.max(coords[:, 1])) * 640.0
                            xc = (x_min + x_max) / 2.0
                            yc = (y_min + y_max) / 2.0
                            w = max(0.0, x_max - x_min)
                            h = max(0.0, y_max - y_min)

                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(640.0, xc + w / 2.0)
                        y2 = min(640.0, yc + h / 2.0)

                        area_px = w * h
                        # Size difficulty score berbasis kategori
                        if area_px < (32.0 * 32.0):
                            size_cat = "Small"
                            size_score = 1.0
                        elif area_px < (96.0 * 96.0):
                            size_cat = "Medium"
                            size_score = 0.5
                        else:
                            size_cat = "Large"
                            size_score = 0.0

                        objects.append({
                            "gt_id": obj_id,
                            "class_id": cls_id,
                            "class_name": CLASS_NAMES[cls_id],
                            "box_xyxy": [x1, y1, x2, y2],
                            "area_px": area_px,
                            "size_cat": size_cat,
                            "size_score": size_score,
                        })
                        total_gt_objects += 1

        gt_by_image[fname] = objects

    print(f"[GROUND TRUTH] Memuat {len(gt_by_image):,} citra dengan total {total_gt_objects:,} objek ground truth.")
    return gt_by_image


def load_oof_predictions(raw_pred_file: Path):
    """Membaca prediksi OOF mentah hasil inferensi diagnostik @ conf = 0.001."""
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


def greedy_one_to_one_matching(gt_objects: list[dict], pred_objects: list[dict], match_thresh: float):
    """
    Algoritma Greedy Global Bipartite Matching:
    - Class-agnostic
    - Threshold minimum IoU >= match_thresh
    - One-to-one: setiap GT maksimal 1 prediksi, setiap prediksi maksimal 1 GT
    """
    if not gt_objects or not pred_objects:
        return {}, set(range(len(pred_objects)))

    # 1. Kumpulkan seluruh pasangan kandidat dengan IoU >= match_thresh
    candidate_pairs = []
    for g_idx, g in enumerate(gt_objects):
        for p_idx, p in enumerate(pred_objects):
            iou_val = compute_iou(g["box_xyxy"], p["box_xyxy"])
            if iou_val >= match_thresh:
                candidate_pairs.append((iou_val, g_idx, p_idx))

    # 2. Urutkan descending berdasarkan IoU
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    # 3. Penugasan serakah satu-ke-satu
    assigned_gt = set()
    assigned_pred = set()
    matches = {}  # g_idx -> (p_idx, iou_val)

    for iou_val, g_idx, p_idx in candidate_pairs:
        if g_idx not in assigned_gt and p_idx not in assigned_pred:
            assigned_gt.add(g_idx)
            assigned_pred.add(p_idx)
            matches[g_idx] = (p_idx, iou_val)

    unassigned_preds = set(range(len(pred_objects))) - assigned_pred
    return matches, unassigned_preds


def process_difficulty_manifest(
    gt_by_image: dict,
    preds_by_image: dict,
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

    all_d_images = []

    for fname, gts in sorted(gt_by_image.items()):
        preds = preds_by_image.get(fname, [])
        n_gt = len(gts)

        matches, unassigned_preds = greedy_one_to_one_matching(
            gts, preds, match_thresh=args.iou_match_thresh
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
                        # Formula Matched & Kelas Benar
                        d_i = 0.40 * (1.0 - conf) + 0.40 * (1.0 - iou_val) + 0.20 * size_score
                    else:
                        if iou_val >= 0.50:
                            taxonomy = "Classification_Confusion"
                        else:
                            taxonomy = "Localization_and_Class_Error"

                        # Formula Matched & Salah Kelas (Design Penalty 0.35)
                        raw_d_i = 0.40 * (1.0 - conf) + 0.40 * (1.0 - iou_val) + 0.20 * size_score + args.confusion_penalty
                        d_i = min(1.0, raw_d_i)

                        wrong_class_total += 1
                        if raw_d_i >= 1.0:
                            wrong_class_clipped += 1

                    match_status = "Matched"
                    matched_conf = conf
                    matched_iou = iou_val
                    pred_class = p_cls
                else:
                    taxonomy = "False_Negative_Missed"
                    match_status = "Unmatched"
                    matched_conf = 0.0
                    matched_iou = 0.0
                    pred_class = -1
                    # Formula Unmatched (FN Penalty Design Convention)
                    d_i = 0.80 + 0.20 * size_score

                taxonomy_counter[taxonomy] += 1
                obj_difficulties.append(d_i)

                # Evaluasi lesi small yang sulit (target potensial context crop)
                if g["size_cat"] == "Small" and d_i >= 0.70:
                    target_small_candidates.append({
                        "gt_id": g["gt_id"],
                        "d_i": d_i,
                        "box": g["box_xyxy"],
                        "class_id": gt_cls,
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

            # Agregasi Citra Berobjek: 0.5 * max + 0.5 * mean
            d_max = max(obj_difficulties)
            d_mean = sum(obj_difficulties) / len(obj_difficulties)
            d_image = 0.5 * d_max + 0.5 * d_mean
            has_gt = 1
        else:
            # Citra tanpa GT (empty / clean background)
            has_gt = 0
            if unassigned_preds:
                # Memunculkan false positive
                max_fp_conf = max(preds[p_idx]["confidence"] for p_idx in unassigned_preds)
                d_image = max_fp_conf
            else:
                d_image = 0.0

        all_d_images.append((fname, d_image, has_gt, target_small_candidates))

    # 4. Hitung Probabilitas Mixture Sampling: P(k) = 0.3 * (1/N) + 0.7 * (D_k / sum(D))
    n_images = len(all_d_images)
    sum_d = sum(d for _, d, _, _ in all_d_images)

    for fname, d_image, has_gt, target_smalls in all_d_images:
        p_uniform = 0.30 * (1.0 / n_images)
        p_adaptive = 0.70 * (d_image / sum_d) if sum_d > 0 else 0.0
        p_mixture = p_uniform + p_adaptive

        # Pilih target tunggal small lesion dengan difficulty tertinggi (i*)
        if target_smalls:
            best_small = max(target_smalls, key=lambda x: x["d_i"])
            target_small_exists = 1
            target_gt_id = best_small["gt_id"]
            target_box = ",".join(f"{v:.1f}" for v in best_small["box"])
            target_d_i = f"{best_small['d_i']:.4f}"
        else:
            target_small_exists = 0
            target_gt_id = -1
            target_box = ""
            target_d_i = "0.0000"

        image_rows.append({
            "image_id": fname,
            "d_image": f"{d_image:.6f}",
            "p_sampling": f"{p_mixture:.8e}",
            "has_gt": has_gt,
            "has_target_small": target_small_exists,
            "target_gt_id": target_gt_id,
            "target_small_di": target_d_i,
            "target_small_box": target_box,
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
            "target_gt_id", "target_small_di", "target_small_box"
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

    # Kuantil D_image
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

    # Evaluasi Status Sanity Gate
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
    report_md = f"""# Laporan Hasil Stage 8.2 — Profil Kesulitan Objek & Citra OOF
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
    print("STAGE 8.2 — BUILD DIFFICULTY MANIFEST & SANITY GATE")
    print("=" * 80)
    print(f"OOF Predictions : {Path(args.oof_dir) / 'oof_predictions_raw.csv'}")
    print(f"Research Dir    : {args.research_dir}")
    print(f"Output Dir      : {args.output_dir}")
    print(f"IoU Match Thresh: {args.iou_match_thresh}")
    print(f"Confusion Pen.  : {args.confusion_penalty}")
    print(f"Mixture Lambda  : {args.mixture_lambda}")
    print("=" * 80)

    raw_pred_file = Path(args.oof_dir) / "oof_predictions_raw.csv"
    if not raw_pred_file.exists():
        print(f"ERROR: File {raw_pred_file} belum ditemukan. Pastikan Stage 8.1 telah selesai!")
        sys.exit(1)

    research_dir = Path(args.research_dir).resolve()
    gt_by_image = load_ground_truth(research_dir / "train")
    preds_by_image = load_oof_predictions(raw_pred_file)

    process_difficulty_manifest(
        gt_by_image=gt_by_image,
        preds_by_image=preds_by_image,
        output_dir=Path(args.output_dir).resolve(),
        args=args,
    )


if __name__ == "__main__":
    main()

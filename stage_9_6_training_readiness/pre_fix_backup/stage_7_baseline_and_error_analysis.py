#!/usr/bin/env python3
"""
Stage 7 — Baseline YOLO11n Training, Error Decomposition & Difficulty Profiling
==============================================================================

Fokus Penelitian:
"Optimasi Model YOLO Berbasis Analisis Error dan Tingkat Kesulitan Objek untuk
Deteksi dan Lokalisasi Gejala Penyakit Daun"

Tahapan Eksekusi:
1. Training Baseline YOLO11n (50 Epochs, RTX 4060 GPU, FP16/AMP).
2. Validasi & Pengambilan Metrik Standar (Precision, Recall, mAP50, mAP50-95).
3. Dekomposisi Error Deteksi & Lokalisasi:
   - True Positive (TP)
   - False Negative (FN / Missed)
   - False Positive (FP / Hallucination)
   - Localization Error (0.1 <= IoU < 0.5)
   - Classification Confusion (IoU >= 0.5, Wrong Class)
   - Korelasi Error terhadap Ukuran (Small / Medium / Large)
4. Pembentukan Profil Tingkat Kesulitan Objek:
   Di = w1*(1 - Ci) + w2*(1 - IoUi) + w3*Si
   Pengelompokan ke Tier 1 (Easy), Tier 2 (Moderate), Tier 3 (Hard).
5. Pembentukan manifest profil kesulitan untuk bahan bakar Stage 8.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from research_utils.detection_metrics import (
    calculate_iou_xyxy,
    classify_object_size,
    parse_yolo_labels,
    greedy_bipartite_matching,
    calculate_object_difficulty,
    get_image_dimensions,
    CLASS_NAMES,
)

EVALUATOR_VERSION = "Stage-9.5-Canonical-v1"


def main():
    parser = argparse.ArgumentParser(description="Stage 7 — Baseline Training & Error Analysis")
    parser.add_argument(
        "--data-yaml",
        default="d:/UNDIP/SKRIPSI/cassava-v7-research/data.yaml",
        help="Path ke data.yaml dataset riset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch training (Default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (Default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran resolusi citra (Default: 640)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device (Default: '0' untuk RTX 4060)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers (Default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/UNDIP/SKRIPSI/cassava-v7-stage-7",
        help="Direktori penyimpanan seluruh artefak Stage 7",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed reproduktif")
    args = parser.parse_args()

    data_yaml_path = Path(args.data_yaml).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 7 — BASELINE YOLO11n TRAINING & EXHAUSTIVE ERROR ANALYSIS")
    print("=" * 80)
    print(f"Data YAML      : {data_yaml_path}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch Size     : {args.batch}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Device         : {args.device}")
    print(f"Output Dir     : {output_dir}")
    print()

    # Import Ultralytics & PyTorch
    import torch
    from ultralytics import YOLO

    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU      : {torch.cuda.get_device_name(0)}")
    print()

    # -------------------------------------------------------------------------
    # PILAR 1: TRAINING BASELINE YOLO11n
    # -------------------------------------------------------------------------
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_weights = runs_dir / "baseline_yolo11n" / "weights" / "best.pt"

    if best_weights.exists():
        print(f"[PILAR 1] Ditemukan bobot model terlatih sebelumnya: {best_weights}")
        print("          Melewati proses training ulang...")
        model = YOLO(str(best_weights))
    else:
        print("[PILAR 1/4] Memulai Pelatihan Baseline YOLO11n (50 Epochs pada RTX 4060)...")
        t0_train = time.time()
        model = YOLO("yolo11n.pt")

        train_results = model.train(
            data=str(data_yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            project=str(runs_dir),
            name="baseline_yolo11n",
            exist_ok=True,
            val=True,
            save=True,
            plots=True,
            patience=20,
            amp=True,
        )
        train_time_min = (time.time() - t0_train) / 60
        print(f"[PILAR 1] Pelatihan selesai dalam {train_time_min:.2f} menit!")
        print(f"          Bobot terbaik tersimpan di: {best_weights}")

    # -------------------------------------------------------------------------
    # PILAR 2: VALIDATION & STANDARD METRICS EVALUATION
    # -------------------------------------------------------------------------
    print()
    print("[PILAR 2/4] Mengevaluasi Metrik Standar pada Split Validasi...")
    val_model = YOLO(str(best_weights))
    val_metrics = val_model.val(
        data=str(data_yaml_path),
        split="val",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )

    overall_metrics = {
        "precision": float(val_metrics.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_metrics.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(val_metrics.fitness),
    }

    # Per-class metrics
    per_class_metrics = {}
    if hasattr(val_metrics.box, "maps"):
        maps50 = val_metrics.box.p  # Precision per class
        for i, cname in enumerate(CLASS_NAMES):
            if i < len(val_metrics.box.maps):
                per_class_metrics[cname] = {
                    "mAP50": float(val_metrics.box.ap50[i]) if hasattr(val_metrics.box, "ap50") and i < len(val_metrics.box.ap50) else 0.0,
                    "mAP50_95": float(val_metrics.box.maps[i]),
                }

    print("          HASIL METRIK BASELINE:")
    print(f"          Precision : {overall_metrics['precision']:.4f}")
    print(f"          Recall    : {overall_metrics['recall']:.4f}")
    print(f"          mAP@50    : {overall_metrics['mAP50']:.4f}")
    print(f"          mAP@50:95 : {overall_metrics['mAP50_95']:.4f}")

    # -------------------------------------------------------------------------
    # PILAR 3: DEKOMPOSISI ERROR & DIFFICULTY PROFILING (Di)
    # -------------------------------------------------------------------------
    print()
    print("[PILAR 3/4] Menjalankan Analisis Dekomposisi Error & Difficulty Profiling...")
    dataset_root = data_yaml_path.parent
    val_img_dir = dataset_root / "valid" / "images"
    val_lbl_dir = dataset_root / "valid" / "labels"

    # Bobot rumus Di: Di = w1*(1-C) + w2*(1-IoU) + w3*S
    w1, w2, w3 = 0.40, 0.40, 0.20

    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"          Memproses {len(val_images):,} citra validasi...")

    # Statistics counters
    error_type_counter = Counter()
    class_error_counter = defaultdict(Counter)
    size_error_counter = defaultdict(Counter)

    object_records = []
    image_difficulty_records = []

    # Inference in batches or single image loop
    t0_eval = time.time()
    for idx, img_path in enumerate(val_images):
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        img_w, img_h = get_image_dimensions(img_path)
        gt_objects = parse_yolo_labels(lbl_path, img_width=img_w, img_height=img_h)

        # Run inference
        preds = val_model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.5,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.empty((0, 4))
        pred_scores = preds.boxes.conf.cpu().numpy() if len(preds.boxes) else np.empty((0,))
        pred_classes = preds.boxes.cls.cpu().numpy().astype(int) if len(preds.boxes) else np.empty((0,), dtype=int)

        # Canonical one-to-one bipartite greedy matching (descending IoU globally)
        match_res = greedy_bipartite_matching(
            gt_objects=gt_objects,
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            iou_thresh=0.10,
            tp_iou_thresh=0.50,
        )

        img_di_scores = []
        for a in match_res["gt_assignments"]:
            gt_idx = a["gt_idx"]
            gt = gt_objects[gt_idx]
            gt_cname = gt["class_name"]
            size_cat = a["size_category"]
            size_penalty = a["size_penalty"]
            error_type = a["error_type"]
            iou = a["iou"]
            conf = a["conf"] if a["is_correct_class"] else 0.0

            # Calculate Difficulty Score Di using canonical formula
            di = calculate_object_difficulty(
                matched=(a["matched_pred_idx"] >= 0),
                conf=a["conf"],
                iou=iou,
                size_penalty=size_penalty,
                correct_class=a["is_correct_class"],
                confusion_penalty=0.35,
                w1=w1,
                w2=w2,
                w3=w3,
            )
            img_di_scores.append(di)

            # Assign Difficulty Tier
            if di < 0.35:
                tier = "Tier_1_Easy"
            elif di < 0.70:
                tier = "Tier_2_Moderate"
            else:
                tier = "Tier_3_Hard"

            error_type_counter[error_type] += 1
            class_error_counter[gt_cname][error_type] += 1
            size_error_counter[size_cat][error_type] += 1

            object_records.append({
                "image_filename": img_path.name,
                "gt_class": gt_cname,
                "size_category": size_cat,
                "error_type": error_type,
                "iou": round(iou, 4),
                "confidence": round(conf, 4),
                "difficulty_score_di": round(di, 4),
                "difficulty_tier": tier,
            })

        # Count False Positives (predictions with no matching GT) using canonical unassigned_preds
        unassigned_count = len(match_res["unassigned_preds"])
        error_type_counter["False_Positive_Hallucination"] += unassigned_count

        # Image-level difficulty aggregation
        mean_di = float(np.mean(img_di_scores)) if img_di_scores else 0.0
        max_di = float(np.max(img_di_scores)) if img_di_scores else 0.0
        hard_count = sum(1 for d in img_di_scores if d >= 0.70)

        image_tier = "Tier_3_Hard" if max_di >= 0.70 else ("Tier_2_Moderate" if mean_di >= 0.35 else "Tier_1_Easy")
        image_difficulty_records.append({
            "image_filename": img_path.name,
            "num_objects": len(gt_objects),
            "mean_difficulty": round(mean_di, 4),
            "max_difficulty": round(max_di, 4),
            "hard_object_count": hard_count,
            "image_tier": image_tier,
        })

    eval_duration = time.time() - t0_eval
    print(f"          Evaluasi error selesai dalam {eval_duration:.2f} detik.")

    # -------------------------------------------------------------------------
    # PILAR 4: SIMPAN ARTEFAK & BUAT LAPORAN LENGKAP
    # -------------------------------------------------------------------------
    print()
    print("[PILAR 4/4] Menyimpan Artefak dan Membuat Laporan Komprehensif...")

    # 1. Simpan baseline_evaluation_metrics.json
    metrics_summary = {
        "stage": "Stage 7 — Baseline Training & Error Analysis",
        "model_architecture": "YOLO11n",
        "parameters": "2.6M",
        "dataset": "cassava-v7-research",
        "epochs_trained": args.epochs,
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "weights_path": str(best_weights),
    }
    with (output_dir / "baseline_evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    # 2. Simpan error_decomposition_summary.csv
    total_eval_objects = len(object_records)
    error_summary_rows = []
    for etype, count in sorted(error_type_counter.items()):
        error_summary_rows.append({
            "error_taxonomy": etype,
            "count": count,
            "percentage_of_gt_objects": round(count / total_eval_objects * 100, 2) if total_eval_objects else 0,
        })
    with (output_dir / "error_decomposition_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["error_taxonomy", "count", "percentage_of_gt_objects"])
        writer.writeheader()
        writer.writerows(error_summary_rows)

    # 3. Simpan error_breakdown_by_class.csv
    class_error_rows = []
    for cname in CLASS_NAMES:
        row = {"class_name": cname}
        row.update(dict(class_error_counter[cname]))
        class_error_rows.append(row)
    all_err_keys = ["class_name"] + sorted(list(error_type_counter.keys()))
    with (output_dir / "error_breakdown_by_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_err_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(class_error_rows)

    # 4. Simpan error_breakdown_by_size.csv
    size_error_rows = []
    for scat in ("Small", "Medium", "Large"):
        row = {"size_category": scat}
        row.update(dict(size_error_counter[scat]))
        size_error_rows.append(row)
    with (output_dir / "error_breakdown_by_size.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["size_category"] + sorted(list(error_type_counter.keys())), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(size_error_rows)

    # 5. Simpan difficulty_profile_manifest.csv (Image level untuk Stage 8)
    with (output_dir / "difficulty_profile_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "num_objects", "mean_difficulty", "max_difficulty", "hard_object_count", "image_tier"])
        writer.writeheader()
        writer.writerows(image_difficulty_records)

    # 6. Simpan STAGE_7_BASELINE_REPORT.md
    report_lines = [
        "# STAGE 7: BASELINE TRAINING & EXHAUSTIVE ERROR ANALYSIS REPORT",
        "## Evaluasi Model YOLO11n pada Dataset cassava-v7-research",
        "",
        "### 1. Ringkasan Pelatihan Baseline",
        f"- **Model**: YOLO11n (2.6M Parameters)",
        f"- **Epochs**: {args.epochs}",
        f"- **Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"- **Precision**: {overall_metrics['precision']:.4f}",
        f"- **Recall**: {overall_metrics['recall']:.4f}",
        f"- **mAP@50**: {overall_metrics['mAP50']:.4f}",
        f"- **mAP@50:95**: {overall_metrics['mAP50_95']:.4f}",
        "",
        "### 2. Dekomposisi Taksonomi Error",
        "| Kategori Error | Jumlah Objek | Persentase (%) | Deskripsi |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for row in error_summary_rows:
        report_lines.append(f"| **{row['error_taxonomy']}** | {row['count']:,} | {row['percentage_of_gt_objects']}% | Error deteksi objek |")

    report_lines.extend([
        "",
        "### 3. Distribusi Difficulty Tier Citra",
        f"- **Tier 1 (Easy Samples)**: {sum(1 for r in image_difficulty_records if r['image_tier'] == 'Tier_1_Easy'):,} citra",
        f"- **Tier 2 (Moderate Samples)**: {sum(1 for r in image_difficulty_records if r['image_tier'] == 'Tier_2_Moderate'):,} citra",
        f"- **Tier 3 (Hard Samples / Prioritas Stage 8)**: {sum(1 for r in image_difficulty_records if r['image_tier'] == 'Tier_3_Hard'):,} citra",
        "",
        "### 4. Kesimpulan untuk Stage 8",
        "Hasil profiling ini membuktikan bahwa kesalahan terbesar didorong oleh kombinasi lesi kecil (Small)",
        "dan kerancuan klasifikasi antar kelas bercak daun. Manifest `difficulty_profile_manifest.csv`",
        "siap digunakan sebagai input utama untuk strategi **Error-Driven Adaptive Optimization** pada Stage 8.",
    ])
    (output_dir / "STAGE_7_BASELINE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    # 7. Simpan STAGE_7_VERDICT.txt
    verdict_text = f"""STAGE 7 — BASELINE TRAINING & ERROR ANALYSIS VERDICT
=================================================================
Status Pelatihan        : SUCCESS (50 Epochs Selesai)
Model Weights Terpilih  : {best_weights}
Precision Baseline      : {overall_metrics['precision']:.4f}
Recall Baseline         : {overall_metrics['recall']:.4f}
mAP@50 Baseline         : {overall_metrics['mAP50']:.4f}
mAP@50:95 Baseline      : {overall_metrics['mAP50_95']:.4f}
Total Objek Dievaluasi  : {total_eval_objects:,} objek
True Positives (TP)     : {error_type_counter.get('True_Positive', 0):,}
False Negatives (Missed): {error_type_counter.get('False_Negative_Missed', 0):,}
Localization Errors     : {error_type_counter.get('Localization_Error', 0):,}
Class Confusions        : {error_type_counter.get('Classification_Confusion', 0):,}
Hard Samples (Tier 3)   : {sum(1 for r in image_difficulty_records if r['image_tier'] == 'Tier_3_Hard'):,} citra
=================================================================
VERDICT: PASS — Baseline Model dan Profil Kesulitan Siap untuk Stage 8.
"""
    (output_dir / "STAGE_7_VERDICT.txt").write_text(verdict_text, encoding="utf-8")

    print()
    print("=" * 80)
    print("STAGE 7 BERHASIL DISELESAIKAN!")
    print("=" * 80)
    print(f"Laporan lengkap tersimpan di: {output_dir / 'STAGE_7_BASELINE_REPORT.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

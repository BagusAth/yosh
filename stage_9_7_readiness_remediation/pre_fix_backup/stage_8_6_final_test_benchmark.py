#!/usr/bin/env python3
"""
Stage 8.6 — One-Door Locked Test Set Final Benchmark (Hardened)
==============================================================

Tujuan Metodologis:
1. Membuka pintu evaluasi akhir pada Locked Test Set (1.282 citra) hanya dengan konfirmasi eksplisit.
2. Mengevaluasi varian eksperimen ablasi secara objektif tanpa data snooping.
3. Menggunakan evaluasi kanonis terpadu dari research_utils.detection_metrics:
   - extract_canonical_metrics (pemisahan kanonis AP@50 dan AP@50-95 per kelas)
   - greedy_bipartite_matching (Global 1-to-1 bipartite greedy matching pada IoU >= 0.10, zero reuse)
   - classify_object_size (Small < 32^2, Medium 32^2-96^2, Large >= 96^2)
   - parse_yolo_labels (dimensi dinamis)
4. Menghitung dekomposisi error berstrata ukuran: Small, Medium, Large (GT, TP, Recall).
5. Single Source of Truth: Menentukan direktori test images dan labels secara dinamis dari --data-yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from research_utils.detection_metrics import (
    classify_object_size,
    extract_canonical_metrics,
    get_image_dimensions,
    greedy_bipartite_matching,
    parse_yolo_labels,
    resolve_data_yaml_split,
)

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


def resolve_test_paths(data_yaml_path: str | Path) -> tuple[Path, Path]:
    """Compatibility wrapper around resolve_data_yaml_split."""
    im_dir, lb_dir, _ = resolve_data_yaml_split(data_yaml_path, "test")
    return im_dir, lb_dir


def evaluate_test_set(
    model_path: Path,
    data_yaml_path: Path,
    test_images: list[Path] | Path,
    test_lbl_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    chunk_size: int = 32,
) -> dict[str, Any]:
    """Mengevaluasi model pada test set via Ultralytics val() kanonis dan greedy bipartite matching."""
    model = YOLO(str(model_path))

    if isinstance(test_images, (str, Path)) and Path(test_images).is_dir():
        t_img_p = Path(test_images)
        test_images_list = sorted([f for f in t_img_p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    else:
        test_images_list = list(test_images)

    # 1. Ultralytics official test split evaluation
    val_res = model.val(
        data=str(data_yaml_path),
        split="test",
        batch=16,
        imgsz=640,
        device="0" if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False,
    )

    # Canonical metrics extraction
    canonical_m = extract_canonical_metrics(val_res, CLASS_NAMES)

    # 2. Granular error decomposition on Test Set
    total_gt = 0
    gt_by_size = Counter()
    tp_by_size = Counter()
    taxonomy_counts = Counter()

    gt_dict = {}
    for img_path in test_images_list:
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        img_w, img_h = get_image_dimensions(img_path)
        gt_objs = parse_yolo_labels(lbl_path, img_width=img_w, img_height=img_h)
        gt_dict[img_path.name] = gt_objs
        for gt in gt_objs:
            total_gt += 1
            gt_by_size[gt["size_category"]] += 1

    for i in range(0, len(test_images_list), chunk_size):
        chunk = test_images_list[i : i + chunk_size]
        chunk_paths = [str(p) for p in chunk]

        preds = model.predict(
            source=chunk_paths,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=640,
            device="0" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        for img_path, pred_res in zip(chunk, preds):
            gt_objs = gt_dict[img_path.name]
            pred_boxes = []
            pred_cls = []

            if pred_res.boxes is not None and len(pred_res.boxes) > 0:
                pred_boxes = pred_res.boxes.xyxy.cpu().numpy()
                pred_cls = pred_res.boxes.cls.cpu().numpy().astype(int)

            gt_boxes = [g["box"] for g in gt_objs]
            # Canonical greedy bipartite matching from research_utils
            matches, unassigned_preds = greedy_bipartite_matching(
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                iou_thresh=0.10,
            )

            for g_idx, gt in enumerate(gt_objs):
                gt_cid = gt["class_id"]
                gt_size = gt["size_category"]

                if g_idx in matches:
                    p_idx, iou_val = matches[g_idx]
                    p_cid = pred_cls[p_idx]
                    is_correct_class = (p_cid == gt_cid)

                    if iou_val >= 0.50:
                        if is_correct_class:
                            taxonomy_counts["True_Positive"] += 1
                            tp_by_size[gt_size] += 1
                        else:
                            taxonomy_counts["Classification_Confusion"] += 1
                    else:
                        if is_correct_class:
                            taxonomy_counts["Localization_Error"] += 1
                        else:
                            taxonomy_counts["Localization_and_Class_Error"] += 1
                else:
                    taxonomy_counts["False_Negative_Missed"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    recall_small = (tp_by_size["Small"] / gt_by_size["Small"]) if gt_by_size["Small"] > 0 else 0.0
    recall_med = (tp_by_size["Medium"] / gt_by_size["Medium"]) if gt_by_size["Medium"] > 0 else 0.0
    recall_large = (tp_by_size["Large"] / gt_by_size["Large"]) if gt_by_size["Large"] > 0 else 0.0

    return {
        "precision": canonical_m["precision"],
        "recall": canonical_m["recall"],
        "map50": canonical_m["map50"],
        "map50_95": canonical_m["map50_95"],
        "per_class_ap50": canonical_m["per_class_ap50"],
        "per_class_map50_95": canonical_m["per_class_map50_95"],
        "per_class_map50": canonical_m["per_class_ap50"],
        "total_gt": total_gt,
        "gt_by_size": dict(gt_by_size),
        "tp_by_size": dict(tp_by_size),
        "recall_small": round(recall_small, 4),
        "recall_medium": round(recall_med, 4),
        "recall_large": round(recall_large, 4),
        "fn_missed": taxonomy_counts.get("False_Negative_Missed", 0),
        "cls_confusion": taxonomy_counts.get("Classification_Confusion", 0),
        "loc_error": taxonomy_counts.get("Localization_Error", 0),
        "tp_count": taxonomy_counts.get("True_Positive", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 8.6: One-Door Locked Test Set Benchmark (Hardened)")
    parser.add_argument("--ablation-dir", default="d:/UNDIP/SKRIPSI/stage_8_ablation_clean", help="Folder hasil ablasi (default: stage_8_ablation_clean)")
    parser.add_argument("--data-yaml", default="d:/UNDIP/SKRIPSI/cassava-v7-standardized/data.yaml", help="Path ke data.yaml")
    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="Konfirmasi eksplisit untuk membuka pintu uji akhir test set (Wajib diberikan untuk menjalankan benchmark).",
    )
    args = parser.parse_args()

    # Safety guard: prevent accidental test set leakage
    if not args.confirm_final_evaluation:
        print("=" * 80)
        print("PROTEKSI INTEGRITAS TEST SET — STAGE 8.6 TERKUNCI")
        print("=" * 80)
        print("Evaluasi final pada Locked Test Set HANYA boleh dijalankan setelah:")
        print("1. Seluruh clean training selesai.")
        print("2. Model kandidat definitif telah dipilih murni berdasarkan Validation Set.")
        print("3. Argumen --confirm-final-evaluation diberikan secara sadar.")
        print("\nSesuai Aturan Stage 9.4 (Comprehensive Audit Remediation), evaluasi final")
        print("pada test set ditangguhkan untuk mencegah unconfirmed test snooping.")
        print("=" * 80)
        sys.exit(0)

    ablation_dir = Path(args.ablation_dir).resolve()
    data_yaml_path = Path(args.data_yaml).resolve()
    test_img_dir, test_lbl_dir, test_images = resolve_data_yaml_split(data_yaml_path, "test")

    experiments = [
        {"id": "EXP-0", "name": "exp_0_baseline", "title": "Baseline (Re-run)"},
        {"id": "EXP-1", "name": "exp_1_adaptive_sampling", "title": "Mixture Adaptive Sampling"},
        {"id": "EXP-2", "name": "exp_2_targeted_aug", "title": "Targeted Context Augmentation"},
        {"id": "EXP-3", "name": "exp_3_full_proposed", "title": "Full Proposed System"},
    ]

    print("=" * 80)
    print("STAGE 8.6 — ONE-DOOR FINAL BENCHMARK ON LOCKED TEST SET")
    print("=" * 80)
    print(f"Data YAML  : {data_yaml_path}")
    print(f"Test Images: {test_img_dir} ({len(test_images):,} citra)")
    print(f"Test Labels: {test_lbl_dir}")
    print("=" * 80)

    test_results = []
    for exp in experiments:
        exp_dir = ablation_dir / exp["name"]
        best_pt = exp_dir / "weights" / "best.pt"

        if not best_pt.exists():
            print(f"[SKIP] Model {exp['id']} belum memiliki best.pt: {best_pt}")
            continue

        print(f"\n[{exp['id']}] Mengevaluasi {exp['title']} pada Test Set...")
        t0 = time.time()
        res = evaluate_test_set(best_pt, data_yaml_path, test_images, test_lbl_dir)
        duration = round(time.time() - t0, 1)

        res["exp_id"] = exp["id"]
        res["title"] = exp["title"]
        res["duration_sec"] = duration
        test_results.append(res)

        print(f"[{exp['id']}] SELESAI ({duration}s): mAP@50 = {res['map50']*100:.2f}% | "
              f"mAP@50-95 = {res['map50_95']*100:.2f}% | P = {res['precision']*100:.2f}% | "
              f"R = {res['recall']*100:.2f}% | R_Small = {res['recall_small']*100:.2f}% | Missed FN = {res['fn_missed']}")


if __name__ == "__main__":
    main()

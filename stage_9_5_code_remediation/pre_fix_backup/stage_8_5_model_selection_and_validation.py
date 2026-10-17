#!/usr/bin/env python3
"""
Stage 8.5 — Model Selection Hierarchy & Granular Validation Error Decomposition (Hardened)
========================================================================================

Tujuan Metodologis:
1. Mengevaluasi seluruh 4 checkpoint terbaik (best.pt) dari Stage 8.4 pada Validation Set (2.567 citra).
2. Menghitung metrik dekomposisi error berstrata ukuran menggunakan evaluasi kanonis terpadu:
   - research_utils.detection_metrics:
     * greedy_bipartite_matching (Global 1-to-1 bipartite greedy matching pada IoU >= 0.10, zero reuse)
     * classify_object_size (Small < 32^2, Medium 32^2-96^2, Large >= 96^2)
     * parse_yolo_labels (dimensi dinamis)
     * select_best_model (Decision rule resmi)
3. Menerapkan Aturan Hierarki Seleksi Model Terkunci (Decision Rule):
   - Kriteria Utama : Validasi mAP@50-95 tertinggi
   - Tie-Break 1    : Recall_Small tertinggi (hanya jika tie numerik selisih <= 1e-5)
   - Tie-Break 2    : Validasi mAP@50 tertinggi
   - Tie-Break 3    : Total False Negative (FN) terendah
4. Single Source of Truth: Menentukan direktori validasi secara dinamis dari --data-yaml.
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
    greedy_bipartite_matching,
    parse_yolo_labels,
    select_best_model,
)

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


def resolve_val_paths(data_yaml_path: str | Path) -> tuple[Path, Path]:
    """Menentukan direktori validasi images dan labels secara dinamis dari data.yaml."""
    yaml_p = Path(data_yaml_path).resolve()
    if not yaml_p.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {yaml_p}")

    with open(yaml_p, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = yaml_p.parent
    if "path" in data_cfg and data_cfg["path"]:
        base_path = Path(data_cfg["path"]).resolve()

    val_entry = data_cfg.get("val", "valid/images")
    val_path = (base_path / val_entry).resolve()
    if val_path.name == "images":
        val_img_dir = val_path
        val_lbl_dir = val_path.parent / "labels"
    else:
        val_img_dir = val_path / "images"
        val_lbl_dir = val_path / "labels"

    if not val_img_dir.exists():
        raise FileNotFoundError(f"Direktori citra validasi tidak ditemukan: {val_img_dir}")
    if not val_lbl_dir.exists():
        raise FileNotFoundError(f"Direktori label validasi tidak ditemukan: {val_lbl_dir}")

    return val_img_dir, val_lbl_dir


def evaluate_model_on_validation(
    model_path: Path,
    val_img_dir: Path,
    val_lbl_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    chunk_size: int = 32,
) -> dict[str, Any]:
    """Menjalankan evaluasi mendalam dan dekomposisi error berstrata ukuran via greedy_bipartite_matching."""
    model = YOLO(str(model_path))
    val_images = sorted(list(val_img_dir.glob("*.jpg")))

    total_gt = 0
    gt_by_size = Counter()
    tp_by_size = Counter()
    taxonomy_counts = Counter()
    class_tp = Counter()
    class_total_gt = Counter()

    gt_dict = {}
    for img_path in val_images:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        gt_objs = parse_yolo_labels(lbl_path, img_width=640.0, img_height=640.0)
        gt_dict[img_path.name] = gt_objs
        for gt in gt_objs:
            total_gt += 1
            gt_by_size[gt["size_category"]] += 1
            class_total_gt[gt["class_name"]] += 1

    t0 = time.time()
    for i in range(0, len(val_images), chunk_size):
        chunk = val_images[i : i + chunk_size]
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
            fname = img_path.name
            gt_objs = gt_dict[fname]

            pred_boxes = []
            pred_scores = []
            pred_cls = []

            if pred_res.boxes is not None and len(pred_res.boxes) > 0:
                pred_boxes = pred_res.boxes.xyxy.cpu().numpy()
                pred_scores = pred_res.boxes.conf.cpu().numpy()
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
                gt_cname = gt["class_name"]

                if g_idx in matches:
                    p_idx, iou_val = matches[g_idx]
                    p_cid = pred_cls[p_idx]
                    is_correct_class = (p_cid == gt_cid)

                    if iou_val >= 0.50:
                        if is_correct_class:
                            taxonomy_counts["True_Positive"] += 1
                            tp_by_size[gt_size] += 1
                            class_tp[gt_cname] += 1
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

    duration_sec = time.time() - t0

    recall_small = (tp_by_size["Small"] / gt_by_size["Small"]) if gt_by_size["Small"] > 0 else 0.0
    recall_medium = (tp_by_size["Medium"] / gt_by_size["Medium"]) if gt_by_size["Medium"] > 0 else 0.0
    recall_large = (tp_by_size["Large"] / gt_by_size["Large"]) if gt_by_size["Large"] > 0 else 0.0
    overall_recall = (sum(tp_by_size.values()) / total_gt) if total_gt > 0 else 0.0

    return {
        "duration_sec": duration_sec,
        "total_gt": total_gt,
        "gt_by_size": dict(gt_by_size),
        "tp_by_size": dict(tp_by_size),
        "recall_small": round(recall_small, 4),
        "recall_medium": round(recall_medium, 4),
        "recall_large": round(recall_large, 4),
        "overall_recall": round(overall_recall, 4),
        "taxonomy": dict(taxonomy_counts),
        "class_recall": {
            cname: round(class_tp[cname] / class_total_gt[cname], 4) if class_total_gt[cname] > 0 else 0.0
            for cname in CLASS_NAMES
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 8.5: Model Selection Hierarchy & Granular Validation Evaluation (Hardened)")
    parser.add_argument("--ablation-dir", default="d:/UNDIP/SKRIPSI/stage_8_ablation", help="Folder hasil Stage 8.4")
    parser.add_argument("--data-yaml", default="d:/UNDIP/SKRIPSI/cassava-v7-standardized/data.yaml", help="Path ke data.yaml")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir).resolve()
    val_img_dir, val_lbl_dir = resolve_val_paths(args.data_yaml)

    experiments = [
        {"id": "EXP-0", "name": "exp_0_baseline", "title": "Baseline (Re-run)"},
        {"id": "EXP-1", "name": "exp_1_adaptive_sampling", "title": "Mixture Adaptive Sampling"},
        {"id": "EXP-2", "name": "exp_2_targeted_aug", "title": "Targeted Context Augmentation"},
        {"id": "EXP-3", "name": "exp_3_full_proposed", "title": "Full Proposed System"},
    ]

    print("=" * 80)
    print("STAGE 8.5 — VALIDATION-BASED MODEL SELECTION & FINE-GRAINED ERROR DECOMPOSITION")
    print("=" * 80)
    print(f"Data YAML        : {args.data_yaml}")
    print(f"Validation Images: {val_img_dir}")
    print(f"Validation Labels: {val_lbl_dir}")
    print(f"Ablation Dir     : {ablation_dir}")
    print("Mengevaluasi 4 model ablasi secara apple-to-apple...")
    print("=" * 80)

    results = []

    for exp in experiments:
        exp_dir = ablation_dir / exp["name"]
        summary_path = exp_dir / "metrics_summary.json"
        best_pt = exp_dir / "weights" / "best.pt"

        if not summary_path.exists() or not best_pt.exists():
            print(f"[ERROR] Eksperimen {exp['id']} belum lengkap: {summary_path}")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            base_summary = json.load(f)

        print(f"\n[{exp['id']}] Menjalankan dekomposisi error berstrata untuk {exp['title']}...")
        fine_eval = evaluate_model_on_validation(
            best_pt,
            val_img_dir,
            val_lbl_dir,
            conf_thresh=0.25,
            iou_thresh=0.45,
            chunk_size=32,
        )

        val_metrics = base_summary["metrics_validation"]
        combined_record = {
            "exp_id": exp["id"],
            "title": exp["title"],
            "sampling": base_summary["sampling"],
            "augmentation": base_summary["augmentation"],
            "mAP50": val_metrics["mAP50"],
            "mAP50_95": val_metrics["mAP50_95"],
            "precision": val_metrics["precision"],
            "recall_overall": val_metrics["recall"],
            "recall_small": fine_eval["recall_small"],
            "recall_medium": fine_eval["recall_medium"],
            "recall_large": fine_eval["recall_large"],
            "fn_missed": fine_eval["taxonomy"].get("False_Negative_Missed", 0),
            "tp_count": fine_eval["taxonomy"].get("True_Positive", 0),
            "loc_error": fine_eval["taxonomy"].get("Localization_Error", 0),
            "cls_confusion": fine_eval["taxonomy"].get("Classification_Confusion", 0),
            "loc_cls_error": fine_eval["taxonomy"].get("Localization_and_Class_Error", 0),
            "per_class_ap50": val_metrics.get("per_class_ap50", val_metrics.get("per_class_mAP50", {})),
            "per_class_map50_95": val_metrics.get("per_class_map50_95", {}),
            "per_class_mAP50": val_metrics.get("per_class_ap50", val_metrics.get("per_class_mAP50", {})),
            "class_recall": fine_eval["class_recall"],
            "best_weight_path": str(best_pt),
        }
        results.append(combined_record)
        print(f"[{exp['id']}] mAP@50: {combined_record['mAP50']:.4f} | mAP@50-95: {combined_record['mAP50_95']:.4f} | "
              f"Recall_Small: {combined_record['recall_small']:.4f} | Missed FN: {combined_record['fn_missed']}")

    if not results:
        print("[FATAL] Tidak ada eksperimen yang berhasil dievaluasi!")
        sys.exit(1)

    csv_out_path = ablation_dir / "validation_fine_grained_comparison.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "exp_id", "title", "sampling", "augmentation", "mAP50", "mAP50_95",
            "precision", "recall_overall", "recall_small", "recall_medium", "recall_large",
            "fn_missed", "tp_count", "loc_error", "cls_confusion", "loc_cls_error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Apply canonical select_best_model from research_utils.detection_metrics
    selected_model, selection_rationale = select_best_model(results, tolerance=1e-5)

    report_path = ablation_dir / "STAGE_8_5_MODEL_SELECTION_REPORT.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Laporan Tahap 8.5 — Seleksi Model & Dekomposisi Error Validasi (Hardened)
Tanggal: {now_str}

## 1. Tabel Komparasi Metrik Dekomposisi Error & Strata Ukuran (Validation Set: 2.567 citra)

| ID Eksperimen | Konfigurasi | mAP@50 | mAP@50-95 | Recall Overall | Recall Small ($<32^2$) | Recall Med ($32-96^2$) | Recall Large ($>96^2$) | Missed FN | Cls Confusion | Loc Error |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        md_content += (
            f"| **{r['exp_id']}** | {r['title']} | "
            f"**{r['mAP50']*100:.2f}%** | **{r['mAP50_95']*100:.2f}%** | "
            f"{r['recall_overall']*100:.2f}% | **{r['recall_small']*100:.2f}%** | "
            f"{r['recall_medium']*100:.2f}% | {r['recall_large']*100:.2f}% | "
            f"{r['fn_missed']} | {r['cls_confusion']} | {r['loc_error']} |\n"
        )

    md_content += f"""
## 2. Performa Per-Kelas AP@50 (Validation Set)

| ID | CBB | CBSD | CGM | CMD | HEALTHY | Mean AP@50 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        pcm = r.get("per_class_ap50", r.get("per_class_mAP50", {}))
        cbb = pcm.get("CBB", 0.0) * 100
        cbsd = pcm.get("CBSD", 0.0) * 100
        cgm = pcm.get("CGM", 0.0) * 100
        cmd = pcm.get("CMD", 0.0) * 100
        hlth = pcm.get("HEALTHY", 0.0) * 100
        md_content += f"| **{r['exp_id']}** | {cbb:.2f}% | {cbsd:.2f}% | {cgm:.2f}% | {cmd:.2f}% | {hlth:.2f}% | **{r['mAP50']*100:.2f}%** |\n"

    md_content += f"""
## 3. Performa Per-Kelas AP@50-95 (Validation Set)

| ID | CBB | CBSD | CGM | CMD | HEALTHY | Mean AP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        pcm_95 = r.get("per_class_map50_95", {})
        cbb = pcm_95.get("CBB", 0.0) * 100
        cbsd = pcm_95.get("CBSD", 0.0) * 100
        cgm = pcm_95.get("CGM", 0.0) * 100
        cmd = pcm_95.get("CMD", 0.0) * 100
        hlth = pcm_95.get("HEALTHY", 0.0) * 100
        md_content += f"| **{r['exp_id']}** | {cbb:.2f}% | {cbsd:.2f}% | {cgm:.2f}% | {cmd:.2f}% | {hlth:.2f}% | **{r['mAP50_95']*100:.2f}%** |\n"

    md_content += f"""
## 4. Penerapan Hierarki Seleksi Model (Model Selection Decision Rule)

Kriteria Seleksi Terkunci Sesuai Protokol:
1. **Kriteria Utama**: Validasi mAP@50-95 tertinggi.
2. **Tie-Break 1**: Recall_Small tertinggi (hanya jika tie numerik selisih <= 1e-5).
3. **Tie-Break 2**: Validasi mAP@50 tertinggi.
4. **Tie-Break 3**: Total Missed False Negative (FN) terendah.

### Evaluasi Seleksi:
- **Model Terpilih Resmi**: **{selected_model['exp_id']}** ({selected_model['title']})
- **Rasional Keputusan**: {selection_rationale}
- **Metrik Kandidat Terpilih**: mAP@50-95 = **{selected_model['mAP50_95']*100:.2f}%**, mAP@50 = **{selected_model['mAP50']*100:.2f}%**, Recall_Small = **{selected_model['recall_small']*100:.2f}%**, FN = **{selected_model['fn_missed']}**
- **Path Bobot Terbaik**: `{selected_model['best_weight_path']}`
"""
    report_path.write_text(md_content, encoding="utf-8")
    print(f"\n[SUKSES] Laporan seleksi model Stage 8.5 tersimpan di: {report_path}")
    print(f"[DECISION] Model Terpilih: {selected_model['exp_id']} ({selected_model['title']})")


if __name__ == "__main__":
    main()

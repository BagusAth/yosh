#!/usr/bin/env python3
"""
Stage 8.4 — Ablation Study Experiments Orchestrator (EXP-0 s.d. EXP-3)
=====================================================================

Matriks Eksperimen Ablasi (Apple-to-Apple, 50 Epoch, Seed 42, YOLO11n):
-----------------------------------------------------------------------
1. EXP-0 (Baseline Re-run)       : Uniform Sampling (1/N) + Standard Augmentation
                                   (Official Stage 8 Baseline)
2. EXP-1 (Adaptive Sampling)     : Mixture Adaptive Sampling P(k) + Standard Augmentation
3. EXP-2 (Targeted Augmentation) : Uniform Sampling (1/N) + On-the-Fly Context Augmentation
4. EXP-3 (Full Proposed)         : Mixture Adaptive Sampling P(k) + Targeted Context Augmentation

Protokol Metodologis:
- Seluruh 4 varian menggunakan arsitektur YOLO11n yang identik.
- Evaluasi tuning & seleksi model dilakukan murni pada Validation Set (2.567 citra).
- Locked Test Set (1.282 citra) tetap terisolasi 100% sampai Stage 8.6.
- Menghasilkan ringkasan komparasi metrik: mAP@50, mAP@50-95, Precision, Recall, dan durasi.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from stage_8_3_adaptive_pipeline import AdaptiveDetectionTrainer

EXPERIMENT_SPECS = {
    0: {
        "id": "EXP-0",
        "name": "exp_0_baseline",
        "title": "Baseline (Re-run)",
        "sampling": "Uniform (1/N)",
        "augmentation": "Standard YOLO11n",
        "enable_mixture_sampling": False,
        "enable_targeted_aug": False,
        "description": "Baseline kontrol resmi Stage 8 (replikasi identik tanpa modifikasi adaptif)",
    },
    1: {
        "id": "EXP-1",
        "name": "exp_1_adaptive_sampling",
        "title": "Mixture Adaptive Sampling",
        "sampling": "Mixture Adaptive P(k)",
        "augmentation": "Standard YOLO11n",
        "enable_mixture_sampling": True,
        "enable_targeted_aug": False,
        "description": "Menguji dampak pemfokusan prioritas citra sulit via mixture sampler",
    },
    2: {
        "id": "EXP-2",
        "name": "exp_2_targeted_aug",
        "title": "Targeted Context Augmentation",
        "sampling": "Uniform (1/N)",
        "augmentation": "On-the-Fly Context Crop (2× bbox)",
        "enable_mixture_sampling": False,
        "enable_targeted_aug": True,
        "description": "Menguji dampak pembesaran resolusi konteks morfologi lesi kecil sulit",
    },
    3: {
        "id": "EXP-3",
        "name": "exp_3_full_proposed",
        "title": "Full Proposed (Sampling + Augmentation)",
        "sampling": "Mixture Adaptive P(k)",
        "augmentation": "On-the-Fly Context Crop (2× bbox)",
        "enable_mixture_sampling": True,
        "enable_targeted_aug": True,
        "description": "Menguji integrasi simultan sampling adaptif dan augmentasi konteks skala",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 8.4: Run Ablation Experiments (EXP-0 .. EXP-3)")
    parser.add_argument(
        "--exp",
        choices=["all", "0", "1", "2", "3"],
        default="all",
        help="Eksperimen yang akan dijalankan: 'all' untuk 0-3 berurutan, atau pilih 0, 1, 2, 3 (default: 'all')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch pelatihan (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size pelatihan (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi citra input (default: 640)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reprodusibilitas (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device CUDA (default: '0')",
    )
    parser.add_argument(
        "--data-yaml",
        default="d:/UNDIP/SKRIPSI/cassava-v7-research/data.yaml",
        help="Path ke data.yaml dataset resmi",
    )
    parser.add_argument(
        "--manifest-path",
        default="d:/UNDIP/SKRIPSI/stage_8_2_difficulty/train_image_difficulty_manifest.csv",
        help="Path ke manifest kesulitan citra dari Stage 8.2",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/UNDIP/SKRIPSI/stage_8_ablation",
        help="Folder penyimpanan seluruh artefak ablasi",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Lewati eksperimen jika model best.pt dan metrics_summary.json sudah lengkap",
    )
    return parser.parse_args()


def run_single_experiment(exp_idx: int, args) -> dict[str, Any]:
    """Menjalankan satu eksperimen ablasi (EXP-0, EXP-1, EXP-2, atau EXP-3)."""
    spec = EXPERIMENT_SPECS[exp_idx]
    output_dir = Path(args.output_dir).resolve()
    exp_dir = output_dir / spec["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = exp_dir / "metrics_summary.json"
    best_weight_path = exp_dir / "weights" / "best.pt"

    # Cek apakah sudah selesai dan bisa dilewati
    if args.skip_if_done and summary_json_path.exists() and best_weight_path.exists():
        print(f"[{spec['id']}] Artefak sudah lengkap. Melewati pelatihan.")
        with open(summary_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n" + "=" * 80)
    print(f"MEMULAI EKSPERIMEN: {spec['id']} — {spec['title']}")
    print("=" * 80)
    print(f"Sampling Mechanism   : {spec['sampling']}")
    print(f"Augmentation Policy  : {spec['augmentation']}")
    print(f"Epochs / Batch       : {args.epochs} / {args.batch}")
    print(f"Image Resolution     : {args.imgsz}")
    print(f"Random Seed          : {args.seed}")
    print(f"Output Directory     : {exp_dir}")
    print("=" * 80)

    # Pastikan VRAM bersih sebelum training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_start = time.time()

    # Inisialisasi AdaptiveDetectionTrainer kustom
    trainer = AdaptiveDetectionTrainer(
        overrides={
            "model": "yolo11n.pt",
            "data": str(Path(args.data_yaml).resolve()),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "device": args.device,
            "workers": 0,  # 0 workers mencegah paging file limit di Windows
            "project": str(output_dir),
            "name": spec["name"],
            "save": True,
            "exist_ok": True,
            "plots": True,
            "val": True,
        },
        enable_mixture_sampling=spec["enable_mixture_sampling"],
        enable_targeted_aug=spec["enable_targeted_aug"],
        manifest_path=args.manifest_path,
        aug_prob=0.50,
    )

    trainer.train()
    t_duration = time.time() - t_start

    # Salin best weight ke exp_dir/weights/best.pt jika belum ada di sana
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    trained_best = exp_dir / "weights" / "best.pt"
    if not trained_best.exists():
        # Cek jika Ultralytics menyimpan di subfolder
        candidate_best = output_dir / spec["name"] / "weights" / "best.pt"
        if candidate_best.exists() and candidate_best != best_weight_path:
            shutil.copy2(candidate_best, best_weight_path)

    # Validasi mandiri pada Validation Set menggunakan checkpoint best.pt
    print(f"\n[{spec['id']}] Mengevaluasi model terbaik (best.pt) pada Validation Set...")
    eval_model = YOLO(str(best_weight_path if best_weight_path.exists() else trained_best))
    val_metrics = eval_model.val(
        data=str(Path(args.data_yaml).resolve()),
        split="val",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
    )

    precision = float(val_metrics.box.mp)
    recall = float(val_metrics.box.mr)
    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)

    # Ekstrak per-class mAP50
    per_class_map50 = {}
    class_names = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
    if hasattr(val_metrics.box, "maps") and val_metrics.box.maps is not None:
        for idx, cname in enumerate(class_names):
            if idx < len(val_metrics.box.maps):
                per_class_map50[cname] = float(val_metrics.box.maps[idx])

    summary = {
        "exp_id": spec["id"],
        "name": spec["name"],
        "title": spec["title"],
        "sampling": spec["sampling"],
        "augmentation": spec["augmentation"],
        "epochs": args.epochs,
        "seed": args.seed,
        "duration_minutes": round(t_duration / 60.0, 2),
        "metrics_validation": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "per_class_mAP50": per_class_map50,
        },
        "best_weight_path": str(best_weight_path),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{spec['id']}] SELESAI ({summary['duration_minutes']} menit): "
          f"mAP@50 = {map50 * 100:.2f}%, mAP@50-95 = {map50_95 * 100:.2f}%, "
          f"P = {precision * 100:.2f}%, R = {recall * 100:.2f}%")

    # Bersihkan memori GPU
    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def generate_ablation_report(summaries: list[dict[str, Any]], output_dir: Path):
    """Menyusun laporan komparatif ABLATION_EXPERIMENTS_REPORT.md."""
    report_path = output_dir / "ABLATION_EXPERIMENTS_REPORT.md"

    # Ambil EXP-0 sebagai baseline acuan resmi
    exp0 = next((s for s in summaries if s["exp_id"] == "EXP-0"), None)
    base_map50 = exp0["metrics_validation"]["mAP50"] if exp0 else 0.0
    base_map50_95 = exp0["metrics_validation"]["mAP50_95"] if exp0 else 0.0

    lines = [
        "# Laporan Eksperimen Ablasi Stage 8 (EXP-0 s.d. EXP-3)",
        f"Tanggal Pembuatan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 1. Tabel Komparasi Hasil Validasi (Validation Set: 2.567 citra)\n",
        "| ID | Konfigurasi | Sampling | Augmentasi | Precision | Recall | mAP@50 | mAP@50-95 | Δ mAP@50 | Δ mAP@50-95 | Durasi |",
        "| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summaries:
        m = s["metrics_validation"]
        p = m["precision"] * 100
        r = m["recall"] * 100
        m50 = m["mAP50"] * 100
        m50_95 = m["mAP50_95"] * 100

        delta_m50 = (m["mAP50"] - base_map50) * 100 if exp0 else 0.0
        delta_m50_95 = (m["mAP50_95"] - base_map50_95) * 100 if exp0 else 0.0

        d50_str = f"{delta_m50:+.2f}%" if s["exp_id"] != "EXP-0" else "0.00% (Ref)"
        d50_95_str = f"{delta_m50_95:+.2f}%" if s["exp_id"] != "EXP-0" else "0.00% (Ref)"

        lines.append(
            f"| **{s['exp_id']}** | {s['title']} | {s['sampling']} | {s['augmentation']} | "
            f"{p:.2f}% | {r:.2f}% | **{m50:.2f}%** | **{m50_95:.2f}%** | {d50_str} | {d50_95_str} | {s['duration_minutes']} m |"
        )

    lines.extend([
        "\n## 2. Analisis Performa Per-Kelas (mAP@50)\n",
        "| ID | CBB | CBSD | CGM | CMD | HEALTHY | Mean mAP@50 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s in summaries:
        pcm = s["metrics_validation"].get("per_class_mAP50", {})
        cbb = pcm.get("CBB", 0.0) * 100
        cbsd = pcm.get("CBSD", 0.0) * 100
        cgm = pcm.get("CGM", 0.0) * 100
        cmd = pcm.get("CMD", 0.0) * 100
        hly = pcm.get("HEALTHY", 0.0) * 100
        m50 = s["metrics_validation"]["mAP50"] * 100
        lines.append(
            f"| **{s['exp_id']}** | {cbb:.2f}% | {cbsd:.2f}% | {cgm:.2f}% | {cmd:.2f}% | {hly:.2f}% | **{m50:.2f}%** |"
        )

    lines.extend([
        "\n## 3. Evaluasi Pertanyaan Penelitian & Hipotesis",
        "- **RQ1 (Efek Mixture Adaptive Sampling):** Komparasi EXP-1 terhadap EXP-0.",
        "- **RQ2 (Efek Targeted Context Augmentation):** Komparasi EXP-2 terhadap EXP-0.",
        "- **RQ3 (Kombinasi Kedua Mekanisme):** Komparasi EXP-3 terhadap EXP-1 dan EXP-2.",
        "\n*Catatan Metodologis:* Hasil seleksi model terbaik berdasarkan hierarki validasi "
        "(Primary: mAP@50-95) akan diteruskan ke evaluasi akhir satu pintu pada Locked Test Set (Stage 8.6).\n"
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[REPORT] Laporan komparasi ablasi tersimpan di {report_path}")


def main():
    args = parse_args()
    print("=" * 80)
    print("STAGE 8.4 — ABLATION EXPERIMENTS RUNNER")
    print("=" * 80)
    print(f"Target Eksperimen : {args.exp}")
    print(f"Epochs per Model  : {args.epochs}")
    print(f"Batch Size        : {args.batch}")
    print(f"Image Size        : {args.imgsz}")
    print(f"Random Seed       : {args.seed}")
    print(f"Output Directory  : {args.output_dir}")
    print("=" * 80)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.exp == "all":
        exp_indices = [0, 1, 2, 3]
    else:
        exp_indices = [int(args.exp)]

    summaries = []
    for idx in exp_indices:
        res = run_single_experiment(idx, args)
        summaries.append(res)

    # Jika semua 4 eksperimen sudah selesai, buat laporan ablasi terpadu
    all_completed = []
    for idx in range(4):
        s_path = output_dir / EXPERIMENT_SPECS[idx]["name"] / "metrics_summary.json"
        if s_path.exists():
            with open(s_path, "r", encoding="utf-8") as f:
                all_completed.append(json.load(f))

    if len(all_completed) == 4:
        generate_ablation_report(all_completed, output_dir)


if __name__ == "__main__":
    main()

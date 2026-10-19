#!/usr/bin/env python3
"""
Stage 8.4 — Ablation Study Experiments Orchestrator (EXP-0 s.d. EXP-3) (Hardened)
================================================================================

Matriks Eksperimen Ablasi (Apple-to-Apple, 50 Epoch, Seed 42, YOLO11n):
-----------------------------------------------------------------------
1. EXP-0 (Baseline Re-run)       : Uniform Sampling (1/N) + Standard Augmentation
2. EXP-1 (Adaptive Sampling)     : Mixture Adaptive Sampling P(k) + Standard Augmentation
3. EXP-2 (Targeted Augmentation) : Uniform Sampling (1/N) + On-the-Fly Context Augmentation
4. EXP-3 (Full Proposed)         : Mixture Adaptive Sampling P(k) + Targeted Context Augmentation

Protokol Metodologis & Safety:
- Menjaga pelestarian artefak historis: output default dialihkan ke stage_10_11_clean_experiments.
  Menulis ke stage_8_ablation dilarang keras kecuali flag eksplisit diberikan.
- Integrasi extract_canonical_metrics dari research_utils.detection_metrics.
- Verifikasi metadata dan checksum untuk mencegah silent reuse pada --skip-if-done.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
import ultralytics
from ultralytics import YOLO

from research_utils.detection_metrics import (
    compute_file_sha256,
    compute_dataset_inventory_fingerprint,
    extract_canonical_metrics,
)
from stage_8_3_adaptive_pipeline import AdaptiveDetectionTrainer

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]

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


def compute_file_sha256(filepath: str | Path) -> str:
    p = Path(filepath)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 8.4: Run Ablation Experiments (EXP-0 .. EXP-3) (Hardened)")
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
        default="d:/UNDIP/SKRIPSI/cassava-v7-standardized/data.yaml",
        help="Path ke data.yaml dataset (Single Source of Truth, default: standardized)",
    )
    parser.add_argument(
        "--manifest-path",
        default="d:/UNDIP/SKRIPSI/stage_8_2_difficulty/train_image_difficulty_manifest.csv",
        help="Path ke manifest kesulitan citra dari Stage 8.2",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/UNDIP/SKRIPSI/stage_10_11_clean_experiments",
        help="Folder penyimpanan seluruh artefak ablasi baru (default: stage_10_11_clean_experiments)",
    )
    parser.add_argument(
        "--allow-historic-overwrite",
        action="store_true",
        help="Izinkan penimpaan ke stage_8_ablation (SANGAT TIDAK DISARANKAN)",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Lewati eksperimen jika model best.pt, metrics_summary.json, dan metadata checksum cocok",
    )
    return parser.parse_args()


def build_experiment_run_identity(
    exp_id: str,
    spec: dict[str, Any],
    script_path: str | Path,
    yaml_path: str | Path,
    manifest_path: str | Path,
    dataset_inventory_fingerprint: str,
    pretrained_weights_path: str | Path,
    epochs: int,
    batch_size: int,
    imgsz: int,
    seed: int,
    device: str = "",
    checkpoint_best_sha256: str = "",
) -> dict[str, Any]:
    """Membangun objek identitas konfigurasi eksperimen ablasi secara kanonis."""
    p_script = Path(script_path) if script_path else None
    p_yaml = Path(yaml_path) if yaml_path else None
    p_manifest = Path(manifest_path) if manifest_path else None
    p_pre = Path(pretrained_weights_path) if pretrained_weights_path else None

    return {
        "exp_id": exp_id,
        "exp_name": spec["name"],
        "sampling": spec["sampling"],
        "augmentation": spec["augmentation"],
        "enable_mixture_sampling": spec["enable_mixture_sampling"],
        "enable_targeted_aug": spec["enable_targeted_aug"],
        "epochs": epochs,
        "batch_size": batch_size,
        "batch": batch_size,
        "imgsz": imgsz,
        "seed": seed,
        "device": str(device),
        "model_base": "yolo11n.pt",
        "script_sha256": compute_file_sha256(p_script) if p_script and p_script.exists() else str(script_path),
        "data_yaml_sha256": compute_file_sha256(p_yaml) if p_yaml and p_yaml.exists() else str(yaml_path),
        "manifest_sha256": compute_file_sha256(p_manifest) if (p_manifest and p_manifest.exists() and (spec["enable_mixture_sampling"] or spec["enable_targeted_aug"])) else "",
        "pretrained_weights_sha256": compute_file_sha256(p_pre) if (p_pre and p_pre.exists()) else "",
        "pretrained_sha256": compute_file_sha256(p_pre) if (p_pre and p_pre.exists()) else "",
        "dataset_inventory_fingerprint": dataset_inventory_fingerprint,
        "best_weight_sha256": checkpoint_best_sha256,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "ultralytics_version": getattr(ultralytics, "__version__", ""),
    }


def verify_ablation_reuse_safety(
    meta_path: Path,
    expected_identity: dict[str, Any],
    checkpoint_best_path: Path,
) -> tuple[bool, str]:
    """
    Memverifikasi apakah artefak ablasi lama valid dan cocok 100% untuk di-skip (--skip-if-done).
    Memeriksa seluruh field konfigurasi, fingerprint skrip, YAML, manifest, dan bobot pretrained.
    """
    if not meta_path.exists():
        return False, "run_metadata.json not found"
    if not checkpoint_best_path.exists():
        return False, "Checkpoint best.pt does not exist"

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("run_metadata", data)
    except Exception as e:
        return False, f"Failed to parse metadata JSON: {e}"

    for k, exp_val in expected_identity.items():
        if k in ("best_weight_sha256", "duration_minutes", "timestamp", "metrics_validation"):
            continue
        if k == "batch_size" and "batch_size" not in meta and "batch" in meta:
            saved_val = meta["batch"]
        elif k == "pretrained_weights_sha256" and "pretrained_weights_sha256" not in meta and "pretrained_sha256" in meta:
            saved_val = meta["pretrained_sha256"]
        else:
            saved_val = meta.get(k)
        if saved_val != exp_val:
            return False, f"{k} mismatch: saved={saved_val} vs exp={exp_val}"

    actual_best_sha = compute_file_sha256(checkpoint_best_path)
    saved_best_sha = meta.get("best_weight_sha256", "")
    if not saved_best_sha or actual_best_sha != saved_best_sha:
        return False, f"best.pt hash mismatch: saved={saved_best_sha[:8]} vs actual={actual_best_sha[:8]}"

    return True, "Ablation run verified and safe for reuse"


def run_single_experiment(exp_idx: int, args) -> dict[str, Any]:
    """Menjalankan satu eksperimen ablasi (EXP-0, EXP-1, EXP-2, atau EXP-3)."""
    spec = EXPERIMENT_SPECS[exp_idx]
    output_dir = Path(args.output_dir).resolve()

    # Safety protection of historical artefacts
    if "stage_8_ablation" in str(output_dir).lower() and not args.allow_historic_overwrite:
        raise PermissionError(
            f"CRITICAL SAFETY ERROR: Menulis ke direktori historis '{output_dir}' dilarang keras!\n"
            "Gunakan direktori baru (misal stage_10_11_clean_experiments) untuk menjaga integritas data historis."
        )

    exp_dir = output_dir / spec["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = exp_dir / "metrics_summary.json"
    best_weight_path = exp_dir / "weights" / "best.pt"

    dataset_inv_sha = ""
    try:
        yaml_p = Path(args.data_yaml).resolve()
        if yaml_p.exists():
            with open(yaml_p, "r", encoding="utf-8") as f:
                dy = yaml.safe_load(f)
            base_p = Path(dy["path"]).resolve() if "path" in dy and dy["path"] else yaml_p.parent
            tr_p = (base_p / dy.get("train", "train/images")).resolve()
            im_dir = tr_p if tr_p.name == "images" else tr_p / "images"
            lbl_dir = im_dir.parent / "labels"
            dataset_inv_sha = compute_dataset_inventory_fingerprint(im_dir, lbl_dir)
    except Exception:
        pass

    expected_identity = build_experiment_run_identity(
        exp_id=spec["id"],
        spec=spec,
        script_path=__file__,
        yaml_path=args.data_yaml,
        manifest_path=args.manifest_path,
        dataset_inventory_fingerprint=dataset_inv_sha,
        pretrained_weights_path="yolo11n.pt",
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        device=str(args.device),
    )

    # Strict Fingerprint and Artifact Verification for --skip-if-done
    if args.skip_if_done and summary_json_path.exists() and best_weight_path.exists():
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=summary_json_path,
            expected_identity=expected_identity,
            checkpoint_best_path=best_weight_path,
        )
        if can_reuse:
            print(f"[{spec['id']}] Artefak dan seluruh metadata checksum terverifikasi cocok 100%. Melewati pelatihan.")
            with open(summary_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"[{spec['id']}] WARNING: Metadata lama tidak cocok dengan parameter saat ini ({reason}). Menolak skip...")

    # Guard against overwriting unverified / mismatched existing checkpoint
    if best_weight_path.exists():
        raise PermissionError(
            f"[{spec['id']}] SAFETY VIOLATION: Checkpoint '{best_weight_path}' sudah ada di folder tujuan tetapi gagal verifikasi metadata/fingerprint!\n"
            "Penimpaan ke direktori dengan artefak yang tidak cocok dilarang keras.\n"
            "Harap gunakan folder --output-dir baru yang bersih."
        )

    print("\n" + "=" * 80)
    print(f"MEMULAI EKSPERIMEN: {spec['id']} — {spec['title']}")
    print("=" * 80)
    print(f"Sampling Mechanism   : {spec['sampling']}")
    print(f"Augmentation Policy  : {spec['augmentation']}")
    print(f"Epochs / Batch       : {args.epochs} / {args.batch}")
    print(f"Image Resolution     : {args.imgsz}")
    print(f"Random Seed          : {args.seed}")
    print(f"Data YAML            : {args.data_yaml}")
    print(f"Output Directory     : {exp_dir}")
    print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_start = time.time()

    trainer = AdaptiveDetectionTrainer(
        overrides={
            "model": "yolo11n.pt",
            "data": str(Path(args.data_yaml).resolve()),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "device": args.device,
            "workers": 0,
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

    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    trained_best = exp_dir / "weights" / "best.pt"
    if not trained_best.exists():
        candidate_best = output_dir / spec["name"] / "weights" / "best.pt"
        if candidate_best.exists() and candidate_best != best_weight_path:
            shutil.copy2(candidate_best, best_weight_path)

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

    # Use canonical extract_canonical_metrics from research_utils
    canonical_m = extract_canonical_metrics(val_metrics, CLASS_NAMES)

    best_weight_sha256 = compute_file_sha256(best_weight_path)

    run_meta = dict(expected_identity)
    run_meta["best_weight_sha256"] = best_weight_sha256

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
            "precision": canonical_m["precision"],
            "recall": canonical_m["recall"],
            "mAP50": canonical_m["map50"],
            "mAP50_95": canonical_m["map50_95"],
            "per_class_ap50": canonical_m["per_class_ap50"],
            "per_class_map50_95": canonical_m["per_class_map50_95"],
            "per_class_mAP50": canonical_m["per_class_ap50"],
        },
        "best_weight_path": str(best_weight_path),
        "best_weight_sha256": best_weight_sha256,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_metadata": run_meta,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{spec['id']}] SELESAI ({summary['duration_minutes']} menit): "
          f"mAP@50 = {canonical_m['map50'] * 100:.2f}%, mAP@50-95 = {canonical_m['map50_95'] * 100:.2f}%, "
          f"P = {canonical_m['precision'] * 100:.2f}%, R = {canonical_m['recall'] * 100:.2f}%")

    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def generate_ablation_report(summaries: list[dict[str, Any]], output_dir: Path):
    """Menyusun laporan komparatif ABLATION_EXPERIMENTS_REPORT.md."""
    report_path = output_dir / "ABLATION_EXPERIMENTS_REPORT.md"

    exp0 = next((s for s in summaries if s["exp_id"] == "EXP-0"), None)
    base_map50 = exp0["metrics_validation"]["mAP50"] if exp0 else 0.0
    base_map50_95 = exp0["metrics_validation"]["mAP50_95"] if exp0 else 0.0

    lines = [
        "# Laporan Eksperimen Ablasi Stage 8 (EXP-0 s.d. EXP-3) (Hardened)",
        f"Tanggal Pembuatan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 1. Tabel Komparasi Hasil Validasi (Validation Set: 2.567 citra)\n",
        "| ID | Konfigurasi | Sampling | Augmentasi | Precision | Recall | mAP@50 | mAP@50-95 | Δ mAP@50 | Δ mAP@50-95 | Durasi |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summaries:
        m = s["metrics_validation"]
        d_map50 = (m["mAP50"] - base_map50) * 100
        d_map50_95 = (m["mAP50_95"] - base_map50_95) * 100
        d50_str = f"{d_map50:+.2f}%" if s["exp_id"] != "EXP-0" else "Ref (0.0%)"
        d95_str = f"{d_map50_95:+.2f}%" if s["exp_id"] != "EXP-0" else "Ref (0.0%)"
        lines.append(
            f"| **{s['exp_id']}** | {s['title']} | {s['sampling']} | {s['augmentation']} | "
            f"{m['precision']*100:.2f}% | {m['recall']*100:.2f}% | "
            f"**{m['mAP50']*100:.2f}%** | **{m['mAP50_95']*100:.2f}%** | "
            f"{d50_str} | {d95_str} | {s['duration_minutes']}m |"
        )

    lines.append("\n## 2. Performa Per-Kelas AP@50 (Validation Set)\n")
    lines.append("| ID | Konfigurasi | CBB | CBSD | CGM | CMD | HEALTHY | Mean AP@50 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for s in summaries:
        pcm = s["metrics_validation"]["per_class_ap50"]
        cbb = pcm.get("CBB", 0.0) * 100
        cbsd = pcm.get("CBSD", 0.0) * 100
        cgm = pcm.get("CGM", 0.0) * 100
        cmd = pcm.get("CMD", 0.0) * 100
        hlth = pcm.get("HEALTHY", 0.0) * 100
        mean_v = s["metrics_validation"]["mAP50"] * 100
        lines.append(
            f"| **{s['exp_id']}** | {s['title']} | "
            f"{cbb:.2f}% | {cbsd:.2f}% | {cgm:.2f}% | {cmd:.2f}% | {hlth:.2f}% | **{mean_v:.2f}%** |"
        )

    lines.append("\n## 3. Performa Per-Kelas AP@50-95 (Validation Set)\n")
    lines.append("| ID | Konfigurasi | CBB | CBSD | CGM | CMD | HEALTHY | Mean AP@50-95 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for s in summaries:
        pcm_95 = s["metrics_validation"]["per_class_map50_95"]
        cbb = pcm_95.get("CBB", 0.0) * 100
        cbsd = pcm_95.get("CBSD", 0.0) * 100
        cgm = pcm_95.get("CGM", 0.0) * 100
        cmd = pcm_95.get("CMD", 0.0) * 100
        hlth = pcm_95.get("HEALTHY", 0.0) * 100
        mean_v = s["metrics_validation"]["mAP50_95"] * 100
        lines.append(
            f"| **{s['exp_id']}** | {s['title']} | "
            f"{cbb:.2f}% | {cbsd:.2f}% | {cgm:.2f}% | {cmd:.2f}% | {hlth:.2f}% | **{mean_v:.2f}%** |"
        )

    lines.append("\n---\n*Laporan dibuat secara otomatis oleh Stage 8.4 Orchestrator.*")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] Laporan komparasi tersimpan di: {report_path}")


def main():
    args = parse_args()
    print("=" * 80)
    print("STAGE 8.4 — ABLATION STUDY EXPERIMENTS (EXP-0 .. EXP-3) (HARDENED)")
    print("=" * 80)
    print(f"Data YAML          : {args.data_yaml}")
    print(f"Manifest Path      : {args.manifest_path}")
    print(f"Output Directory   : {args.output_dir}")
    print(f"Target Eksperimen  : {args.exp}")
    print(f"Epochs per Eksper. : {args.epochs}")
    print(f"Batch Size         : {args.batch}")
    print(f"Image Resolution   : {args.imgsz}")
    print(f"Random Seed        : {args.seed}")
    print(f"Skip if Done       : {args.skip_if_done}")
    print("=" * 80)

    if args.exp == "all":
        exp_indices = [0, 1, 2, 3]
    else:
        exp_indices = [int(args.exp)]

    summaries = []
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in exp_indices:
        summary = run_single_experiment(idx, args)
        summaries.append(summary)

    if len(summaries) > 1:
        generate_ablation_report(summaries, output_dir)

    print("\n" + "=" * 80)
    print("SELURUH EKSPERIMEN ABLASI STAGE 8.4 SELESAI!")
    print("=" * 80)


if __name__ == "__main__":
    main()

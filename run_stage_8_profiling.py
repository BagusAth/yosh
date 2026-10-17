#!/usr/bin/env python3
"""
Stage 8 Profiling Orchestrator (Step 8.1 + Step 8.2) (Hardened)
==============================================================

Menjalankan pipeline profiling kesulitan objek dan citra secara otomatis:
1. Stage 8.1: 4-Fold Group-Aware OOF Probe Training & Diagnostic Inference (@conf=0.001)
   dengan Nested Group-Aware Holdout Isolation.
2. Stage 8.2: Class-Agnostic Greedy Bipartite Matching (IoU >= 0.10), D_i & D_image Calculation, Sanity Gates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 8 Profiling Pipeline Orchestrator (8.1 + 8.2)")
    parser.add_argument("--data-yaml", default=None, help="Path ke data.yaml (Single Source of Truth)")
    parser.add_argument("--dataset-dir", default=None, help="Path root dataset (misal cassava-v7-standardized)")
    parser.add_argument("--research-dir", default="d:/UNDIP/SKRIPSI/cassava-v7-research", help="Path fallback dataset")
    parser.add_argument("--output-dir-81", default="d:/UNDIP/SKRIPSI/stage_8_1_oof", help="Folder output Stage 8.1")
    parser.add_argument("--output-dir-82", default="d:/UNDIP/SKRIPSI/stage_8_2_difficulty", help="Folder output Stage 8.2")
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=["nested_inner_val", "fixed_epoch_last"],
        default="nested_inner_val",
        help="Metode pemilihan checkpoint: 'nested_inner_val' (default) atau 'fixed_epoch_last'",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Jumlah epoch probe model (default: 12)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size pelatihan (default: 16)")
    parser.add_argument("--conf-oof", type=float, default=0.001, help="Threshold confidence diagnostik OOF (default: 0.001)")
    parser.add_argument("--iou-match-thresh", type=float, default=0.10, help="Ambang batas minimum IoU matching (default: 0.10)")
    parser.add_argument("--mixture-lambda", type=float, default=0.70, help="Bobot mixture adaptive sampling (default: 0.70)")
    parser.add_argument("--skip-training-if-done", action="store_true", help="Lewati training jika artefak sudah lengkap dan valid")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan partisi dan audit tanpa training GPU")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("STAGE 8 — PROFILING PIPELINE ORCHESTRATOR (HARDENED)")
    print(f"Waktu Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data YAML  : {args.data_yaml}")
    print(f"Dataset Dir: {args.dataset_dir or args.research_dir}")
    print(f"Mode OOF   : {args.checkpoint_selection_mode}")
    print("=" * 80)

    t0 = time.time()
    print("\n" + "#" * 80)
    print("LANGKAH 1: MENJALANKAN STAGE 8.1 (GROUP-AWARE 4-FOLD OOF PROFILING)")
    print("#" * 80)

    cmd_8_1 = [
        sys.executable,
        "-u",
        "stage_8_1_group_oof_profiling.py",
        "--output-dir", args.output_dir_81,
        "--checkpoint-selection-mode", args.checkpoint_selection_mode,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--conf-oof", str(args.conf_oof),
    ]
    if args.data_yaml:
        cmd_8_1.extend(["--data-yaml", args.data_yaml])
    if args.dataset_dir:
        cmd_8_1.extend(["--dataset-dir", args.dataset_dir])
    else:
        cmd_8_1.extend(["--research-dir", args.research_dir])
    if args.skip_training_if_done:
        cmd_8_1.append("--skip-training-if-done")
    if args.dry_run:
        cmd_8_1.append("--dry-run")

    res1 = subprocess.run(cmd_8_1)
    if res1.returncode != 0:
        print(f"\n[FATAL ERROR] Stage 8.1 gagal dengan exit code {res1.returncode}!")
        sys.exit(res1.returncode)

    t_step1 = time.time() - t0
    print(f"\n[SUKSES] Stage 8.1 selesai dalam {t_step1 / 60:.2f} menit.")

    if args.dry_run:
        print("\n[DRY-RUN SELESAI] Tahap 8.1 terverifikasi aman. Menghentikan orchestrator sebelum tahap 8.2.")
        return

    # 2. Jalankan Stage 8.2
    t1 = time.time()
    print("\n" + "#" * 80)
    print("LANGKAH 2: MENJALANKAN STAGE 8.2 (BUILD DIFFICULTY MANIFEST & SANITY GATES)")
    print("#" * 80)

    cmd_8_2 = [
        sys.executable,
        "-u",
        "stage_8_2_build_difficulty_manifest.py",
        "--oof-dir", args.output_dir_81,
        "--output-dir", args.output_dir_82,
        "--iou-match-thresh", str(args.iou_match_thresh),
        "--mixture-lambda", str(args.mixture_lambda),
    ]
    if args.data_yaml:
        cmd_8_2.extend(["--data-yaml", args.data_yaml])
    if args.dataset_dir:
        cmd_8_2.extend(["--dataset-dir", args.dataset_dir])
    else:
        cmd_8_2.extend(["--research-dir", args.research_dir])

    res2 = subprocess.run(cmd_8_2)
    if res2.returncode != 0:
        print(f"\n[FATAL ERROR] Stage 8.2 gagal dengan exit code {res2.returncode}!")
        sys.exit(res2.returncode)

    t_step2 = time.time() - t1
    t_total = time.time() - t0

    print(f"\n[SUKSES] Stage 8.2 selesai dalam {t_step2:.1f} detik.")
    print("=" * 80)
    print(f"PIPELINE STAGE 8.1 & 8.2 SELESAI 100% SUKSES! (Total waktu: {t_total / 60:.2f} menit)")
    print(f"Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

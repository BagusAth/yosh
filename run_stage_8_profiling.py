#!/usr/bin/env python3
"""
Stage 8 Profiling Orchestrator (Step 8.1 + Step 8.2)
====================================================

Menjalankan pipeline profiling kesulitan objek dan citra secara otomatis:
1. Stage 8.1: 4-Fold Group-Aware OOF Probe Training (12 epoch per fold) & Diagnostic Inference (@conf=0.001)
2. Stage 8.2: Class-Agnostic Greedy Bipartite Matching (IoU >= 0.10), D_i & D_image Calculation, Sanity Gates
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    print("=" * 80)
    print("STAGE 8 — PROFILING PIPELINE ORCHESTRATOR")
    print(f"Waktu Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Jalankan Stage 8.1
    t0 = time.time()
    print("\n" + "#" * 80)
    print("LANGKAH 1: MENJALANKAN STAGE 8.1 (GROUP-AWARE 4-FOLD OOF PROFILING)")
    print("#" * 80)

    cmd_8_1 = [
        sys.executable,
        "stage_8_1_group_oof_profiling.py",
        "--epochs", "12",
        "--batch", "16",
        "--conf-oof", "0.001",
        "--skip-training-if-done",
    ]
    res1 = subprocess.run(cmd_8_1)
    if res1.returncode != 0:
        print(f"\n[FATAL ERROR] Stage 8.1 gagal dengan exit code {res1.returncode}!")
        sys.exit(res1.returncode)

    t_step1 = time.time() - t0
    print(f"\n[SUKSES] Stage 8.1 selesai dalam {t_step1 / 60:.2f} menit.")

    # 2. Jalankan Stage 8.2
    t1 = time.time()
    print("\n" + "#" * 80)
    print("LANGKAH 2: MENJALANKAN STAGE 8.2 (BUILD DIFFICULTY MANIFEST & SANITY GATES)")
    print("#" * 80)

    cmd_8_2 = [
        sys.executable,
        "stage_8_2_build_difficulty_manifest.py",
        "--iou-match-thresh", "0.10",
        "--confusion-penalty", "0.35",
        "--mixture-lambda", "0.70",
    ]
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

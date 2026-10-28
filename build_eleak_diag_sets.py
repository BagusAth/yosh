#!/usr/bin/env python3
"""
E-LEAK diagnostik (RQ1 bagian A, Pakem v1.3)
===========================================
Memisahkan citra validasi simulasi yang TIDAK pernah terlihat oleh model simulasi (twinval + single)
berdasarkan asal split bersihnya:
  diag_A: berasal dari validasi bersih (juga tidak terlihat oleh model split per grup)
  diag_B: berasal dari latih bersih
Tujuan: menguji apakah keunggulan model simulasi pada citra yang tidak terlihat berasal dari model yang
lebih baik (sim(A) tinggi) atau dari perbedaan tingkat kesulitan antara asal latih dan validasi bersih
(sim(A) rendah, sim(B) tinggi). Hanya menulis file daftar dan yaml ke eleak_sim_rq1/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = ROOT / "eleak_sim_rq1"
WIN = "D:/UNDIP/SKRIPSI"


def names(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


unseen = names(E / "sim_val_twinval.txt") + names(E / "sim_val_single.txt")
A = sorted(p for p in unseen if "/cassava-v7-runtime-mirror/valid/" in p)
B = sorted(p for p in unseen if "/cassava-v7-runtime-mirror/train/" in p)
(E / "diag_A_cleanval_unseen.txt").write_text("\n".join(A) + "\n", encoding="utf-8")
(E / "diag_B_cleantrain_unseen.txt").write_text("\n".join(B) + "\n", encoding="utf-8")
names_yaml = "['CBB', 'CBSD', 'CGM', 'CMD', 'HEALTHY']"
for y, v in (("data_diag_A.yaml", "diag_A_cleanval_unseen.txt"), ("data_diag_B.yaml", "diag_B_cleantrain_unseen.txt")):
    (E / y).write_text(f"path: {WIN}/eleak_sim_rq1\ntrain: sim_train.txt\nval: {v}\n\nnc: 5\nnames: {names_yaml}\n", encoding="utf-8")
summary = {"unseen_total": len(unseen), "diag_A_cleanval_unseen": len(A), "diag_B_cleantrain_unseen": len(B)}
(E / "eleak_diag_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

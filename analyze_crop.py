#!/usr/bin/env python3
"""
Analisis log context crop EXP-2 dan EXP-3 (Pakem v1.4, Fase 5 A4, RQ3/RQ4)
==========================================================================
Membaca crop_logs/crop_events_pid*.csv dan epoch_marks.csv dari 6 run (EXP-2, EXP-3 x 3 seed).
Aturan crop (stage_8_3_adaptive_pipeline.py): jendela = 2 x kotak lesi target (dibatasi ukuran citra, minimum 4 px),
lalu di-resize ke 640 x 640 (skala x dan y terpisah). Jadi lesi target menempati sekitar setengah lebar dan tinggi
citra hasil crop (sekitar 320 px), berapa pun ukuran aslinya.

Yang dihitung per run: jumlah crop per epoch dan porsinya terhadap 8.980 citra latih, sebaran ukuran jendela dan
faktor zoom, distorsi rasio aspek (zoom_x / zoom_y), jumlah kotak yang tersisa, citra unik yang pernah di-crop,
serta ukuran lesi target sebelum dan sesudah crop dibanding sebaran ukuran objek validasi.

Pemakaian: python analyze_crop.py
Keluaran: stage_v2_ablation/analysis/crop/ (CROP_REPORT.md, crop_results.json)
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = [(s, e) for e in ("exp_2_targeted_aug", "exp_3_full_proposed") for s in (42, 123, 2026)]
LABEL = {"exp_2_targeted_aug": "EXP-2", "exp_3_full_proposed": "EXP-3"}
N_TRAIN = 8980
IMGSZ = 640


def q(x, p):
    return float(np.percentile(x, p))


def main() -> None:
    out = ROOT / "stage_v2_ablation/analysis/crop"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv")
    obj = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_object_difficulty.csv", usecols=["image_id", "class_name"])
    dom = obj.groupby("image_id").class_name.agg(lambda s: s.mode().iat[0])
    G = pd.read_csv(ROOT / "stage_v2_ablation/per_image/gt_cleanval.csv")
    g_side = np.sqrt((G.x2 - G.x1) * (G.y2 - G.y1))
    res, rows, kunci = {}, [], {}
    for s, e in RUNS:
        d = ROOT / f"stage_v2_ablation/seed_{s}/{e}/crop_logs"
        D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(d / "crop_events_pid*.csv")))], ignore_index=True)
        D = D[D.event == "applied"].copy()
        marks = pd.read_csv(d / "epoch_marks.csv").sort_values("epoch")
        D["epoch"] = np.searchsorted(marks.time.to_numpy(), D.time.to_numpy(), side="right")
        kunci[(LABEL[e], s)] = Counter(zip(D.epoch, D.image_id, D.crop_w.round(1), D.crop_h.round(1)))
        per_ep = D.groupby("epoch").size()
        zoom = np.sqrt(D.zoom_x * D.zoom_y)                # zoom rata-rata geometrik
        aspek = D.zoom_x / D.zoom_y
        target_side = np.sqrt((D.crop_w / 2) * (D.crop_h / 2))   # sisi lesi target asli (px, kanvas 640)
        target_after = target_side * zoom                   # sisi lesi setelah resize
        cls = D.image_id.map(dom).fillna("(tanpa kotak)")
        r = {
            "n_crop": int(len(D)), "n_epoch": int(marks.epoch.max()),
            "crop_per_epoch_median": float(per_ep.median()), "crop_per_epoch_min": int(per_ep.min()), "crop_per_epoch_max": int(per_ep.max()),
            "porsi_citra_latih_per_epoch_pct": float(per_ep.median() / N_TRAIN * 100),
            "citra_unik": int(D.image_id.nunique()),
            "jendela_px": {"p25": q(np.sqrt(D.crop_w * D.crop_h), 25), "median": q(np.sqrt(D.crop_w * D.crop_h), 50), "p75": q(np.sqrt(D.crop_w * D.crop_h), 75)},
            "zoom": {"p10": q(zoom, 10), "p25": q(zoom, 25), "median": q(zoom, 50), "p75": q(zoom, 75), "p90": q(zoom, 90), "maks": float(zoom.max()),
                     "pct_ge_10": float((zoom >= 10).mean() * 100), "pct_ge_32": float((zoom >= 32).mean() * 100)},
            "aspek_zoom_x_per_y": {"median": q(aspek, 50), "pct_di_luar_0_8_1_25": float(((aspek < 0.8) | (aspek > 1.25)).mean() * 100)},
            "lesi_target_sebelum_px": {"median": q(target_side, 50), "pct_kecil_lt_32": float((target_side < 32).mean() * 100)},
            "lesi_target_sesudah_px": {"median": q(target_after, 50), "pct_besar_gt_96": float((target_after > 96).mean() * 100)},
            "kotak_tersisa": {"rata2": float(D.n_boxes_kept.mean()), "pct_hanya_1": float((D.n_boxes_kept == 1).mean() * 100)},
            "kelas_citra_dicrop_pct": {k: float(v) for k, v in (cls.value_counts(normalize=True) * 100).round(1).items()},
        }
        res[f"{s}|{LABEL[e]}"] = r
        rows.append((LABEL[e], s, r))
    val = {"sisi_objek_validasi_px": {"p10": q(g_side, 10), "p25": q(g_side, 25), "median": q(g_side, 50), "p75": q(g_side, 75), "p90": q(g_side, 90)},
           "pct_objek_validasi_kecil_lt_32": float((g_side < 32).mean() * 100)}
    res["validasi"] = val
    n_target = int(man.get("has_target_small", pd.Series(dtype=int)).astype(bool).sum()) if "has_target_small" in man else None
    res["citra_latih_dengan_target_kecil"] = n_target
    # Kesamaan peristiwa crop antar seed: Jaccard multiset (epoch, citra, ukuran jendela)
    sama = {}
    for lab in ("EXP-2", "EXP-3"):
        for s2 in (123, 2026):
            a, b = kunci[(lab, 42)], kunci[(lab, s2)]
            sama[f"{lab} 42 vs {s2}"] = float(sum((a & b).values()) / sum((a | b).values()) * 100)
    res["kesamaan_peristiwa_antar_seed_pct"] = sama

    L = ["# Analisis log context crop (Fase 5 A4)", f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · 6 run (EXP-2, EXP-3 × 3 seed) · 30 epoch",
         "", "Aturan crop: jendela = 2 × kotak lesi target, di-resize ke 640 × 640 dengan skala x dan y terpisah.",
         f"Citra latih yang punya target kecil di manifest: {n_target:,}." if n_target is not None else "", "",
         "| Run | Crop total | Crop/epoch (median) | % citra latih per epoch | Citra unik | Jendela median (px) | Zoom median | Zoom p90 | Zoom ≥ 10× | Zoom ≥ 32× | Aspek terdistorsi* | Lesi target sesudah crop (median px) | Hanya 1 kotak |",
         "| --- |" + " --- |" * 12]
    for lab, s, r in rows:
        L.append(f"| {lab} seed {s} | {r['n_crop']:,} | {r['crop_per_epoch_median']:.0f} | {r['porsi_citra_latih_per_epoch_pct']:.1f}% | {r['citra_unik']:,} | "
                 f"{r['jendela_px']['median']:.1f} | {r['zoom']['median']:.1f}× | {r['zoom']['p90']:.1f}× | {r['zoom']['pct_ge_10']:.1f}% | {r['zoom']['pct_ge_32']:.1f}% | "
                 f"{r['aspek_zoom_x_per_y']['pct_di_luar_0_8_1_25']:.1f}% | {r['lesi_target_sesudah_px']['median']:.0f} | {r['kotak_tersisa']['pct_hanya_1']:.1f}% |")
    L += ["", "*Aspek terdistorsi = zoom_x / zoom_y di luar 0,8–1,25.", "",
          "## Kesamaan peristiwa crop antar seed", "",
          "Jaccard multiset peristiwa (epoch, citra, ukuran jendela) terhadap seed 42: " +
          "; ".join(f"{k} {v:.1f}%" for k, v in sama.items()) + ".", "",
          "Penyebab: build_dataloader Ultralytics 8.3.235 menyemai generator DataLoader dengan konstanta 6148914691236517205 + RANK, "
          "bukan dengan seed pelatihan. Tanpa sampler khusus, urutan citra dan seluruh augmentasi (termasuk keputusan crop) "
          "identik di ketiga seed, sehingga variasi antar seed EXP-0 dan EXP-2 hanya berasal dari inisialisasi kepala deteksi "
          "dan nondeterminisme GPU. EXP-0R, EXP-1, dan EXP-3 memakai WeightedRandomSampler yang mengikuti seed global, "
          "sehingga urutan datanya berbeda antar seed (tampak pada jumlah crop EXP-3 yang berbeda per seed).", "",
          "## Ukuran lesi: target crop vs objek validasi", "",
          f"- Lesi target sebelum crop: median sisi {rows[0][2]['lesi_target_sebelum_px']['median']:.1f} px; {rows[0][2]['lesi_target_sebelum_px']['pct_kecil_lt_32']:.1f}% < 32 px (EXP-2 seed 42).",
          f"- Lesi target sesudah crop: median sisi {rows[0][2]['lesi_target_sesudah_px']['median']:.0f} px; {rows[0][2]['lesi_target_sesudah_px']['pct_besar_gt_96']:.1f}% > 96 px (ukuran \"besar\" COCO).",
          f"- Objek validasi: median sisi {val['sisi_objek_validasi_px']['median']:.1f} px (p10 {val['sisi_objek_validasi_px']['p10']:.1f}, p90 {val['sisi_objek_validasi_px']['p90']:.1f}); "
          f"{val['pct_objek_validasi_kecil_lt_32']:.1f}% < 32 px.", "",
          "## Kelas dominan citra yang di-crop (%)", "", "| Run | " + " | ".join(["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]) + " |", "| --- |" + " --- |" * 5]
    for lab, s, r in rows:
        L.append(f"| {lab} seed {s} | " + " | ".join(f"{r['kelas_citra_dicrop_pct'].get(c, 0):.1f}" for c in ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]) + " |")
    (out / "CROP_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "crop_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

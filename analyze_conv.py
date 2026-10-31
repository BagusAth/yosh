#!/usr/bin/env python3
"""
Analisis CONV (Fase 4, Pakem v1.3.2) - EXP-0 dan EXP-1, seed 42, 30 vs 60 epoch
================================================================================
Hitung ulang mAP dari prediksi per citra (protokol sama dengan analyze_ablation.py) dan bootstrap klaster per grup
foto yang berpasangan (resampel citra yang sama untuk keempat model). Satu seed, jadi hasilnya deskriptif:
IK hanya menangkap ketidakpastian sampel citra, bukan variasi antar seed.

Masukan:
  stage_v2_ablation/per_image/preds_seed42_exp_{0_baseline,1_adaptive_sampling}.csv   (30 epoch)
  stage_v2_conv60/per_image/preds_seed42_exp_{0_baseline,1_adaptive_sampling}_e60.csv (60 epoch, early stopping)
  results.csv dan metrics_summary.json tiap run (kurva val() selama training)
Keluaran (--out, default stage_v2_conv60/analysis): CONV_REPORT.md, conv_results.json, conv_curves.csv, boot_conv.csv

Pemakaian: python analyze_conv.py [--iters 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_ablation import fast_map, split_by_class  # noqa: E402
from analyze_eleak_per_image import CLASS_NAMES, load_groups, prepare_run  # noqa: E402

RUNS = {
    "EXP-0 30ep": ("stage_v2_ablation/per_image/preds_seed42_exp_0_baseline.csv", "stage_8_ablation_clean_20261020/exp_0_baseline"),
    "EXP-1 30ep": ("stage_v2_ablation/per_image/preds_seed42_exp_1_adaptive_sampling.csv", "stage_v2_ablation/seed_42/exp_1_adaptive_sampling"),
    "EXP-0 60ep": ("stage_v2_conv60/per_image/preds_seed42_exp_0_baseline_e60.csv", "stage_v2_conv60/seed_42/exp_0_baseline"),
    "EXP-1 60ep": ("stage_v2_conv60/per_image/preds_seed42_exp_1_adaptive_sampling_e60.csv", "stage_v2_conv60/seed_42/exp_1_adaptive_sampling"),
}
CONTRASTS = [
    ("EXP-0: 60ep − 30ep", {"EXP-0 60ep": 1, "EXP-0 30ep": -1}),
    ("EXP-1: 60ep − 30ep", {"EXP-1 60ep": 1, "EXP-1 30ep": -1}),
    ("EXP-1 − EXP-0 pada 30ep", {"EXP-1 30ep": 1, "EXP-0 30ep": -1}),
    ("EXP-1 − EXP-0 pada 60ep", {"EXP-1 60ep": 1, "EXP-0 60ep": -1}),
    ("Perubahan selisih (60ep − 30ep)", {"EXP-1 60ep": 1, "EXP-0 60ep": -1, "EXP-1 30ep": -1, "EXP-0 30ep": 1}),
]


def kurva(run_dir: Path) -> pd.DataFrame:
    r = pd.read_csv(run_dir / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    return r[["epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)", "lr/pg0"]].rename(
        columns={"metrics/mAP50-95(B)": "map50_95", "metrics/mAP50(B)": "map50", "lr/pg0": "lr"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="stage_v2_conv60/analysis")
    ap.add_argument("--gt", default="stage_v2_ablation/per_image/gt_cleanval.csv")
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    runs, images = {}, None
    for k, (p, _) in RUNS.items():
        z = prepare_run(root / p, root / args.gt)
        if images is None:
            images = z["images"]
        assert np.array_equal(images, z["images"]), "himpunan citra berbeda"
        runs[k] = split_by_class(z)
    ones = np.ones(len(images), dtype=np.int64)
    point = {}
    for k, sp in runs.items():
        m50, m, pc = fast_map(sp, ones, per_class=True)
        point[k] = {"mAP50": m50 * 100, "mAP50_95": m * 100,
                    "per_class_ap50_95": {CLASS_NAMES[c]: float(np.nanmean(pc[c]) * 100) for c in range(len(CLASS_NAMES))}}

    groups = load_groups(root)
    gids = np.array([groups.get(n, n) for n in images])
    uniq, inv = np.unique(gids, return_inverse=True)
    rng = np.random.default_rng(4242)
    rows = []
    for _ in range(args.iters):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        rows.append({k: fast_map(sp, w)[1] * 100 for k, sp in runs.items()})
    B = pd.DataFrame(rows)
    B.to_csv(out / "boot_conv.csv", index=False)

    res = {"point": point, "contrasts": {}}
    for name, coef in CONTRASTS:
        d = sum(c * point[k]["mAP50_95"] for k, c in coef.items())
        b = sum(c * B[k] for k, c in coef.items())
        lo, hi = np.percentile(b, [2.5, 97.5])
        res["contrasts"][name] = {"diff": float(d), "ci95": [float(lo), float(hi)]}

    curves, meta = [], {}
    for k, (_, d) in RUNS.items():
        c = kurva(root / d)
        c.insert(0, "run", k)
        curves.append(c)
        ms = json.loads((root / d / "metrics_summary.json").read_text(encoding="utf-8"))
        best = c.loc[c.map50_95.idxmax()]
        meta[k] = {"epoch_selesai": int(c.epoch.max()), "epoch_terbaik": int(best.epoch),
                   "val_terbaik": float(best.map50_95 * 100), "lr_saat_berhenti": float(c.lr.iloc[-1]),
                   "val_summary_map50_95": float(ms["metrics_validation"]["mAP50_95"] * 100),
                   "stopped_early": bool(ms.get("stopped_early", False))}
    C = pd.concat(curves, ignore_index=True)
    C.to_csv(out / "conv_curves.csv", index=False)
    res["runs"] = meta

    L = ["# CONV (Fase 4): EXP-0 dan EXP-1, seed 42, 30 vs 60 epoch",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · {len(images):,} citra validasi bersih · bootstrap klaster {args.iters} iterasi",
         "", "Satu seed: IK hanya menangkap ketidakpastian sampel citra. Hasil dilaporkan deskriptif, tidak memakai aturan keputusan tiga seed.",
         "", "## Jalannya training (val() Ultralytics selama training)", "",
         "| Run | Epoch selesai | Epoch terbaik | mAP@50–95 terbaik (val) | LR saat berhenti | Early stopping |", "| --- | --- | --- | --- | --- | --- |"]
    for k, m in meta.items():
        L.append(f"| {k} | {m['epoch_selesai']} | {m['epoch_terbaik']} | {m['val_terbaik']:.2f} | {m['lr_saat_berhenti']:.6f} | {'ya' if m['stopped_early'] else 'tidak'} |")
    L += ["", "## mAP hitung ulang (bobot best.pt, prediksi per citra)", "",
          "| Run | mAP@50 | mAP@50–95 | " + " | ".join(CLASS_NAMES) + " |", "| --- | --- | --- |" + " --- |" * len(CLASS_NAMES)]
    for k, p in point.items():
        L.append(f"| {k} | {p['mAP50']:.2f} | {p['mAP50_95']:.2f} | " + " | ".join(f"{p['per_class_ap50_95'][c]:.2f}" for c in CLASS_NAMES) + " |")
    L += ["", "## Selisih (poin mAP@50–95, hitung ulang)", "", "| Perbandingan | Selisih | IK 95% (sampel citra) |", "| --- | --- | --- |"]
    for name, r in res["contrasts"].items():
        L.append(f"| {name} | {r['diff']:+.2f} | {r['ci95'][0]:+.2f} s.d. {r['ci95'][1]:+.2f} |")
    L += ["", "Catatan: run 60 epoch memakai jadwal LR dan close_mosaic (6 epoch terakhir) untuk 60 epoch. Early stopping (patience 10)",
          "menghentikan run sebelum LR habis dan sebelum fase tanpa mosaic, sehingga run 60 epoch bukan versi panjang yang tuntas dari run 30 epoch."]
    (out / "CONV_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "conv_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

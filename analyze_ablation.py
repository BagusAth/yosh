#!/usr/bin/env python3
"""
Analisis ablasi multi-seed (Fase 3/5, Pakem v1.3.1) — mAP dari prediksi per citra + bootstrap klaster berpasangan
===============================================================================================================
Semua model dinilai pada 2.567 citra validasi bersih yang sama dengan protokol yang sama
(prediksi 640 px, NMS seperti val(), AP 101 titik pada 10 ambang IoU seperti Ultralytics).

Mode:
  prep   : baca stage_v2_ablation/per_image/preds_seed<seed>_<exp>.csv + gt_cleanval.csv, simpan cache .npz
  boot   : bootstrap klaster per grup foto; satu resampel citra dipakai bersama untuk SEMUA model (berpasangan)
  report : tabel per model, selisih antar strategi (rata-rata antar seed), IK 95%, dan verdict aturan keputusan

Aturan keputusan (TERKUNCI di pakem): "meningkatkan secara konsisten" bila selisih positif di ketiga seed dan
IK 95% tidak memuat nol; "menurunkan secara konsisten" kebalikannya; selain itu "tidak terbukti berbeda".
Dengan kurang dari 3 seed, verdict ditandai SEMENTARA.

Pemakaian: python analyze_ablation.py prep ; python analyze_ablation.py boot --seed 0 --iters 250 ; python analyze_ablation.py report
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_eleak_per_image import CLASS_NAMES, load_groups, prepare_run  # noqa: E402

EXPS = [("exp_0_baseline", "EXP-0"), ("exp_0r_uniform_replacement", "EXP-0R"), ("exp_1_adaptive_sampling", "EXP-1"),
        ("exp_2_targeted_aug", "EXP-2"), ("exp_3_full_proposed", "EXP-3")]
LABEL = dict(EXPS)
SEEDS = [42, 123, 2026]
# (nama, {label: koefisien}, arti)
CONTRASTS = [
    ("EXP-0R − EXP-0", {"EXP-0R": 1, "EXP-0": -1}, "Efek replacement saja (kontrol mekanisme)"),
    ("EXP-1 − EXP-0R", {"EXP-1": 1, "EXP-0R": -1}, "RQ2 inti: efek pembobotan kesulitan"),
    ("EXP-1 − EXP-0", {"EXP-1": 1, "EXP-0": -1}, "RQ2: sampling adaptif vs baseline"),
    ("EXP-2 − EXP-0", {"EXP-2": 1, "EXP-0": -1}, "RQ3: context crop vs baseline (aturan keputusan bingkai)"),
    ("EXP-3 − EXP-0", {"EXP-3": 1, "EXP-0": -1}, "Gabungan vs baseline"),
    ("EXP-3 − EXP-2", {"EXP-3": 1, "EXP-2": -1}, "Menambah sampling adaptif di atas crop"),
    ("EXP-3 − EXP-1", {"EXP-3": 1, "EXP-1": -1}, "Menambah crop di atas sampling adaptif"),
    ("Interaksi", {"EXP-3": 1, "EXP-2": -1, "EXP-1": -1, "EXP-0": 1}, "(EXP-3 − EXP-2) − (EXP-1 − EXP-0)"),
]
EPS = 1e-16
X101 = np.linspace(0, 1, 101)
_trap = getattr(np, "trapezoid", None) or np.trapz


def split_by_class(run: dict, nc: int = 5) -> dict:
    pc = []
    for c in range(nc):
        m = run["det_cls"] == c
        pc.append({"img": run["det_img"][m], "tp": run["det_tp"][m].astype(np.float64)})
    return {"pc": pc, "gt_counts": run["gt_counts"]}


def fast_map(sp: dict, img_w: np.ndarray, per_class: bool = False):
    """Setara map_weighted di analyze_eleak_per_image (diuji identik), sekitar 2x lebih cepat."""
    nl = (sp["gt_counts"] * img_w[:, None]).sum(0)
    classes = np.nonzero(nl > 0)[0]
    ap = np.zeros((len(classes), 10))
    for ci, c in enumerate(classes):
        d = sp["pc"][c]
        if len(d["img"]) == 0:
            continue
        w = img_w[d["img"]].astype(np.float64)
        k = w > 0
        if not k.any():
            continue
        w = w[k]
        tpc = np.cumsum(d["tp"][k] * w[:, None], axis=0)
        cw = np.cumsum(w)
        n = len(w)
        mpre = np.empty((n + 2, 10)); mpre[0] = 1; mpre[1:-1] = tpc / cw[:, None]; mpre[-1] = 0
        mpre = np.maximum.accumulate(mpre[::-1], axis=0)[::-1]
        mrec = np.empty((n + 2, 10)); mrec[0] = 0; mrec[1:-1] = tpc / (nl[c] + EPS); mrec[-1] = 1
        for j in range(10):
            ap[ci, j] = _trap(np.interp(X101, mrec[:, j], mpre[:, j]), X101)
    if per_class:
        full = np.full((sp["gt_counts"].shape[1], 10), np.nan)
        full[classes] = ap
        return float(ap[:, 0].mean()), float(ap.mean()), full
    return float(ap[:, 0].mean()), float(ap.mean())


def model_keys(cache_dir: Path) -> list[tuple[int, str]]:
    keys = []
    for s in SEEDS:
        for e, lab in EXPS:
            if (cache_dir / f"seed{s}_{e}.npz").exists():
                keys.append((s, lab))
    return keys


def load_models(cache_dir: Path):
    keys = model_keys(cache_dir)
    inv = {lab: e for e, lab in EXPS}
    runs = {}
    images = None
    for s, lab in keys:
        z = dict(np.load(cache_dir / f"seed{s}_{inv[lab]}.npz", allow_pickle=False))
        if images is None:
            images = z["images"]
        assert np.array_equal(images, z["images"]), "Himpunan citra antar model berbeda"
        runs[(s, lab)] = split_by_class(z)
    return keys, runs, images


def read_val_metrics(root: Path) -> dict:
    out = {}
    for s in SEEDS:
        for e, lab in EXPS:
            d = root / "stage_8_ablation_clean_20261020" / e if (s == 42 and e == "exp_0_baseline") else root / "stage_v2_ablation" / f"seed_{s}" / e
            f = d / "metrics_summary.json"
            if f.exists():
                m = json.loads(f.read_text(encoding="utf-8"))["metrics_validation"]
                out[f"{s}|{lab}"] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "boot", "report"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--per-image-dir", default="stage_v2_ablation/per_image")
    ap.add_argument("--out", default="stage_v2_ablation/analysis")
    ap.add_argument("--val-json", default=None, help="JSON angka val() bila folder run tidak tersedia")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=250)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    pdir = root / args.per_image_dir
    out = root / args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    if args.mode == "prep":
        gt = pdir / "gt_cleanval.csv"
        for s in SEEDS:
            for e, lab in EXPS:
                f = pdir / f"preds_seed{s}_{e}.csv"
                dst = cache / f"seed{s}_{e}.npz"
                if f.exists() and (not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime):
                    np.savez_compressed(dst, **prepare_run(f, gt))
                    print(f"cache: {dst.name}")
        keys, runs, images = load_models(cache)
        ones = np.ones(len(images), dtype=np.int64)
        pt = {}
        for k in keys:
            m50, m, pc = fast_map(runs[k], ones, per_class=True)
            pt[f"{k[0]}|{k[1]}"] = {"mAP50": m50, "mAP50_95": m, "per_class_ap50_95": {CLASS_NAMES[c]: float(np.nanmean(pc[c])) for c in range(len(CLASS_NAMES))}}
        val = json.loads(Path(args.val_json).read_text()) if args.val_json else read_val_metrics(root)
        (out / "point.json").write_text(json.dumps({"recomputed": pt, "val": val, "n_images": int(len(images))}, indent=2), encoding="utf-8")
        print(f"{len(keys)} model, {len(images)} citra")
        return

    if args.mode == "boot":
        keys, runs, images = load_models(cache)
        groups = load_groups(root)
        gids = np.array([groups.get(n, n) for n in images])
        uniq, inv = np.unique(gids, return_inverse=True)
        rng = np.random.default_rng(9000 + args.seed)
        rows = []
        for _ in range(args.iters):
            cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            w = cnt[inv]
            r = {}
            for k in keys:
                m50, m = fast_map(runs[k], w)
                r[f"{k[0]}|{k[1]}|m50"], r[f"{k[0]}|{k[1]}|m"] = m50, m
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / f"boot_seed{args.seed}.csv", index=False)
        print(f"boot seed {args.seed}: {len(rows)} iterasi, {len(keys)} model")
        return

    # ---------------------------------------------------------------- report
    P = json.loads((out / "point.json").read_text(encoding="utf-8"))
    rec, val = P["recomputed"], P["val"]
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "boot_seed*.csv")))], ignore_index=True)
    seeds_avail = sorted({int(k.split("|")[0]) for k in rec})
    labs = [lab for _, lab in EXPS]
    lengkap = all(f"{s}|{lab}" in rec for s in SEEDS for lab in labs)
    common_all = [s for s in seeds_avail if all(f"{s}|{lab}" in rec for lab in labs)] or seeds_avail
    L = [f"# Ablasi multi-seed — hasil {'' if lengkap else 'SEMENTARA '}(Fase 3)",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · seed tersedia: {', '.join(map(str, seeds_avail))} · "
         f"{P['n_images']:,} citra validasi bersih · bootstrap klaster {len(B)} iterasi", "",
         "## mAP@50–95 per model (hitung ulang dari prediksi per citra; angka val() Ultralytics dalam kurung)", "",
         "| Model | " + " | ".join(f"Seed {s}" for s in seeds_avail) + " | Rata-rata ± SD |", "| --- |" + " --- |" * (len(seeds_avail) + 1)]
    for lab in labs:
        cells, vals = [], []
        for s in seeds_avail:
            k = f"{s}|{lab}"
            if k in rec:
                v = rec[k]["mAP50_95"] * 100
                vals.append(v)
                vv = f" ({val[k]['mAP50_95']*100:.2f})" if k in val else ""
                cells.append(f"{v:.2f}{vv}")
            else:
                cells.append("—")
        ms = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}" if len(vals) > 1 else (f"{vals[0]:.2f}" if vals else "—")
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {ms} |")
    L += ["", "## Selisih antar strategi (poin mAP@50–95; rata-rata seed yang tersedia untuk kedua sisi)", "",
          "| Perbandingan | Arti | Per seed | Rata-rata | IK 95% | Selisih val() | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    res = {}
    for name, coef, arti in CONTRASTS:
        common = [s for s in seeds_avail if all(f"{s}|{lab}" in rec for lab in coef)]
        if not common:
            continue
        per_seed = [sum(c * rec[f"{s}|{lab}"]["mAP50_95"] for lab, c in coef.items()) * 100 for s in common]
        boot = np.mean([sum(c * B[f"{s}|{lab}|m"] for lab, c in coef.items()) for s in common], axis=0) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        vdiff = [sum(c * val[f"{s}|{lab}"]["mAP50_95"] for lab, c in coef.items()) * 100 for s in common if all(f"{s}|{lab}" in val for lab in coef)]
        mean = float(np.mean(per_seed))
        if all(d > 0 for d in per_seed) and lo > 0:
            verdict = "meningkatkan secara konsisten"
        elif all(d < 0 for d in per_seed) and hi < 0:
            verdict = "menurunkan secara konsisten"
        else:
            verdict = "tidak terbukti berbeda"
        if len(common) < 3:
            verdict += " (sementara)"
        res[name] = {"per_seed": dict(zip(common, per_seed)), "mean": mean, "ci95": [float(lo), float(hi)], "val_diff": vdiff, "verdict": verdict}
        L.append(f"| {name} | {arti} | " + " / ".join(f"{d:+.2f}" for d in per_seed) + f" | {mean:+.2f} | {lo:+.2f} s.d. {hi:+.2f} | "
                 + (" / ".join(f"{d:+.2f}" for d in vdiff) if vdiff else "—") + f" | {verdict} |")
    L += ["", f"## AP@50–95 per kelas (hitung ulang, rata-rata seed {', '.join(map(str, common_all))} agar sebanding)", "", "| Model | " + " | ".join(CLASS_NAMES) + " |", "| --- |" + " --- |" * len(CLASS_NAMES)]
    for lab in labs:
        ks = [f"{s}|{lab}" for s in common_all if f"{s}|{lab}" in rec]
        if ks:
            L.append(f"| {lab} | " + " | ".join(f"{np.mean([rec[k]['per_class_ap50_95'][c] for k in ks])*100:.2f}" for c in CLASS_NAMES) + " |")
    L += ["", "Catatan:",
          "- IK 95% berasal dari bootstrap klaster per grup foto (resampel citra yang sama untuk semua model), lalu dirata-rata antar seed. "
          "IK ini menangkap ketidakpastian sampel citra; variasi antar seed terlihat dari kolom per seed.",
          "- Angka val() Ultralytics menjalankan citra 640 px dalam bingkai 672 px (rect, pad 0,5) dan presisi setengah saat training, "
          "sehingga berbeda 0,3–0,9 poin dari hitung ulang. Metrik utama Fase 5 adalah hitung ulang (protokol sama untuk semua model)."]
    (out / "ABLATION_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "ablation_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

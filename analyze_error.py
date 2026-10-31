#!/usr/bin/env python3
"""
Pergeseran jenis error antar strategi (Pakem v1.4, Fase 5 A2, RQ4)
==================================================================
Memakai pencocokan kanonis v2 yang TERKUNCI (research_utils.detection_metrics.match_detections_v2: tau = 0,25,
TP pada IoU >= 0,5 sekelas, kandidat error pada IoU >= 0,10) pada prediksi validasi 15 model.

Per objek GT: True_Positive, Localization_Error, Classification_Confusion, Localization_and_Class_Error,
False_Negative_Missed. Per prediksi conf >= 0,25 yang bukan TP (false positive):
  fp_latar      : IoU < 0,10 dengan semua GT (deteksi di latar)
  fp_sekelas    : IoU >= 0,10 dengan GT sekelas (duplikat atau kotak meleset)
  fp_kelas_lain : hanya bersinggungan (IoU >= 0,10) dengan GT kelas lain (salah kelas)

Bootstrap klaster per grup foto (1.000 iterasi, resampel sama untuk semua model) untuk selisih proporsi.
Pemakaian: python analyze_error.py prep ; python analyze_error.py report [--iters 1000]
Keluaran: stage_v2_ablation/analysis/error/ (ERROR_REPORT.md, error_results.json, cache/)
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
from analyze_ablation import EXPS, SEEDS  # noqa: E402
from analyze_eleak_per_image import CLASS_NAMES, iou_matrix, load_groups  # noqa: E402
from research_utils.detection_metrics import TAXONOMY_ORDER, match_detections_v2  # noqa: E402

TAU = 0.25
NC = len(CLASS_NAMES)
TAX_ID = {t: i for i, t in enumerate(TAXONOMY_ORDER)}
TAX_ID_NAMA = ["TP", "kotak meleset", "salah kelas", "meleset + salah kelas", "tidak terdeteksi"]
FP_NAMA = ["fp_latar", "fp_sekelas", "fp_kelas_lain"]
CONTRASTS = [("EXP-1 − EXP-0R", {"EXP-1": 1, "EXP-0R": -1}), ("EXP-1 − EXP-0", {"EXP-1": 1, "EXP-0": -1}),
             ("EXP-2 − EXP-0", {"EXP-2": 1, "EXP-0": -1}), ("EXP-3 − EXP-1", {"EXP-3": 1, "EXP-1": -1})]


def prep_model(preds: Path, G: pd.DataFrame, images: list[str]) -> dict:
    P = pd.read_csv(preds)
    P = P[P.confidence >= TAU]
    pg = P.groupby("image_id").indices
    gg = G.groupby("image_id").indices
    Pb = P[["x1", "y1", "x2", "y2"]].to_numpy(float); Pc = P.class_id.to_numpy(int); Pf = P.confidence.to_numpy(float)
    Gb = G[["x1", "y1", "x2", "y2"]].to_numpy(float); Gc = G.class_id.to_numpy(int)
    tax = np.zeros((len(images), NC, 5), dtype=np.int32)   # citra x kelas GT x taksonomi
    fp = np.zeros((len(images), NC, 3), dtype=np.int32)    # citra x kelas prediksi x jenis FP
    for i, name in enumerate(images):
        gi = gg.get(name, np.array([], dtype=int)); pi = pg.get(name, np.array([], dtype=int))
        gts = [{"box_xyxy": Gb[k].tolist(), "class_id": int(Gc[k])} for k in gi]
        prs = [{"box_xyxy": Pb[k].tolist(), "class_id": int(Pc[k]), "confidence": float(Pf[k])} for k in pi]
        if gts:
            m = match_detections_v2(gts, prs, tau=TAU, tp_iou=0.50, cand_iou=0.10)
            for g, rec in zip(gts, m["per_gt"]):
                tax[i, g["class_id"], TAX_ID[rec["taxonomy"]]] += 1
            tp_set = set(m["tp_pred_indices"])
        else:
            tp_set = set()
        if prs:
            iou = iou_matrix(Gb[gi], Pb[pi]) if len(gi) else np.zeros((0, len(pi)))
            for j, p in enumerate(prs):
                if j in tp_set:
                    continue
                col = iou[:, j] if len(gi) else np.zeros(0)
                same = (Gc[gi] == p["class_id"]) if len(gi) else np.zeros(0, bool)
                if len(gi) and (col[same] >= 0.10).any():
                    k = 1
                elif len(gi) and (col >= 0.10).any():
                    k = 2
                else:
                    k = 0
                fp[i, p["class_id"], k] += 1
    return {"tax": tax, "fp": fp}


def ringkas(z: dict, w: np.ndarray) -> dict:
    t = (z["tax"] * w[:, None, None]).sum(0).astype(float)   # kelas x 5
    f = (z["fp"] * w[:, None, None]).sum(0).astype(float)    # kelas x 3
    n_img = float(w.sum())
    r = {}
    tot = t.sum(1)
    for k, nm in enumerate(TAX_ID_NAMA):
        r[f"semua|{nm}"] = t[:, k].sum() / tot.sum() * 100
        for c, cn in enumerate(CLASS_NAMES):
            r[f"{cn}|{nm}"] = t[c, k] / tot[c] * 100 if tot[c] else np.nan
    for k, nm in enumerate(FP_NAMA):
        r[f"semua|{nm}"] = f[:, k].sum() / n_img
        for c, cn in enumerate(CLASS_NAMES):
            r[f"{cn}|{nm}"] = f[c, k] / n_img
    r["semua|fp_total"] = f.sum() / n_img
    for c, cn in enumerate(CLASS_NAMES):
        r[f"{cn}|fp_total"] = f[c].sum() / n_img
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "report"])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="stage_v2_ablation/analysis/error")
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = root / args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    pdir = root / "stage_v2_ablation/per_image"
    G = pd.read_csv(pdir / "gt_cleanval.csv")
    keys = [(s, e, lab) for s in SEEDS for e, lab in EXPS]
    f_img = cache / "images.json"
    if not f_img.exists():
        imgs = set(G.image_id)
        for s, e, _ in keys:
            imgs |= set(pd.read_csv(pdir / f"preds_seed{s}_{e}.csv", usecols=["image_id"]).image_id)
        f_img.write_text(json.dumps(sorted(imgs)))
    images = json.loads(f_img.read_text())

    if args.mode == "prep":
        for s, e, lab in keys:
            dst = cache / f"seed{s}_{e}.npz"
            if not dst.exists():
                np.savez_compressed(dst, **prep_model(pdir / f"preds_seed{s}_{e}.csv", G, images))
                print(f"cache {dst.name}", flush=True)
        return

    Z = {(s, lab): dict(np.load(cache / f"seed{s}_{e}.npz")) for s, e, lab in keys}
    ones = np.ones(len(images), dtype=np.int64)
    point = {f"{s}|{lab}": ringkas(z, ones) for (s, lab), z in Z.items()}
    groups = load_groups(root)
    gids = np.array([groups.get(n, n) for n in images])
    uniq, inv = np.unique(gids, return_inverse=True)
    rng = np.random.default_rng(6200)
    want = [f"{a}|{b}" for a in ["semua"] + CLASS_NAMES for b in ["TP", "tidak terdeteksi", "kotak meleset", "salah kelas", "fp_total", "fp_latar", "fp_sekelas", "fp_kelas_lain"]]
    labs_needed = sorted({lab for _, coef in CONTRASTS for lab in coef})
    boots = {k: [] for k in want}
    store = {(s, lab): {k: [] for k in want} for s in SEEDS for lab in labs_needed}
    for _ in range(args.iters):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        for s in SEEDS:
            for lab in labs_needed:
                r = ringkas(Z[(s, lab)], w)
                for k in want:
                    store[(s, lab)][k].append(r[k])
    labs = [lab for _, lab in EXPS]
    L = ["# Pergeseran jenis error (Fase 5 A2)",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · {len(images):,} citra validasi bersih · pencocokan v2 (tau 0,25) · bootstrap klaster {args.iters} iterasi",
         "", "## Komposisi objek GT menurut taksonomi (%, rata-rata tiga seed)", "",
         "| Model | Kelas | TP | Tidak terdeteksi | Kotak meleset | Salah kelas | Meleset + salah kelas |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for lab in labs:
        for cn in ["semua"] + CLASS_NAMES:
            v = lambda k: np.nanmean([point[f"{s}|{lab}"][f"{cn}|{k}"] for s in SEEDS])
            L.append(f"| {lab} | {cn} | {v('TP'):.1f} | {v('tidak terdeteksi'):.1f} | {v('kotak meleset'):.1f} | {v('salah kelas'):.1f} | {v('meleset + salah kelas'):.1f} |")
    L += ["", "## False positive per citra (conf ≥ 0,25, rata-rata tiga seed)", "",
          "| Model | Kelas prediksi | Total | Latar | Sekelas (duplikat/meleset) | Kelas lain |", "| --- | --- | --- | --- | --- | --- |"]
    for lab in labs:
        for cn in ["semua"] + CLASS_NAMES:
            v = lambda k: np.nanmean([point[f"{s}|{lab}"][f"{cn}|{k}"] for s in SEEDS])
            L.append(f"| {lab} | {cn} | {v('fp_total'):.3f} | {v('fp_latar'):.3f} | {v('fp_sekelas'):.3f} | {v('fp_kelas_lain'):.3f} |")
    res = {"point": point, "contrasts": {}}
    L += ["", "## Kontras (poin persen untuk proporsi GT; FP per citra untuk fp_*)", "",
          "| Kontras | Kelas | Metrik | Per seed (42 / 123 / 2026) | Rata-rata | IK 95% | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, coef in CONTRASTS:
        for k in want:
            per = [sum(c * point[f"{s}|{lab}"][k] for lab, c in coef.items()) for s in SEEDS]
            boot = np.mean([sum(c * np.array(store[(s, lab)][k]) for lab, c in coef.items()) for s in SEEDS], axis=0)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            if all(d > 0 for d in per) and lo > 0:
                vd = "naik secara konsisten"
            elif all(d < 0 for d in per) and hi < 0:
                vd = "turun secara konsisten"
            else:
                vd = "tidak terbukti berbeda"
            res["contrasts"][f"{name}|{k}"] = {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)], "verdict": vd}
            kelas, met = k.split("|")
            fmt = "{:+.3f}" if met.startswith("fp") else "{:+.2f}"
            L.append(f"| {name} | {kelas} | {met} | " + " / ".join(fmt.format(d) for d in per) + f" | {fmt.format(np.mean(per))} | "
                     f"{fmt.format(lo)} s.d. {fmt.format(hi)} | {vd} |")
    (out / "ERROR_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "error_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()

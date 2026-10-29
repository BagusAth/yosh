#!/usr/bin/env python3
"""
E-LEAK uji mekanisme (RQ1, Pakem v1.3) — selisih-dalam-selisih pada citra yang sama
==================================================================================
Memakai cache dari analyze_eleak_per_image.py (prediksi kedua model sudah tersimpan).

  L_cv : citra validasi BERSIH yang di split acak masuk validasi dan kembarannya masuk latih.
         Model split acak sudah melihat kembarannya; model split per grup belum.
  A    : citra validasi bersih yang tidak terlihat kedua model (kembarannya pun tidak).

  d_twin = [sim - group](L_cv) - [sim - group](A)  -> efek melihat kembaran, bersih dari beda model.
Bootstrap klaster per grup foto, berpasangan antar model.

Pemakaian: python analyze_eleak_did.py run --seed 0 --iters 250 ; python analyze_eleak_did.py report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_eleak_per_image import cluster_weights, load_groups, map_weighted, names_from_list, op_means  # noqa: E402

ROOT = Path(__file__).resolve().parent
PDIR = ROOT / os.environ.get("ELEAK_PER_IMAGE_DIR", "stage_v2_eleak_sim/per_image")  # per_image_v2 = prediksi NMS seperti val
OUT = PDIR / "analysis"


def slice_run(run: dict, keep_names: set[str]) -> dict:
    keep_img = np.array([n in keep_names for n in run["images"]])
    new_idx = -np.ones(len(keep_img), dtype=np.int64)
    new_idx[keep_img] = np.arange(keep_img.sum())
    dmask = keep_img[run["det_img"]]
    return {"images": run["images"][keep_img], "det_img": new_idx[run["det_img"][dmask]], "det_cls": run["det_cls"][dmask],
            "det_conf": run["det_conf"][dmask], "det_tp": run["det_tp"][dmask], "gt_counts": run["gt_counts"][keep_img],
            "op": run["op"][keep_img]}


def setup(control: str = "A"):
    z = np.load(PDIR / "analysis_cache.npz", allow_pickle=False)
    sim = {k[4:]: z[k] for k in z.files if k.startswith("sim_")}
    grp = {k[4:]: z[k] for k in z.files if k.startswith("grp_")}
    E = ROOT / "eleak_sim_rq1"
    clean_val = set(grp["images"])
    leaked = names_from_list(E / "sim_val_leaked.txt")
    L_cv = leaked & clean_val
    if control == "TW":  # grup 2 salinan yang KEDUA salinannya jatuh di validasi split acak (struktur grup sama dengan L_cv)
        A = names_from_list(E / "sim_val_twinval.txt") & clean_val
    else:
        A = names_from_list(E / "diag_A_cleanval_unseen.txt")
    keep = L_cv | A
    sim_s, grp_s = slice_run(sim, keep), slice_run(grp, keep)
    return sim_s, grp_s, L_cv, A, load_groups(ROOT)


def mask(run, names):
    return np.array([n in names for n in run["images"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--control", choices=["A", "TW"], default="A", help="A = diag_A (227); TW = grup 2 salinan, keduanya di validasi split acak")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sim, grp, L_cv, A, groups = setup(args.control)
    suf = "" if args.control == "A" else "_tw"
    clab = "A (tidak terlihat kedua model)" if args.control == "A" else "TW (grup 2 salinan, keduanya di validasi split acak)"
    mL_s, mL_g, mA_s, mA_g = mask(sim, L_cv), mask(grp, L_cv), mask(sim, A), mask(grp, A)

    def stats(ws_L, wg_L, ws_A, wg_A):
        r = {}
        for key, run, w in (("sim_L", sim, ws_L), ("grp_L", grp, wg_L), ("sim_A", sim, ws_A), ("grp_A", grp, wg_A)):
            m50, m5095 = map_weighted(run, w)
            r[f"{key}_m50"], r[f"{key}_m5095"] = m50, m5095
            r[f"{key}_f1"] = op_means(run, w)["f1_img"]
            r[f"{key}_rec"] = op_means(run, w)["recall_img"]
        return r

    if args.mode == "run":
        if args.seed == 0:
            pt = stats(mL_s.astype(int), mL_g.astype(int), mA_s.astype(int), mA_g.astype(int))
            pt["n_L_cv"], pt["n_A"] = int(mL_s.sum()), int(mA_s.sum())
            (OUT / f"did_point{suf}.json").write_text(json.dumps(pt, indent=2), encoding="utf-8")
        rng = np.random.default_rng(5000 + args.seed)
        rows = []
        for _ in range(args.iters):
            shL, shA = {}, {}
            rows.append(stats(cluster_weights(sim["images"], mL_s, groups, rng, shL), cluster_weights(grp["images"], mL_g, groups, rng, shL),
                              cluster_weights(sim["images"], mA_s, groups, rng, shA), cluster_weights(grp["images"], mA_g, groups, rng, shA)))
        pd.DataFrame(rows).to_csv(OUT / f"did_boot{suf}_seed{args.seed}.csv", index=False)
        print(f"seed {args.seed}: {len(rows)} iterasi")
        return

    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / f"did_boot{suf}_seed*.csv")))], ignore_index=True)
    pt = json.loads((OUT / f"did_point{suf}.json").read_text(encoding="utf-8"))
    lines = ["# E-LEAK — uji efek kembaran pada citra yang sama (selisih-dalam-selisih)", f"Tanggal: {datetime.now():%Y-%m-%d %H:%M}",
             f"Iterasi bootstrap: {len(B)}; L_cv = {pt['n_L_cv']} citra, kontrol {args.control} = {pt['n_A']} citra", "",
             "| Subset | Model split acak | Model split per grup |", "| --- | --- | --- |"]
    for sub, lab in (("L", "L_cv (kembaran dilihat model split acak)"), ("A", clab)):
        lines.append(f"| {lab}: mAP@50–95 | {pt[f'sim_{sub}_m5095']*100:.2f}% | {pt[f'grp_{sub}_m5095']*100:.2f}% |")
        lines.append(f"| {lab}: F1 per citra | {pt[f'sim_{sub}_f1']*100:.2f}% | {pt[f'grp_{sub}_f1']*100:.2f}% |")
    lines += ["", "| Selisih (poin) | Rata-rata | IK 95% |", "| --- | --- | --- |"]
    res = {}
    for metric, lab in (("m5095", "mAP@50–95"), ("m50", "mAP@50"), ("f1", "F1 per citra"), ("rec", "Recall per citra")):
        dL = B[f"sim_L_{metric}"] - B[f"grp_L_{metric}"]
        dA = B[f"sim_A_{metric}"] - B[f"grp_A_{metric}"]
        for name, v in ((f"{lab}: sim − group pada L_cv", dL), (f"{lab}: sim − group pada {args.control}", dA), (f"{lab}: efek kembaran (L_cv − {args.control})", dL - dA)):
            res[name] = [float(v.mean() * 100), float(v.quantile(0.025) * 100), float(v.quantile(0.975) * 100)]
            lines.append(f"| {name} | {res[name][0]:+.2f} | [{res[name][1]:+.2f}, {res[name][2]:+.2f}] |")
    (OUT / f"ELEAK_DID{suf}_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"eleak_did{suf}_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

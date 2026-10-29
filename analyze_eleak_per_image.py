#!/usr/bin/env python3
"""
E-LEAK per citra (RQ1, Pakem v1.3) — mAP gaya Ultralytics, metrik per citra, dan bootstrap klaster
==================================================================================================
Mode:
  prep   : baca CSV prediksi + GT, cocokkan per citra (aturan match_predictions Ultralytics, 10 ambang IoU),
           simpan array siap-bootstrap (.npz) dan estimasi titik.
  boot   : bootstrap klaster per grup foto (pasangan kembaran ikut diresampel bersama). Bisa dipecah per seed.
  report : gabungkan hasil bootstrap, tulis ELEAK_PER_IMAGE_REPORT.md dan .json.

Perbandingan utama (mAP@50-95, juga mAP@50):
  d_same   = sim(A) - group(A)                    berpasangan: citra sama, tidak terlihat kedua model
  d_leak   = sim(bocor) - sim(kembar-di-validasi) populasi sama, model sama
  d_report = sim(val_all) - group(val_all)        angka yang dilaporkan
  d_infl   = d_report - d_same                    inflasi relatif terhadap kemampuan sebenarnya
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

IOUV = np.linspace(0.5, 0.95, 10)
NC = 5
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
EPS = 1e-16


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_predictions(gt_cls, pr_cls, iou, iouv=IOUV) -> np.ndarray:
    """Replika ultralytics DetectionValidator.match_predictions (iou: n_gt x n_pred)."""
    correct = np.zeros((len(pr_cls), len(iouv)), dtype=bool)
    if len(gt_cls) == 0 or len(pr_cls) == 0:
        return correct
    iou = iou * (gt_cls[:, None] == pr_cls[None, :])
    for i, thr in enumerate(iouv):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.interp(x, mrec, mpre), x))


def prepare_run(preds_csv: str, gt_csv: str, tau: float = 0.25) -> dict:
    P = pd.read_csv(preds_csv)
    G = pd.read_csv(gt_csv)
    images = sorted(set(G.image_id) | set(P.image_id))
    pg = P.groupby("image_id").indices
    gg = G.groupby("image_id").indices
    Pb = P[["x1", "y1", "x2", "y2"]].to_numpy(float); Pc = P.class_id.to_numpy(int); Pf = P.confidence.to_numpy(float)
    Gb = G[["x1", "y1", "x2", "y2"]].to_numpy(float); Gc = G.class_id.to_numpy(int)
    det_img, det_cls, det_conf, det_tp = [], [], [], []
    gt_counts = np.zeros((len(images), NC), dtype=np.int64)
    op = np.zeros((len(images), 3), dtype=np.int64)  # TP, FP, FN pada conf >= tau, IoU 0.5
    for k, im in enumerate(images):
        gi = gg.get(im, np.array([], dtype=int)); pi = pg.get(im, np.array([], dtype=int))
        gcls = Gc[gi]; pcls = Pc[pi]
        for c in gcls:
            gt_counts[k, c] += 1
        iou = iou_matrix(Gb[gi], Pb[pi])
        cor = match_predictions(gcls, pcls, iou)
        det_img.append(np.full(len(pi), k)); det_cls.append(pcls); det_conf.append(Pf[pi]); det_tp.append(cor)
        hi = Pf[pi] >= tau
        cor50 = match_predictions(gcls, pcls[hi], iou[:, hi], iouv=np.array([0.5]))[:, 0]
        tp = int(cor50.sum()); op[k] = (tp, int(hi.sum()) - tp, len(gi) - tp)
    det_img = np.concatenate(det_img) if det_img else np.zeros(0, int)
    det_cls = np.concatenate(det_cls) if det_cls else np.zeros(0, int)
    det_conf = np.concatenate(det_conf) if det_conf else np.zeros(0)
    det_tp = np.concatenate(det_tp) if det_tp else np.zeros((0, 10), bool)
    order = np.argsort(-det_conf, kind="stable")
    return {"images": np.array(images), "det_img": det_img[order], "det_cls": det_cls[order],
            "det_conf": det_conf[order], "det_tp": det_tp[order], "gt_counts": gt_counts, "op": op}


def map_weighted(run: dict, img_w: np.ndarray) -> tuple[float, float]:
    """mAP@50 dan mAP@50-95 dengan bobot multiplisitas per citra (setara duplikasi citra)."""
    w_det = img_w[run["det_img"]]
    keep = w_det > 0
    cls = run["det_cls"][keep]; tp = run["det_tp"][keep].astype(float); w = w_det[keep].astype(float)
    nl = (run["gt_counts"] * img_w[:, None]).sum(0)
    classes = np.nonzero(nl > 0)[0]
    if len(classes) == 0:
        return float("nan"), float("nan")
    ap = np.zeros((len(classes), len(IOUV)))
    for ci, c in enumerate(classes):
        m = cls == c
        if not m.any():
            continue
        tpw = tp[m] * w[m, None]; fpw = (1 - tp[m]) * w[m, None]
        tpc = tpw.cumsum(0); fpc = fpw.cumsum(0)
        recall = tpc / (nl[c] + EPS); precision = tpc / (tpc + fpc)
        for j in range(len(IOUV)):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return float(ap[:, 0].mean()), float(ap.mean())


def op_means(run: dict, img_w: np.ndarray) -> dict:
    tp, fp, fn = (run["op"][:, i].astype(float) for i in range(3))
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)
    has_gt = (tp + fn) > 0
    rec = np.where(has_gt, tp / np.maximum(tp + fn, 1), np.nan)
    w = img_w.astype(float)
    return {"f1_img": float((f1 * w).sum() / w.sum()),
            "recall_img": float(np.nansum(rec * w) / w[has_gt].sum()) if w[has_gt].sum() > 0 else float("nan")}


def load_groups(root: Path) -> dict[str, str]:
    g = {}
    m = pd.read_csv(root / "cassava-v7-stage-6" / "resplit_group_manifest.csv", encoding="utf-8-sig")
    for gid, mem in zip(m.clean_group_id, m.member_filenames):
        for fn in str(mem).split(";"):
            if fn.strip():
                g[fn.strip()] = gid
    return g


def subset_mask(run: dict, names: set[str]) -> np.ndarray:
    return np.array([n in names for n in run["images"]])


def names_from_list(p: Path) -> set[str]:
    return {os.path.basename(l.strip()) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def cluster_weights(run_images: np.ndarray, mask: np.ndarray, groups: dict, rng, shared: dict | None = None) -> np.ndarray:
    """Bobot multiplisitas per citra dari resampel klaster (grup foto) dalam subset mask."""
    idx = np.nonzero(mask)[0]
    gids = np.array([groups.get(n, n) for n in run_images[idx]])
    uniq, inv = np.unique(gids, return_inverse=True)
    if shared is not None:
        key = tuple(uniq)
        if key not in shared:
            shared[key] = rng.integers(0, len(uniq), len(uniq))
        draw = shared[key]
    else:
        draw = rng.integers(0, len(uniq), len(uniq))
    cnt = np.bincount(draw, minlength=len(uniq))
    w = np.zeros(len(run_images), dtype=np.int64)
    w[idx] = cnt[inv]
    return w


def build(root: Path, per_image_dir: Path):
    sim = prepare_run(per_image_dir / "preds_sim_on_simval.csv", per_image_dir / "gt_simval.csv")
    grp = prepare_run(per_image_dir / "preds_group_on_cleanval.csv", per_image_dir / "gt_cleanval.csv")
    E = root / "eleak_sim_rq1"
    sets_sim = {
        "val_all": set(sim["images"]),
        "val_leaked": names_from_list(E / "sim_val_leaked.txt"),
        "val_twinval": names_from_list(E / "sim_val_twinval.txt"),
        "val_single": names_from_list(E / "sim_val_single.txt"),
        "diag_A": names_from_list(E / "diag_A_cleanval_unseen.txt"),
        "diag_B": names_from_list(E / "diag_B_cleantrain_unseen.txt"),
    }
    sets_grp = {
        "val_all": set(grp["images"]),
        "val_twin": names_from_list(E / "clean_val_twin.txt"),
        "diag_A": names_from_list(E / "diag_A_cleanval_unseen.txt"),
    }
    return sim, grp, sets_sim, sets_grp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "boot", "report"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--per-image-dir", default="stage_v2_eleak_sim/per_image")
    ap.add_argument("--cache", default="stage_v2_eleak_sim/per_image/analysis_cache.npz")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    pdir = root / args.per_image_dir
    outdir = pdir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    cache = root / args.cache

    if args.mode == "prep":
        sim, grp, sets_sim, sets_grp = build(root, pdir)
        np.savez_compressed(cache, **{f"sim_{k}": v for k, v in sim.items()}, **{f"grp_{k}": v for k, v in grp.items()})
        point = {}
        for tag, run, sets in (("sim", sim, sets_sim), ("group", grp, sets_grp)):
            for name, s in sets.items():
                w = subset_mask(run, s).astype(np.int64)
                m50, m5095 = map_weighted(run, w)
                point[f"{tag}/{name}"] = {"n_images": int(w.sum()), "map50": m50, "map50_95": m5095, **op_means(run, w)}
        (outdir / "point_estimates.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
        print(json.dumps(point, indent=2))
        return

    z = np.load(cache, allow_pickle=False)
    sim = {k[4:]: z[k] for k in z.files if k.startswith("sim_")}
    grp = {k[4:]: z[k] for k in z.files if k.startswith("grp_")}
    E = root / "eleak_sim_rq1"
    groups = load_groups(root)

    if args.mode == "boot":
        rng = np.random.default_rng(1000 + args.seed)
        mA_s = subset_mask(sim, names_from_list(E / "diag_A_cleanval_unseen.txt"))
        mA_g = subset_mask(grp, names_from_list(E / "diag_A_cleanval_unseen.txt"))
        mL = subset_mask(sim, names_from_list(E / "sim_val_leaked.txt"))
        mT = subset_mask(sim, names_from_list(E / "sim_val_twinval.txt"))
        mS = subset_mask(sim, names_from_list(E / "sim_val_single.txt"))
        mAll_s = np.ones(len(sim["images"]), bool); mAll_g = np.ones(len(grp["images"]), bool)
        rows = []
        for _ in range(args.iters):
            shared: dict = {}
            wA_s = cluster_weights(sim["images"], mA_s, groups, rng, shared)
            wA_g = cluster_weights(grp["images"], mA_g, groups, rng, shared)  # resampel berpasangan
            r = {}
            for key, run, w in (("sim_A", sim, wA_s), ("grp_A", grp, wA_g),
                                ("sim_L", sim, cluster_weights(sim["images"], mL, groups, rng)),
                                ("sim_T", sim, cluster_weights(sim["images"], mT, groups, rng)),
                                ("sim_S", sim, cluster_weights(sim["images"], mS, groups, rng)),
                                ("sim_all", sim, cluster_weights(sim["images"], mAll_s, groups, rng)),
                                ("grp_all", grp, cluster_weights(grp["images"], mAll_g, groups, rng))):
                m50, m5095 = map_weighted(run, w)
                r[f"{key}_m50"], r[f"{key}_m5095"] = m50, m5095
                r[f"{key}_f1"] = op_means(run, w)["f1_img"]
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(outdir / f"boot_seed{args.seed}.csv", index=False)
        print(f"boot seed {args.seed}: {len(df)} iterasi")
        return

    # report
    files = sorted(glob.glob(str(outdir / "boot_seed*.csv")))
    B = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    point = json.loads((outdir / "point_estimates.json").read_text(encoding="utf-8"))
    res = {"n_boot": int(len(B))}
    for metric in ("m5095", "m50", "f1"):
        d = {
            "d_same (sim A - group A)": B[f"sim_A_{metric}"] - B[f"grp_A_{metric}"],
            "d_leak (sim bocor - sim kembar)": B[f"sim_L_{metric}"] - B[f"sim_T_{metric}"],
            "d_report (sim val_all - group val_all)": B[f"sim_all_{metric}"] - B[f"grp_all_{metric}"],
            "d_infl (d_report - d_same)": (B[f"sim_all_{metric}"] - B[f"grp_all_{metric}"]) - (B[f"sim_A_{metric}"] - B[f"grp_A_{metric}"]),
        }
        res[metric] = {k: {"mean": float(v.mean() * 100), "ci95": [float(v.quantile(0.025) * 100), float(v.quantile(0.975) * 100)]} for k, v in d.items()}
    label = {"m5095": "mAP@50–95", "m50": "mAP@50", "f1": "F1 per citra (conf ≥ 0,25)"}
    lines = ["# E-LEAK per citra — bootstrap klaster per grup foto", f"Tanggal: {datetime.now():%Y-%m-%d %H:%M}",
             f"Iterasi bootstrap: {res['n_boot']}", "", "## Estimasi titik (dihitung ulang dari prediksi per citra)", "",
             "| Model/subset | Citra | mAP@50 | mAP@50–95 | F1 per citra | Recall per citra |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in point.items():
        lines.append(f"| {k} | {v['n_images']:,} | {v['map50']*100:.2f}% | {v['map50_95']*100:.2f}% | {v['f1_img']*100:.2f}% | {v['recall_img']*100:.2f}% |")
    for metric in ("m5095", "m50", "f1"):
        lines += ["", f"## Selisih {label[metric]} (poin, rata-rata bootstrap dan IK 95%)", "", "| Perbandingan | Rata-rata | IK 95% |", "| --- | --- | --- |"]
        for k, v in res[metric].items():
            lines.append(f"| {k} | {v['mean']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    lines += ["", "Catatan: resampel per grup foto (pasangan kembaran ikut bersama); d_same berpasangan pada citra yang sama."]
    (outdir / "ELEAK_PER_IMAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "eleak_per_image_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

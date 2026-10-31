#!/usr/bin/env python3
"""
AP dan recall per ukuran objek untuk 15 model ablasi (Pakem v1.4, Fase 5 A1, RQ3)
=================================================================================
Replikasi numpy dari COCOeval bbox (pycocotools): area kecil < 32^2, sedang 32^2-96^2, besar > 96^2 piksel (citra
640 px), maxDets 100 per citra per kelas, 10 ambang IoU, 101 titik recall, pencocokan rakus yang sama (GT yang
diabaikan di urutan akhir, deteksi tak cocok di luar rentang area diabaikan). Dicocokkan dengan pycocotools lewat
cek_coco_ukuran.py (B2).

Ditambah recall operasional per ukuran: proporsi objek GT yang ditemukan prediksi sekelas dengan conf >= 0,25 dan
IoU >= 0,5 (pencocokan rakus menurut keyakinan), objek dikelompokkan menurut luas kotak GT-nya.

Bootstrap klaster per grup foto (resampel citra sama untuk semua model), 1.000 iterasi (4 x 250). Kontras dan aturan
keputusan sama dengan analyze_ablation.py.

Pemakaian: python analyze_ukuran.py prep ; python analyze_ukuran.py boot --seed 0 --iters 250 ; python analyze_ukuran.py report
Keluaran: stage_v2_ablation/analysis/ukuran/ (UKURAN_REPORT.md, ukuran_results.json, boot_seed*.csv, cache/)
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_ablation import CONTRASTS, EXPS, SEEDS  # noqa: E402
from analyze_eleak_per_image import CLASS_NAMES, iou_matrix, load_groups  # noqa: E402

AREAS = {"all": (0.0, 1e10), "small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2), "large": (96.0 ** 2, 1e10)}
BOOT_AREAS = ["small", "medium", "large"]
IOUT = np.linspace(0.5, 0.95, 10)
RECT = np.linspace(0.0, 1.0, 101)
MAXDET = 100
TAU = 0.25
NC = len(CLASS_NAMES)


def eval_img_cat(gb, ga, db, ds, da, rng):
    """Satu (citra, kelas, rentang area) persis seperti COCOeval.evaluateImg. Mengembalikan (tp[nd,10], keep[nd], npig)."""
    lo, hi = rng
    g_ig = (ga < lo) | (ga > hi)
    gord = np.argsort(g_ig, kind="mergesort")
    g_ig = g_ig[gord]
    nd, ng = len(ds), len(gord)
    dt_m = np.zeros((10, nd), dtype=bool)
    dt_ig = np.zeros((10, nd), dtype=bool)
    if nd and ng:
        ious = iou_matrix(db, gb[gord])   # nd x ng (db sudah urut skor)
        cand = np.nonzero(ious.max(axis=1) >= 0.5)[0]
        for t, thr in enumerate(IOUT):
            gt_used = np.zeros(ng, dtype=bool)
            for d in cand:
                iou = min(thr, 1 - 1e-10)
                m = -1
                for g in range(ng):
                    if gt_used[g]:
                        continue
                    if m > -1 and not g_ig[m] and g_ig[g]:
                        break
                    if ious[d, g] < iou:
                        continue
                    iou = ious[d, g]
                    m = g
                if m == -1:
                    continue
                dt_ig[t, d] = g_ig[m]
                dt_m[t, d] = True
                gt_used[m] = True
    out_rng = (da < lo) | (da > hi)
    dt_ig = dt_ig | (~dt_m & out_rng[None, :])
    keep = ~dt_ig  # per ambang
    return dt_m, keep, int((~g_ig).sum())


def recall_operasional(gb, ga, db, ds):
    """Objek GT (urut asli) yang cocok dengan deteksi conf >= TAU, IoU >= 0,5, rakus menurut skor."""
    hit = np.zeros(len(gb), dtype=bool)
    k = ds >= TAU
    if k.any() and len(gb):
        ious = iou_matrix(db[k], gb)
        for d in range(ious.shape[0]):
            j = np.where(~hit, ious[d], -1).argmax()
            if ious[d, j] >= 0.5 and not hit[j]:
                hit[j] = True
    return hit


def prep_model(preds: Path, G: pd.DataFrame, images: list[str]) -> dict:
    P = pd.read_csv(preds)
    pg = P.groupby(["image_id", "class_id"]).indices
    gg = G.groupby(["image_id", "class_id"]).indices
    Pb = P[["x1", "y1", "x2", "y2"]].to_numpy(float); Ps = P.confidence.to_numpy(float)
    Gb = G[["x1", "y1", "x2", "y2"]].to_numpy(float)
    Pa = (Pb[:, 2] - Pb[:, 0]) * (Pb[:, 3] - Pb[:, 1]); Ga = (Gb[:, 2] - Gb[:, 0]) * (Gb[:, 3] - Gb[:, 1])
    out = {}
    rec_rows = []  # (img, cls, area, hit)
    for a, rng in AREAS.items():
        for c in range(NC):
            sc, im, tp, kp = [], [], [], []
            npig = np.zeros(len(images), dtype=np.int64)
            for i, name in enumerate(images):
                gi = gg.get((name, c), np.array([], dtype=int))
                di = pg.get((name, c), np.array([], dtype=int))
                if len(di):
                    order = np.argsort(-Ps[di], kind="mergesort")[:MAXDET]
                    di = di[order]
                dm, keep, n = eval_img_cat(Gb[gi], Ga[gi], Pb[di], Ps[di], Pa[di], rng)
                npig[i] = n
                if len(di):
                    sc.append(Ps[di]); im.append(np.full(len(di), i)); tp.append(dm.T); kp.append(keep.T)
                if a == "all" and len(gi):
                    hit = recall_operasional(Gb[gi], Ga[gi], Pb[di], Ps[di])
                    for g_area, h in zip(Ga[gi], hit):
                        rec_rows.append((i, c, g_area, h))
            if sc:
                sc = np.concatenate(sc); im = np.concatenate(im); tp = np.concatenate(tp); kp = np.concatenate(kp)
                o = np.argsort(-sc, kind="mergesort")
                sc, im, tp, kp = sc[o], im[o], tp[o], kp[o]
            else:
                sc = np.zeros(0); im = np.zeros(0, dtype=int); tp = np.zeros((0, 10), bool); kp = np.zeros((0, 10), bool)
            out[f"{a}|{c}|img"] = im
            out[f"{a}|{c}|tp"] = tp
            out[f"{a}|{c}|keep"] = kp
            out[f"{a}|{c}|npig"] = npig
    R = np.array(rec_rows, dtype=float) if rec_rows else np.zeros((0, 4))
    out["rec_img"] = R[:, 0].astype(int); out["rec_cls"] = R[:, 1].astype(int)
    out["rec_area"] = R[:, 2]; out["rec_hit"] = R[:, 3].astype(bool)
    return out


def coco_metrics(z: dict, w: np.ndarray, areas=("all", "small", "medium", "large")) -> dict:
    res = {}
    for a in areas:
        aps, ars, per = [], [], {}
        for c in range(NC):
            npig = float((z[f"{a}|{c}|npig"] * w).sum())
            if npig <= 0:
                per[CLASS_NAMES[c]] = (None, None)
                continue
            im = z[f"{a}|{c}|img"]; tp = z[f"{a}|{c}|tp"]; kp = z[f"{a}|{c}|keep"]
            wd = w[im].astype(float)
            ap_t, ar_t = [], []
            for t in range(10):
                k = kp[:, t] & (wd > 0)
                if not k.any():
                    ap_t.append(0.0); ar_t.append(0.0); continue
                tpw = np.cumsum(tp[k, t] * wd[k]); fpw = np.cumsum((~tp[k, t]) * wd[k])
                rc = tpw / npig
                pr = tpw / (tpw + fpw + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                inds = np.searchsorted(rc, RECT, side="left")
                q = np.where(inds < len(pr), pr[np.minimum(inds, len(pr) - 1)], 0.0)
                ap_t.append(float(q.mean())); ar_t.append(float(rc[-1]))
            per[CLASS_NAMES[c]] = (float(np.mean(ap_t)), float(np.mean(ar_t)))
            aps.append(np.mean(ap_t)); ars.append(np.mean(ar_t))
        res[f"AP_{a}"] = float(np.mean(aps)) * 100 if aps else np.nan
        res[f"AR_{a}"] = float(np.mean(ars)) * 100 if ars else np.nan
        for nm, (apk, ark) in per.items():
            res[f"AP_{a}_{nm}"] = apk * 100 if apk is not None else np.nan
    # recall operasional per ukuran (batas inklusif seperti COCO)
    for a in ("small", "medium", "large"):
        lo, hi = AREAS[a]
        m = (z["rec_area"] >= lo) & (z["rec_area"] <= hi)
        ww = w[z["rec_img"][m]].astype(float)
        res[f"rec025_{a}"] = float((z["rec_hit"][m] * ww).sum() / ww.sum() * 100) if ww.sum() > 0 else np.nan
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "boot", "report", "cek"])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="stage_v2_ablation/analysis/ukuran")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--coco-dir", default="stage_v2_ablation/analysis/coco_cek")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = root / args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    pdir = root / "stage_v2_ablation/per_image"
    G = pd.read_csv(pdir / "gt_cleanval.csv")
    keys = [(s, e, lab) for s in SEEDS for e, lab in EXPS]

    def images_list():
        imgs = set(G.image_id)
        for s, e, _ in keys:
            imgs |= set(pd.read_csv(pdir / f"preds_seed{s}_{e}.csv", usecols=["image_id"]).image_id)
        return sorted(imgs)

    if args.mode == "prep":
        f_img = cache / "images.json"
        images = json.loads(f_img.read_text()) if f_img.exists() else images_list()
        f_img.write_text(json.dumps(images))
        for s, e, lab in keys:
            dst = cache / f"seed{s}_{e}.npz"
            if dst.exists():
                continue
            np.savez_compressed(dst, **prep_model(pdir / f"preds_seed{s}_{e}.csv", G, images))
            print(f"cache {dst.name}", flush=True)
        Ga = (G.x2 - G.x1) * (G.y2 - G.y1)
        cnt = {a: {CLASS_NAMES[c]: int(((G.class_id == c) & (Ga >= lo) & (Ga <= hi)).sum()) for c in range(NC)}
               for a, (lo, hi) in AREAS.items()}
        (out / "jumlah_objek.json").write_text(json.dumps(cnt, indent=2))
        print(json.dumps(cnt, indent=2))
        return

    images = json.loads((cache / "images.json").read_text())
    Z = {(s, lab): dict(np.load(cache / f"seed{s}_{e}.npz")) for s, e, lab in keys}

    if args.mode == "cek":
        ones = np.ones(len(images), dtype=np.int64)
        for f in sorted(glob.glob(str(root / args.coco_dir / "coco_*.json"))):
            J = json.loads(Path(f).read_text(encoding="utf-8"))
            name = Path(f).stem.replace("coco_seed", "")
            s, e = name.split("_", 1)
            lab = dict(EXPS)[e]
            m = coco_metrics(Z[(int(s), lab)], ones)
            for k_np, k_co in [("AP_all", "AP"), ("AP_small", "AP_small"), ("AP_medium", "AP_medium"), ("AP_large", "AP_large"),
                               ("AR_small", "AR_small"), ("AR_medium", "AR_medium"), ("AR_large", "AR_large"), ("AR_all", "AR100")]:
                print(f"{name:32s} {k_np:10s} numpy {m[k_np]:8.4f}  pycocotools {J['stats'][k_co]*100:8.4f}  selisih {m[k_np]-J['stats'][k_co]*100:+.5f}")
        return

    groups = load_groups(root)
    gids = np.array([groups.get(n, n) for n in images])
    uniq, inv = np.unique(gids, return_inverse=True)

    if args.mode == "boot":
        rng = np.random.default_rng(5100 + args.seed)
        rows = []
        for _ in range(args.iters):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
            r = {}
            for (s, lab), z in Z.items():
                for k, v in coco_metrics(z, w, BOOT_AREAS).items():
                    if k.startswith(("AP_small", "AP_medium", "AP_large", "AR_small", "rec025")) and k.count("_") <= 1:
                        r[f"{s}|{lab}|{k}"] = v
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / f"boot_seed{args.seed}.csv", index=False)
        print(f"boot seed {args.seed}: {len(rows)} iterasi")
        return

    # ------------------------------------------------------------------ report
    ones = np.ones(len(images), dtype=np.int64)
    point = {f"{s}|{lab}": coco_metrics(z, ones) for (s, lab), z in Z.items()}
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "boot_seed*.csv")))], ignore_index=True)
    cnt = json.loads((out / "jumlah_objek.json").read_text())
    labs = [lab for _, lab in EXPS]
    mets = [("AP_small", "AP objek kecil"), ("AP_medium", "AP objek sedang"), ("AP_large", "AP objek besar"),
            ("AR_small", "AR objek kecil (maxDets 100)"), ("rec025_small", "Recall objek kecil, conf ≥ 0,25")]
    L = ["# AP dan recall per ukuran objek (Fase 5 A1)",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · {len(images):,} citra validasi bersih · 15 model · bootstrap klaster {len(B)} iterasi",
         "", "Protokol COCO: kecil < 32², sedang 32²–96², besar > 96² piksel (citra 640 px); maxDets 100; replikasi numpy dari pycocotools.",
         "", "## Jumlah objek GT per ukuran", "", "| Ukuran | " + " | ".join(CLASS_NAMES) + " | Total |", "| --- |" + " --- |" * (NC + 1)]
    for a in ("small", "medium", "large"):
        L.append(f"| {a} | " + " | ".join(str(cnt[a][c]) for c in CLASS_NAMES) + f" | {sum(cnt[a].values())} |")
    L += ["", "## Nilai per model (rata-rata tiga seed)", "",
          "| Model | AP semua (COCO) | AP kecil | AP sedang | AP besar | AR kecil | Recall kecil conf ≥ 0,25 | Recall sedang | Recall besar |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lab in labs:
        v = lambda k: np.mean([point[f"{s}|{lab}"][k] for s in SEEDS])
        L.append(f"| {lab} | {v('AP_all'):.2f} | {v('AP_small'):.2f} | {v('AP_medium'):.2f} | {v('AP_large'):.2f} | {v('AR_small'):.2f} | "
                 f"{v('rec025_small'):.2f} | {v('rec025_medium'):.2f} | {v('rec025_large'):.2f} |")
    res = {"jumlah_objek": cnt, "point": point, "contrasts": {}}
    for k, arti in mets:
        L += ["", f"## Kontras: {arti} (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)", "",
              "| Kontras | Per seed | Rata-rata | IK 95% | Verdict |", "| --- | --- | --- | --- | --- |"]
        for name, coef, _ in CONTRASTS:
            per = [sum(c * point[f"{s}|{lab}"][k] for lab, c in coef.items()) for s in SEEDS]
            boot = np.mean([sum(c * B[f"{s}|{lab}|{k}"] for lab, c in coef.items()) for s in SEEDS], axis=0)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            if all(d > 0 for d in per) and lo > 0:
                vd = "meningkatkan secara konsisten"
            elif all(d < 0 for d in per) and hi < 0:
                vd = "menurunkan secara konsisten"
            else:
                vd = "tidak terbukti berbeda"
            res["contrasts"][f"{k}|{name}"] = {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)], "verdict": vd}
            L.append(f"| {name} | " + " / ".join(f"{d:+.2f}" for d in per) + f" | {np.mean(per):+.2f} | {lo:+.2f} s.d. {hi:+.2f} | {vd} |")
    L += ["", "## AP objek kecil per kelas (rata-rata tiga seed)", "", "| Model | " + " | ".join(CLASS_NAMES) + " |", "| --- |" + " --- |" * NC]
    for lab in labs:
        L.append(f"| {lab} | " + " | ".join(f"{np.nanmean([point[f'{s}|{lab}'][f'AP_small_{c}'] for s in SEEDS]):.2f}" for c in CLASS_NAMES) + " |")
    L += ["", "Catatan: AP COCO memakai maxDets 100 dan definisi area COCO, sehingga sedikit berbeda dari mAP@50–95 utama "
          "(prediksi hingga 300 per citra). IK berasal dari bootstrap klaster per grup foto, dirata-rata antar seed."]
    (out / "UKURAN_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "ukuran_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

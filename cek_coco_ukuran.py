#!/usr/bin/env python3
"""
AP dan AR per ukuran objek dengan pycocotools (Pakem v1.4, Fase 5 A1, pengecekan B2)
=====================================================================================
Menghitung metrik COCO standar (bbox, area kecil < 32^2, sedang 32^2-96^2, besar > 96^2 piksel pada citra 640 px,
maxDets 100) dari prediksi per citra yang sudah ada. Hasilnya dipakai untuk mencocokkan replikasi numpy
(analyze_ukuran.py) yang menjalankan bootstrap untuk 15 model.

Masukan: CSV prediksi per citra (image_id, class_id, confidence, x1, y1, x2, y2) dan gt_cleanval.csv.
Daftar citra = seluruh isi cassava-v7-runtime-mirror/valid/images (citra tanpa GT tetap dihitung).
Keluaran: <out-dir>/coco_<nama>.json (12 angka summarize, AP/AR per kelas per area, jumlah GT per area).

Pemakaian (.venv-cuda, perlu pycocotools; CPU saja):
  python cek_coco_ukuran.py                      # tiga model seed 42 (EXP-0, EXP-1, EXP-2)
  python cek_coco_ukuran.py --preds <csv> ...    # model lain
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
AREA = ["all", "small", "medium", "large"]
STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
DEFAULT = [ROOT / "stage_v2_ablation/per_image" / f"preds_seed42_{e}.csv"
           for e in ("exp_0_baseline", "exp_1_adaptive_sampling", "exp_2_targeted_aug")]


def build_gt(gt_csv: Path, img_dir: Path) -> tuple[COCO, dict]:
    G = pd.read_csv(gt_csv)
    names = sorted(f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"})
    w0 = int(G.img_w.mode().iat[0]); h0 = int(G.img_h.mode().iat[0])
    size = G.groupby("image_id")[["img_w", "img_h"]].first().to_dict("index")
    ids = {n: i + 1 for i, n in enumerate(names)}
    images = [{"id": ids[n], "file_name": n, "width": int(size.get(n, {}).get("img_w", w0)),
               "height": int(size.get(n, {}).get("img_h", h0))} for n in names]
    anns = []
    for k, r in enumerate(G.itertuples(index=False)):
        w, h = float(r.x2 - r.x1), float(r.y2 - r.y1)
        anns.append({"id": k + 1, "image_id": ids[r.image_id], "category_id": int(r.class_id) + 1,
                     "bbox": [float(r.x1), float(r.y1), w, h], "area": w * h, "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": c + 1, "name": n} for c, n in enumerate(NAMES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt, ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=[str(p) for p in DEFAULT])
    ap.add_argument("--gt", default=str(ROOT / "stage_v2_ablation/per_image/gt_cleanval.csv"))
    ap.add_argument("--img-dir", default=str(ROOT / "cassava-v7-runtime-mirror/valid/images"))
    ap.add_argument("--out-dir", default=str(ROOT / "stage_v2_ablation/analysis/coco_cek"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt, ids = build_gt(Path(args.gt), Path(args.img_dir))
    print(f"GT: {len(gt.getImgIds())} citra, {len(gt.getAnnIds())} objek")

    for p in map(Path, args.preds):
        t0 = time.time()
        P = pd.read_csv(p)
        dets = [{"image_id": ids[r.image_id], "category_id": int(r.class_id) + 1,
                 "bbox": [float(r.x1), float(r.y1), float(r.x2 - r.x1), float(r.y2 - r.y1)],
                 "score": float(r.confidence)} for r in P.itertuples(index=False)]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            E.evaluate(); E.accumulate()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.summarize()
        prec = E.eval["precision"]   # [T, R, K, A, M]
        rec = E.eval["recall"]       # [T, K, A, M]
        per_class = {}
        for k, nm in enumerate(NAMES):
            per_class[nm] = {}
            for a, an in enumerate(AREA):
                pk = prec[:, :, k, a, 2]
                rk = rec[:, k, a, 2]
                per_class[nm][an] = {"AP": float(pk[pk > -1].mean()) if (pk > -1).any() else None,
                                     "AR100": float(rk[rk > -1].mean()) if (rk > -1).any() else None}
        n_gt = {}
        for a, an in enumerate(AREA):
            lo, hi = E.params.areaRng[a]
            n_gt[an] = {nm: int(sum(1 for g in gt.dataset["annotations"]
                                    if g["category_id"] == k + 1 and lo <= g["area"] <= hi))
                        for k, nm in enumerate(NAMES)}
        res = {"preds": str(p), "n_det": len(dets), "stats": dict(zip(STAT_NAMES, map(float, E.stats))),
               "per_class": per_class, "n_gt_per_area": n_gt, "area_rng": E.params.areaRng,
               "max_dets": E.params.maxDets, "detik": round(time.time() - t0, 1), "summarize": buf.getvalue()}
        name = p.stem.replace("preds_", "")
        (out / f"coco_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"{name}: AP {E.stats[0]*100:.2f} | AP_small {E.stats[3]*100:.2f} | AP_medium {E.stats[4]*100:.2f} | "
              f"AP_large {E.stats[5]*100:.2f} | AR_small {E.stats[9]*100:.2f} ({res['detik']} dtk)", flush=True)
    print(f"Selesai. Keluaran: {out}")


if __name__ == "__main__":
    main()

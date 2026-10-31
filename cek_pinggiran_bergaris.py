#!/usr/bin/env python3
"""
Prevalensi pinggiran bergaris (isian rotasi BORDER_REPLICATE) - Pakem v1.4, Fase 5 A5
=====================================================================================
Rotasi dengan isian replikasi tepi mengisi sudut kanvas dengan piksel tepi yang diulang, sehingga muncul pola garis
lurus sejajar di pinggir citra. Detektor bebas geometri (tidak menebak sudut rotasi):
  1. Citra diubah ke abu-abu 320 x 320 dan dibagi menjadi 8 x 8 petak (40 px); hanya 28 petak di cincin pinggir dinilai.
  2. Untuk 60 arah (0-177 derajat, langkah 3), hitung rata-rata |I(p + 8 px * arah) - I(p)| per petak.
  3. rasio petak = minimum antar arah / median antar arah. Isian replikasi memberi rasio sangat kecil pada satu arah
     (piksel konstan sepanjang garis); tekstur daun alami tidak.
  4. Petak bergaris bila rasio < AMBANG_PETAK dan median > TEKSTUR_MIN (bukan area polos atau hitam).
     Citra bergaris bila jumlah petak bergaris >= MIN_PETAK.
Kalibrasi (31 Okt 2026): 40 citra validasi acak tanpa baji hitam, dilabeli visual (13 positif, 2 ragu);
AMBANG_PETAK 0,30, TEKSTUR_MIN 3, MIN_PETAK 2 memberi TP 11, FP 0, FN 2 (sensitivitas ~85%, tanpa positif palsu).
Kolom "theta" di CSV keluaran berisi jumlah petak bergaris (nama kolom dipertahankan dari versi detektor sudut).

Pemakaian (CPU): python cek_pinggiran_bergaris.py --daftar <file.txt berisi path citra> --out <csv>
                  python cek_pinggiran_bergaris.py --montase <csv> --out-img <jpg>   (cek visual)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

S = 320
ARAH = np.deg2rad(np.arange(0, 180, 3))
GESER = 8          # px pada resolusi 320 (= 16 px pada 640)
PETAK = 40         # 8 x 8 petak
AMBANG_PETAK = 0.30   # dikalibrasi 31 Okt pada 40 citra berlabel visual: TP 11, FP 0, FN 2
TEKSTUR_MIN = 3.0
MIN_PETAK = 2
_grid = None


def nilai_citra(path: str) -> dict:
    """Detektor bebas geometri: per petak 40 px, rasio (selisih minimum antar arah) / (median antar arah) dari
    |I(p + 8 px * arah) - I(p)|. Isian replikasi tepi memberi rasio sangat kecil pada satu arah (piksel konstan
    sepanjang garis); tekstur alami tidak. Hanya petak di cincin pinggir (28 petak) yang dihitung."""
    global _grid
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    hitam = (img.max(axis=2) <= 12).mean() * 100
    g = cv2.cvtColor(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if _grid is None:
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        _grid = (xx, yy)
    xx, yy = _grid
    T = []
    for t in ARAH:
        sh = cv2.remap(g, (xx + GESER * np.cos(t)).astype(np.float32), (yy + GESER * np.sin(t)).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        T.append(np.abs(sh - g).reshape(S // PETAK, PETAK, S // PETAK, PETAK).mean(axis=(1, 3)))
    T = np.stack(T)
    mn, md = T.min(0), np.median(T, 0)
    rasio = mn / np.maximum(md, 1e-3)
    n = S // PETAK
    cincin = np.zeros((n, n), bool); cincin[0, :] = cincin[-1, :] = cincin[:, 0] = cincin[:, -1] = True
    garis = (rasio < AMBANG_PETAK) & (md > TEKSTUR_MIN) & cincin
    k = int(garis.sum())
    return {"rasio": round(float(np.min(np.where(cincin & (md > TEKSTUR_MIN), rasio, 9.0))), 4), "theta": k,
            "tekstur_lr": round(float(md[cincin].mean()), 2), "tekstur_tb": 0.0, "pct_hitam": round(hitam, 3),
            "bergaris": int(k >= MIN_PETAK)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daftar")
    ap.add_argument("--out")
    ap.add_argument("--montase")
    ap.add_argument("--out-img")
    args = ap.parse_args()
    if args.montase:
        import pandas as pd
        D = pd.read_csv(args.montase)
        pilih = pd.concat([D[D.bergaris == 1].sample(min(8, int(D.bergaris.sum())), random_state=0),
                           D[(D.bergaris == 0) & (D.pct_hitam < 0.5)].nsmallest(4, "rasio"),
                           D[(D.bergaris == 0) & (D.pct_hitam < 0.5)].sample(4, random_state=1)])
        tiles = []
        for _, r in pilih.iterrows():
            im = cv2.resize(cv2.imread(r.path), (240, 240))
            cv2.putText(im, f"{'GARIS' if r.bergaris else 'tidak'} r={r.rasio:.2f} petak={r.theta}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(args.out_img, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return
    paths = [l.strip() for l in open(args.daftar, encoding="utf-8") if l.strip()]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "image_id", "rasio", "theta", "tekstur_lr", "tekstur_tb", "pct_hitam", "bergaris"])
        for i, p in enumerate(paths):
            v = nilai_citra(p)
            w.writerow([p, Path(p).name, v["rasio"], v["theta"], v["tekstur_lr"], v["tekstur_tb"], v["pct_hitam"], v["bergaris"]])
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gambar contoh untuk Bab 4.1-4.3 (Pakem v1.4.2, Fase 5 A6) - CPU saja, tanpa test set
=====================================================================================
gbr4_1_kembaran.jpg        salinan augmentasi Roboflow dalam satu grup foto (alasan split per grup, RQ1)
gbr4_2_nama_generik.jpg    satu nama file generik dipakai foto berbeda di beberapa kelas (positif palsu audit nama, RQ1)
gbr4_3_bobot_exp1.jpg      citra latih berbobot tertinggi dan terendah pada EXP-1 (RQ2, A3)
gbr4_4_prediksi_exp0_exp1.jpg  prediksi EXP-0 vs EXP-1 seed 42 pada citra validasi yang sama, conf >= 0,25 (RQ2, A2)
gbr4_5_context_crop.jpg    context crop EXP-2: citra asli + jendela crop dan hasil crop 640 px (RQ3, A4)
gbr4_6_objek_kecil.jpg     objek kecil (< 32 px) di validasi yang tidak terdeteksi EXP-0 maupun EXP-2 (RQ3, A1)
Semua citra dari split latih/validasi bersih; citra yang masuk split test tidak dipakai.

Pemakaian: python buat_gambar_bab4.py      Keluaran: _gambar_bab4/
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_gambar_bab4"
NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
DIRS = {"train": ROOT / "cassava-v7-runtime-mirror/train", "valid": ROOT / "cassava-v7-runtime-mirror/valid"}
HIJAU, MERAH, BIRU, KUNING, CYAN, PUTIH = (60, 200, 60), (40, 40, 230), (230, 120, 30), (0, 220, 255), (230, 220, 0), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cari(image_id: str) -> tuple[Path, str] | tuple[None, None]:
    for s, d in DIRS.items():
        p = d / "images" / image_id
        if p.exists():
            return p, s
    return None, None


def label(img_path: Path) -> list[tuple[int, float, float, float, float]]:
    lp = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    h, w = 640, 640
    out = []
    if lp.exists():
        for ln in lp.read_text().splitlines():
            if ln.strip():
                c, xc, yc, bw, bh = (float(x) for x in ln.split()[:5])
                out.append((int(c), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def teks(img, s, xy, skala=0.6, warna=PUTIH, tebal=1):
    x, y = xy
    (tw, th), _ = cv2.getTextSize(s, FONT, skala, tebal)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), FONT, skala, warna, tebal, cv2.LINE_AA)


def kotak(img, b, warna, tebal=2, s=None):
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    cv2.rectangle(img, (x1, y1), (x2, y2), warna, tebal)
    if s:
        teks(img, s, (x1 + 2, max(14, y1 - 4)), 0.45, warna)


def kisi(tiles, kolom, pad=6):
    h, w = tiles[0].shape[:2]
    while len(tiles) % kolom:
        tiles.append(np.full_like(tiles[0], 255))
    baris = []
    for i in range(0, len(tiles), kolom):
        row = []
        for t in tiles[i:i + kolom]:
            row += [t, np.full((h, pad, 3), 255, np.uint8)]
        baris += [np.hstack(row[:-1]), np.full((pad, w * kolom + pad * (kolom - 1), 3), 255, np.uint8)]
    return np.vstack(baris[:-1])


def simpan(nama, img):
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / nama), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("tulis", OUT / nama, img.shape)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main() -> None:
    rnd = random.Random(2026)
    man = pd.read_csv(ROOT / "cassava-v7-stage-6/resplit_group_manifest.csv", encoding="utf-8-sig")
    man["anggota"] = man.member_filenames.str.split(";")
    man["base"] = man.anggota.str[0].str.split(".rf.").str[0]

    # --- 4.1 kembaran augmentasi --------------------------------------------------------------------------
    cand = man[(man.num_images >= 2) & man.new_split.isin(["train", "valid"])]
    pilih = cand.sample(40, random_state=7)
    tiles = []
    n_grup = 0
    for r in pilih.itertuples():
        imgs = []
        for f in r.anggota[:2]:
            p, _ = cari(f.strip())
            if p is None:
                break
            im = cv2.resize(cv2.imread(str(p)), (320, 320))
            teks(im, f"{r.primary_class} | salinan {len(imgs) + 1}", (6, 22), 0.55)
            imgs.append(im)
        if len(imgs) == 2:
            tiles += imgs
            n_grup += 1
        if n_grup == 4:
            break
    simpan("gbr4_1_kembaran.jpg", kisi(tiles, 4))

    # --- 4.2 nama generik ---------------------------------------------------------------------------------
    for base, g in man.groupby("base"):
        if g.primary_class.nunique() >= 4 and g.new_split.isin(["train", "valid"]).all():
            break
    tiles = []
    for r in g.sort_values("primary_class").itertuples():
        p, _ = cari(r.anggota[0].strip())
        im = cv2.resize(cv2.imread(str(p)), (320, 320))
        teks(im, f"{r.primary_class} | asal {r.original_splits}", (6, 22), 0.55)
        teks(im, base, (6, 306), 0.5, KUNING)
        tiles.append(im)
    simpan("gbr4_2_nama_generik.jpg", kisi(tiles, len(tiles)))
    print("nama generik:", base)

    # --- 4.3 bobot EXP-1 -------------------------------------------------------------------------------------
    M = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv")
    M["w"] = M.p_sampling / M.p_sampling.sum() * len(M)
    M = M[M.has_gt == 1].sort_values("w")
    tinggi = M.tail(len(M) // 10).sample(4, random_state=3)   # 10% berbobot tertinggi
    rendah = M.head(len(M) // 10).sample(4, random_state=3)   # 10% berbobot terendah
    tiles = []
    for grp, lab in [(tinggi, "bobot tinggi"), (rendah, "bobot rendah")]:
        for r in grp.itertuples():
            p, _ = cari(r.image_id)
            im = cv2.imread(str(p))
            gt = label(p)
            for c, *b in gt:
                kotak(im, b, HIJAU, 4)
            im = cv2.resize(im, (320, 320))
            teks(im, f"{lab}: w = {r.w:.2f}", (6, 22), 0.55)
            teks(im, f"D = {r.d_image:.2f} | {len(gt)} objek", (6, 306), 0.5)
            tiles.append(im)
    simpan("gbr4_3_bobot_exp1.jpg", kisi(tiles, 4))

    # --- 4.4 prediksi EXP-0 vs EXP-1 -----------------------------------------------------------------------
    G = pd.read_csv(ROOT / "stage_v2_ablation/per_image/gt_cleanval.csv")
    P0 = pd.read_csv(ROOT / "stage_v2_ablation/per_image/preds_seed42_exp_0_baseline.csv")
    P1 = pd.read_csv(ROOT / "stage_v2_ablation/per_image/preds_seed42_exp_1_adaptive_sampling.csv")
    P0, P1 = P0[P0.confidence >= 0.25], P1[P1.confidence >= 0.25]
    gg, g0, g1 = G.groupby("image_id"), P0.groupby("image_id"), P1.groupby("image_id")

    def nilai(pred, gts):
        tp, fp, used = [], [], set()
        for pr in pred.sort_values("confidence", ascending=False).itertuples():
            b = (pr.x1, pr.y1, pr.x2, pr.y2)
            best, bj = 0.5, None
            for j, gt in enumerate(gts.itertuples()):
                if j in used or gt.class_id != pr.class_id:
                    continue
                v = iou(b, (gt.x1, gt.y1, gt.x2, gt.y2))
                if v >= best:
                    best, bj = v, j
            (tp if bj is not None else fp).append((pr.class_id, b, pr.confidence))
            if bj is not None:
                used.add(bj)
        return tp, fp

    kand = []
    for im_id in sorted(set(G.image_id) & set(P0.image_id) & set(P1.image_id)):
        gts = gg.get_group(im_id)
        tp0, fp0 = nilai(g0.get_group(im_id), gts)
        tp1, fp1 = nilai(g1.get_group(im_id), gts)
        if len(fp1) - len(fp0) >= 2 and len(tp1) >= len(tp0) + 1 and len(gts) <= 8:
            kand.append(im_id)
    pilih = random.Random(5).sample(kand, min(3, len(kand)))
    tiles = []
    for im_id in pilih:
        p, _ = cari(im_id)
        for nama, grp in [("EXP-0", g0), ("EXP-1", g1)]:
            im = cv2.imread(str(p))
            gts = gg.get_group(im_id)
            for gt in gts.itertuples():
                kotak(im, (gt.x1, gt.y1, gt.x2, gt.y2), HIJAU, 3)
            tp, fp = nilai(grp.get_group(im_id), gts)
            for c, b, cf in tp:
                kotak(im, b, BIRU, 2, f"{NAMES[c]} {cf:.2f}")
            for c, b, cf in fp:
                kotak(im, b, MERAH, 2, f"{NAMES[c]} {cf:.2f}")
            im = cv2.resize(im, (400, 400))
            teks(im, f"{nama} seed 42: TP {len(tp)}, FP {len(fp)}", (6, 22), 0.6)
            tiles.append(im)
    simpan("gbr4_4_prediksi_exp0_exp1.jpg", kisi(tiles, 2))
    print("kandidat 4.4:", len(kand), pilih)

    # --- 4.5 context crop ------------------------------------------------------------------------------------
    T = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv")
    T = T[T.has_target_small == 1].sample(30, random_state=11)
    tiles, n = [], 0
    for r in T.itertuples():
        p, _ = cari(r.image_id)
        if p is None:
            continue
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        bx1, by1, bx2, by2 = (float(v) for v in r.target_small_box.split(","))
        bx1, bx2, by1, by2 = bx1 * W, bx2 * W, by1 * H, by2 * H
        bw, bh = bx2 - bx1, by2 - by1
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        wc, hc = max(4.0, min(W, 2 * bw)), max(4.0, min(H, 2 * bh))
        x0 = max(0.0, min(W - wc, cx - wc / 2)); y0 = max(0.0, min(H - hc, cy - hc / 2))
        ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x0 + wc)), int(round(y0 + hc))
        crop = cv2.resize(im[iy0:iy1, ix0:ix1], (640, 640), interpolation=cv2.INTER_LINEAR)
        sx, sy = 640 / (ix1 - ix0), 640 / (iy1 - iy0)
        kotak(crop, ((bx1 - ix0) * sx, (by1 - iy0) * sy, (bx2 - ix0) * sx, (by2 - iy0) * sy), KUNING, 3)
        asli = im.copy()
        kotak(asli, (bx1, by1, bx2, by2), KUNING, 2)
        kotak(asli, (ix0, iy0, ix1, iy1), CYAN, 2)
        a = cv2.resize(asli, (360, 360)); c = cv2.resize(crop, (360, 360))
        teks(a, f"asli: lesi {bw:.0f} x {bh:.0f} px", (6, 22), 0.55)
        teks(c, f"crop {ix1 - ix0} x {iy1 - iy0} px -> 640 (zoom {sx:.0f}x/{sy:.0f}x)", (6, 22), 0.5)
        tiles += [a, c]
        n += 1
        if n == 3:
            break
    simpan("gbr4_5_context_crop.jpg", kisi(tiles, 2))

    # --- 4.6 objek kecil yang terlewat ----------------------------------------------------------------------
    P2 = pd.read_csv(ROOT / "stage_v2_ablation/per_image/preds_seed42_exp_2_targeted_aug.csv")
    P2 = P2[P2.confidence >= 0.25]
    G["sisi"] = np.sqrt((G.x2 - G.x1) * (G.y2 - G.y1))
    kecil = G[G.sisi < 32]
    lewat = []
    for gt in kecil.itertuples():
        hit = False
        for Pp in (P0, P2):
            sub = Pp[(Pp.image_id == gt.image_id) & (Pp.class_id == gt.class_id)]
            if any(iou((s.x1, s.y1, s.x2, s.y2), (gt.x1, gt.y1, gt.x2, gt.y2)) >= 0.5 for s in sub.itertuples()):
                hit = True
        if not hit:
            lewat.append(gt)
        if len(lewat) >= 60:
            break
    tiles = []
    for gt in random.Random(9).sample(lewat, min(8, len(lewat))):
        p, _ = cari(gt.image_id)
        im = cv2.imread(str(p))
        cx, cy = (gt.x1 + gt.x2) / 2, (gt.y1 + gt.y2) / 2
        x0, y0 = int(max(0, min(640 - 96, cx - 48))), int(max(0, min(640 - 96, cy - 48)))
        patch = cv2.resize(im[y0:y0 + 96, x0:x0 + 96], (288, 288), interpolation=cv2.INTER_NEAREST)
        f = 288 / 96
        kotak(patch, ((gt.x1 - x0) * f, (gt.y1 - y0) * f, (gt.x2 - x0) * f, (gt.y2 - y0) * f), KUNING, 2)
        teks(patch, f"{NAMES[gt.class_id]} {gt.x2 - gt.x1:.0f}x{gt.y2 - gt.y1:.0f} px", (6, 22), 0.55)
        tiles.append(patch)
    simpan("gbr4_6_objek_kecil.jpg", kisi(tiles, 4))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Uji jalan pintas baji hitam (Pakem v1.3.2) - intervensi pada citra validasi bersih, inferensi saja
=================================================================================================
Latar: baji hitam isian rotasi ada pada ~41% citra asal train Roboflow (kelas CGM, CMD, HEALTHY) dan hanya ~2,5%
citra asal valid/test (hampir semua CBB dan CBSD). Model bisa memakai baji sebagai jalan pintas kelas.
Baji berasal dari rotasi terhadap pusat citra dengan kanvas tetap: keempat sudut berisi segitiga hitam.

Desain berpasangan pada 2.567 citra validasi bersih (test set tidak disentuh):
  Set W = citra dengan baji: kriteria cek_struktur_kelas.py (>30% piksel RGB<=3 di salah satu sudut 24x24) ATAU
          pola piksel hitam cocok dengan rotasi (IoU >= 0,5). Yang kedua menangkap rotasi kecil (~1 derajat) yang
          lolos dari kriteria sudut; kolom kriteria_lama_sudut24 di meta mencatat kriteria lama.
  Set N = sisanya (tanpa baji).
    N_asli        citra asli
    N_baji_hitam  diberi baji hitam; sudut rotasi dipinjam dari citra set W (donor acak berseed). Tepi baji dibuat
                  seperti hasil rotasi sungguhan (interpolasi linear), isi citra tidak digeser.
    N_baji_rata   bentuk baji yang sama, diisi warna rata-rata citra itu sendiri (kontrol oklusi)
  Set W = citra dengan baji (sudut rotasi diestimasi dari piksel hitam, lalu masker diperlebar 3 px).
    W_asli        citra asli (baji hitam)
    W_rata        piksel baji diganti warna rata-rata citra (bentuk sama, tidak hitam)
    W_inpaint     piksel baji diisi inpainting Telea (baji hilang)
Kontras utama: hitam - rata, yaitu efek warna hitam pada oklusi yang sama. Kotak GT tidak diubah; porsi kotak GT yang
tertutup baji dicatat di meta_baji.csv untuk analisis sensitivitas.

Analisis (pra-registrasi, dijalankan di workspace cloud setelah run ini):
  Primer   set N, citra berkelas dominan CBB/CBSD: selisih AP50-95 per kelas dan selisih porsi massa keyakinan
           (conf >= 0,25) ke kelas CGM/CMD/HEALTHY, N_baji_hitam - N_baji_rata.
  Sekunder set W (CGM/CMD/HEALTHY): W_asli - W_rata dan W_asli - W_inpaint, AP per kelas dan porsi ke CBB/CBSD.
  Aturan keputusan pakem: "terbukti" bila tanda selisih sama di ketiga seed dan IK 95% (bootstrap klaster per
  clean_group_id, 1.000 iterasi) tidak memuat nol. Selain itu "tidak terbukti".

Model default: EXP-0 seed 42, 123, 2026. Pengaturan prediksi sama dengan predict_eleak_per_image.py
(conf 0,001, IoU NMS 0,7, max_det 300, imgsz 640, NMS seperti val()).

Keluaran (--out-dir, default stage_v2_baji/):
  preds_<model>_<varian>.csv  (image_id, class_id, confidence, x1, y1, x2, y2)
  gt_cleanval.csv, meta_baji.csv, run_info.json, SELESAI.txt (penanda selesai; ada = run dilewati)

Pemakaian (.venv-cuda):
  python uji_baji_hitam.py                     # run penuh, beberapa menit GPU
  python uji_baji_hitam.py --cek-visual 6      # tanpa GPU/ultralytics: montase varian + statistik masker
  python uji_baji_hitam.py --batas 64 --out-dir <folder uji>   # uji cepat sebagian citra
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
VAL_IMAGES = ROOT / "cassava-v7-runtime-mirror" / "valid" / "images"
MODELS_DEFAULT = {
    "exp0_s42": ROOT / "stage_8_ablation_clean_20261020" / "exp_0_baseline" / "weights" / "best.pt",
    "exp0_s123": ROOT / "stage_v2_ablation" / "seed_123" / "exp_0_baseline" / "weights" / "best.pt",
    "exp0_s2026": ROOT / "stage_v2_ablation" / "seed_2026" / "exp_0_baseline" / "weights" / "best.pt",
}
VARIAN_N = ["N_asli", "N_baji_hitam", "N_baji_rata"]
VARIAN_W = ["W_asli", "W_rata", "W_inpaint"]
K_SUDUT, FRAK_SUDUT, AMBANG_MURNI = 24, 0.30, 3  # sama dengan cek_struktur_kelas.py
AMBANG_GELAP = 12          # piksel "hitam" untuk mengestimasi sudut rotasi
LEBAR_TEPI = 3             # pelebaran masker baji W (tepi interpolasi + JPEG)
AMBANG_IOU_W = 0.5         # citra masuk set W bila pola hitamnya cocok dengan rotasi (IoU >= 0,5)
GRID_SUDUT = [round(t, 2) for t in np.arange(-15.0, 15.0001, 0.25) if abs(t) >= 0.25]
SEED_DONOR = 42

_cache: dict = {}


def alpha_rotasi(theta: float, h: int, w: int) -> np.ndarray:
    """Porsi tiap piksel kanvas yang tertutup citra setelah rotasi theta derajat (1 = tertutup, 0 = baji)."""
    key = ("a", round(theta, 2), h, w)
    if key not in _cache:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
        one = np.ones((h, w), np.float32)
        _cache[key] = cv2.warpAffine(one, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _cache[key]


def masker_rotasi(theta: float, h: int, w: int, skala: int = 1) -> np.ndarray:
    key = ("m", round(theta, 2), h, w, skala)
    if key not in _cache:
        a = alpha_rotasi(theta, h, w) < 0.5
        _cache[key] = a[::skala, ::skala].copy() if skala > 1 else a
    return _cache[key]


def punya_baji(img: np.ndarray) -> bool:
    z = img.max(axis=2) <= AMBANG_MURNI
    k = K_SUDUT
    return max(z[:k, :k].mean(), z[:k, -k:].mean(), z[-k:, :k].mean(), z[-k:, -k:].mean()) > FRAK_SUDUT


def ada_baji(img: np.ndarray) -> bool:
    """Kriteria sensitif: kriteria sudut 24 px ATAU pola hitam cocok dengan rotasi (menangkap rotasi kecil ~1 derajat)."""
    return punya_baji(img) or estimasi_sudut(img)[1] >= AMBANG_IOU_W


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def estimasi_sudut(img: np.ndarray) -> tuple[float, float]:
    """Sudut rotasi yang paling cocok dengan piksel hitam di citra (pencarian kasar di resolusi 1/4, lalu penuh)."""
    h, w = img.shape[:2]
    gelap = img.max(axis=2) <= AMBANG_GELAP
    g4 = gelap[::4, ::4]
    kasar = sorted(GRID_SUDUT, key=lambda t: -iou(masker_rotasi(t, h, w, 4), g4))[:4]
    halus = {t: iou(masker_rotasi(t, h, w), gelap) for t in kasar}
    best = max(halus, key=halus.get)
    return best, halus[best]


def masker_w(img: np.ndarray, theta: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = masker_rotasi(theta, h, w).astype(np.uint8)
    m = cv2.dilate(m, np.ones((2 * LEBAR_TEPI + 1, 2 * LEBAR_TEPI + 1), np.uint8))
    # Piksel hitam yang menempel ke baji tetapi belum tercakup (bila estimasi meleset sedikit),
    # dibatasi pada pita 15 px di sekitar baji agar bayangan gelap di isi citra tidak ikut terhapus
    pita = cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0
    gelap = (img.max(axis=2) <= AMBANG_GELAP).astype(np.uint8)
    n, lab = cv2.connectedComponents(gelap, connectivity=8)
    if n > 1:
        ids = np.unique(lab[(m > 0) & (gelap > 0)])
        for i in ids[ids > 0]:
            m[(lab == i) & pita] = 1
    return m > 0


def warna_rata(img: np.ndarray, masker: np.ndarray) -> np.ndarray:
    sisa = img[~masker]
    return (sisa.mean(axis=0) if len(sisa) else img.reshape(-1, 3).mean(axis=0)).astype(np.float32)


def buat_varian(img: np.ndarray, set_: str, theta: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mengembalikan dict varian -> citra BGR uint8, serta masker baji (bool) yang dipakai."""
    h, w = img.shape[:2]
    if set_ == "N":
        a = alpha_rotasi(theta, h, w)[..., None]
        masker = a[..., 0] < 0.5
        mean = warna_rata(img, masker)
        f = img.astype(np.float32)
        hitam = np.clip(f * a + 0.5, 0, 255).astype(np.uint8)
        rata = np.clip(f * a + mean * (1 - a) + 0.5, 0, 255).astype(np.uint8)
        return {"N_asli": img, "N_baji_hitam": hitam, "N_baji_rata": rata}, masker
    masker = masker_w(img, theta)
    mean = warna_rata(img, masker)
    rata = img.copy()
    rata[masker] = np.clip(mean + 0.5, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(img, masker.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return {"W_asli": img, "W_rata": rata, "W_inpaint": inp}, masker


def label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def baca_gt(img_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    lp = label_path(img_path)
    out = []
    if lp.exists():
        for line in lp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c, xc, yc, bw, bh = (float(x) for x in line.split()[:5])
            out.append((int(c), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def tutupan_gt(gt: list, masker: np.ndarray) -> list[float]:
    h, w = masker.shape
    res = []
    for _, x1, y1, x2, y2 in gt:
        a, b = int(max(0, np.floor(x1))), int(min(w, np.ceil(x2)))
        c, d = int(max(0, np.floor(y1))), int(min(h, np.ceil(y2)))
        res.append(float(masker[c:d, a:b].mean()) if (b > a and d > c) else 0.0)
    return res


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def daftar_citra(batas: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    fs = sorted(p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in exts)
    if batas:
        fs = random.Random(0).sample(fs, min(batas, len(fs)))
        fs.sort()
    return fs


def tahap_masker(paths: list[Path]) -> tuple[list[dict], dict]:
    """Lintasan 1 (CPU): set N/W per citra, sudut rotasi W, donor untuk N."""
    info = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Gagal membaca {p}")
        lama = punya_baji(img)
        th, cocok = estimasi_sudut(img)
        if lama or cocok >= AMBANG_IOU_W:
            info.append({"image_id": p.name, "set": "W", "theta": th, "iou_estimasi": round(cocok, 4), "donor": "",
                         "kriteria_lama": int(lama)})
        else:
            info.append({"image_id": p.name, "set": "N", "theta": None, "iou_estimasi": round(cocok, 4), "donor": "",
                         "kriteria_lama": int(lama)})
        if (i + 1) % 200 == 0:
            print(f"  masker {i + 1}/{len(paths)} ({time.time() - t0:.0f} dtk)", flush=True)
    donor_pool = [r for r in info if r["set"] == "W" and r["iou_estimasi"] >= AMBANG_IOU_W]
    if not donor_pool:
        raise RuntimeError("Tidak ada citra W dengan estimasi sudut yang andal (IoU >= 0,5) untuk dijadikan donor.")
    rnd = random.Random(SEED_DONOR)
    for r in info:
        if r["set"] == "N":
            d = rnd.choice(donor_pool)
            r["theta"], r["donor"] = d["theta"], d["image_id"]
    nw = sum(r["set"] == "W" for r in info)
    th = np.array([abs(r["theta"]) for r in info if r["set"] == "W"])
    ringkas = {"n_citra": len(info), "n_W": nw, "n_N": len(info) - nw,
               "theta_abs_W": {"median": float(np.median(th)) if len(th) else None,
                               "p10": float(np.percentile(th, 10)) if len(th) else None,
                               "p90": float(np.percentile(th, 90)) if len(th) else None},
               "n_W_iou_estimasi_lt_0_5": int(sum(1 for r in info if r["set"] == "W" and r["iou_estimasi"] < 0.5)),
               "n_W_hanya_dari_estimasi_sudut": int(sum(1 for r in info if r["set"] == "W" and not r["kriteria_lama"])),
               "n_donor": len(donor_pool), "detik_lintasan_masker": round(time.time() - t0, 1)}
    return info, ringkas


def cek_visual(args) -> None:
    out = Path(args.out_dir) / "_cek_visual"
    out.mkdir(parents=True, exist_ok=True)
    paths = daftar_citra(args.batas or 160)
    info, ringkas = tahap_masker(paths)
    print(json.dumps(ringkas, indent=2))
    rnd = random.Random(1)
    pilih = rnd.sample([r for r in info if r["set"] == "W"], min(args.cek_visual, ringkas["n_W"])) + \
        rnd.sample([r for r in info if r["set"] == "N"], min(args.cek_visual, ringkas["n_N"]))
    baris = []
    for r in pilih:
        img = cv2.imread(str(VAL_IMAGES / r["image_id"]))
        var, masker = buat_varian(img, r["set"], r["theta"])
        cek = {k: int(ada_baji(v)) for k, v in var.items()}
        print(f'{r["set"]} {r["image_id"][:28]:28s} theta={r["theta"]:+6.2f} iou={r["iou_estimasi"]!s:6s} '
              f'masker={100 * masker.mean():5.2f}%  kriteria baji per varian={cek}')
        sel = [cv2.resize(v, (213, 213), interpolation=cv2.INTER_AREA) for v in var.values()]
        for s, nama in zip(sel, var.keys()):
            cv2.putText(s, nama, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        baris.append(np.hstack(sel))
    cv2.imwrite(str(out / "montase_baji.jpg"), np.vstack(baris), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Montase: {out / 'montase_baji.jpg'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Uji jalan pintas baji hitam (inferensi saja, validasi bersih)")
    ap.add_argument("--out-dir", default=str(ROOT / "stage_v2_baji"))
    ap.add_argument("--models", nargs="*", default=None, help="nama=path best.pt; default EXP-0 tiga seed")
    ap.add_argument("--batas", type=int, default=None, help="pakai sebagian citra (uji cepat)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--cek-visual", type=int, default=0, help="buat montase N citra per set lalu berhenti (tanpa GPU)")
    ap.add_argument("--paksa", action="store_true", help="jalankan ulang walau SELESAI.txt sudah ada")
    args = ap.parse_args()

    if args.cek_visual:
        cek_visual(args)
        return

    out = Path(args.out_dir)
    if (out / "SELESAI.txt").exists() and not args.paksa:
        print(f"Sudah selesai sebelumnya ({out / 'SELESAI.txt'}); dilewati. Pakai --paksa untuk mengulang.")
        return
    out.mkdir(parents=True, exist_ok=True)

    models = dict(MODELS_DEFAULT)
    if args.models:
        models = {}
        for m in args.models:
            k, v = m.split("=", 1)
            models[k] = Path(v)
    hilang = [f"{k}: {v}" for k, v in models.items() if not Path(v).exists()]
    if hilang:
        raise FileNotFoundError("Bobot tidak ditemukan: " + "; ".join(hilang))

    t_mulai = time.time()
    paths = daftar_citra(args.batas)
    print(f"Lintasan 1: masker baji untuk {len(paths)} citra validasi bersih", flush=True)
    info, ringkas = tahap_masker(paths)
    print(json.dumps(ringkas, indent=2), flush=True)

    from predict_eleak_per_image import pasang_nms_seperti_val
    from ultralytics import YOLO
    import torch
    import ultralytics

    pasang_nms_seperti_val()
    yolo = {k: YOLO(str(v)) for k, v in models.items()}

    fh, wr = {}, {}
    for m in yolo:
        for v in VARIAN_N + VARIAN_W:
            p = out / f"preds_{m}_{v}.csv.tmp"
            fh[(m, v)] = open(p, "w", newline="", encoding="utf-8")
            wr[(m, v)] = csv.writer(fh[(m, v)])
            wr[(m, v)].writerow(["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    fgt = open(out / "gt_cleanval.csv.tmp", "w", newline="", encoding="utf-8")
    wgt = csv.writer(fgt)
    wgt.writerow(["image_id", "gt_id", "class_id", "x1", "y1", "x2", "y2", "img_w", "img_h"])
    fmeta = open(out / "meta_baji.csv.tmp", "w", newline="", encoding="utf-8")
    wmeta = csv.writer(fmeta)
    wmeta.writerow(["image_id", "set", "kriteria_lama_sudut24", "kelas_dominan", "n_gt", "theta", "iou_estimasi",
                    "donor", "masker_pct",
                    "gt_tutupan_maks", "n_gt_tutupan_ge_0_3", "baji_sesudah_hitam_atau_asli", "baji_sesudah_rata",
                    "baji_sesudah_inpaint"])

    n_pred = {k: 0 for k in wr}
    t1 = time.time()
    print(f"Lintasan 2: varian + prediksi ({len(yolo)} model x 3 varian per set)", flush=True)
    for i in range(0, len(paths), args.batch):
        chunk = list(zip(paths[i: i + args.batch], info[i: i + args.batch]))
        per_varian: dict[str, list[tuple[str, np.ndarray]]] = {v: [] for v in VARIAN_N + VARIAN_W}
        for p, r in chunk:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            var, masker = buat_varian(img, r["set"], r["theta"])
            for v, arr in var.items():
                per_varian[v].append((p.name, arr))
            gt = baca_gt(p, w, h)
            for gi, (c, x1, y1, x2, y2) in enumerate(gt):
                wgt.writerow([p.name, gi, c, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", w, h])
            tut = tutupan_gt(gt, masker)
            kelas = [g[0] for g in gt]
            dom = NAMES[max(set(kelas), key=kelas.count)] if kelas else "(tanpa kotak)"
            nama = list(var.keys())
            cek = [ada_baji(var[k]) for k in nama]
            wmeta.writerow([p.name, r["set"], r["kriteria_lama"], dom, len(gt), f'{r["theta"]:.2f}', r["iou_estimasi"],
                            r["donor"],
                            f"{100 * masker.mean():.3f}", f"{max(tut):.3f}" if tut else "0.000",
                            sum(t >= 0.3 for t in tut), int(cek[1] if r["set"] == "N" else cek[0]),
                            int(cek[2] if r["set"] == "N" else cek[1]), int(cek[2]) if r["set"] == "W" else ""])
        for m, model in yolo.items():
            for v, items in per_varian.items():
                if not items:
                    continue
                res = model.predict([a for _, a in items], imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                    max_det=args.max_det, device=args.device, verbose=False)
                for (name, _), rr in zip(items, res):
                    b = rr.boxes
                    xyxy = b.xyxy.cpu().numpy()
                    cls = b.cls.cpu().numpy().astype(int)
                    conf = b.conf.cpu().numpy()
                    for k in range(len(cls)):
                        wr[(m, v)].writerow([name, int(cls[k]), f"{conf[k]:.6f}", *(f"{x:.2f}" for x in xyxy[k])])
                    n_pred[(m, v)] += len(cls)
        done = min(i + args.batch, len(paths))
        if done % 256 < args.batch or done == len(paths):
            print(f"  {done}/{len(paths)} citra ({time.time() - t1:.0f} dtk)", flush=True)

    for f in list(fh.values()) + [fgt, fmeta]:
        f.close()
    for p in out.glob("*.csv.tmp"):
        os.replace(p, p.with_suffix(""))
    run_info = {
        "tanggal": datetime.now().isoformat(timespec="seconds"),
        "skrip": "uji_baji_hitam.py", "skrip_sha16": sha16(Path(__file__)),
        "model": {k: {"path": str(v), "sha16": sha16(Path(v))} for k, v in models.items()},
        "varian_N": VARIAN_N, "varian_W": VARIAN_W,
        "kriteria_baji": {"sudut_px": K_SUDUT, "frak": FRAK_SUDUT, "rgb_maks": AMBANG_MURNI},
        "ambang_gelap_estimasi": AMBANG_GELAP, "lebar_tepi_px": LEBAR_TEPI, "seed_donor": SEED_DONOR,
        "prediksi": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
                     "nms": "seperti val() (multi_label=True, tanpa batas waktu)"},
        "ringkasan_masker": ringkas, "jumlah_prediksi": {f"{m}|{v}": n for (m, v), n in n_pred.items()},
        "versi": {"python": sys.version.split()[0], "torch": torch.__version__,
                  "ultralytics": getattr(ultralytics, "__version__", ""), "opencv": cv2.__version__},
        "detik_total": round(time.time() - t_mulai, 1), "batas": args.batas,
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    if not args.batas:
        (out / "SELESAI.txt").write_text(f"selesai {run_info['tanggal']}\n", encoding="utf-8")
    print(f"Selesai dalam {run_info['detik_total'] / 60:.1f} menit. Keluaran: {out}")


if __name__ == "__main__":
    main()

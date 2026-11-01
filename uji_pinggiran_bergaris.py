#!/usr/bin/env python3
"""
Uji jalan pintas pinggiran bergaris (Pakem v1.4.1, Fase 5) - intervensi pada citra validasi bersih, inferensi saja
=================================================================================================================
Latar (A5, 31 Okt 2026): pinggiran bergaris = isian rotasi BORDER_REPLICATE (piksel tepi diulang menjadi garis lurus
di sudut kanvas). Di validasi bersih, pinggiran bergaris ada pada 42% citra CBB tanpa baji hitam, dibanding 20-22% pada
CGM, CMD, dan HEALTHY. Bila model memakai garis itu sebagai isyarat kelas, menambahkan garis akan menggeser keyakinan
ke CBB. Desainnya cermin uji baji hitam (uji_baji_hitam.py).

Himpunan citra (set G): citra set N uji baji (tanpa baji hitam; stage_v2_baji/meta_baji.csv) yang oleh detektor petak
dinilai TIDAK bergaris (stage_v2_ablation/analysis/pinggiran/pinggiran_val.csv, bergaris = 0). Test set tidak disentuh.
Geometri isian per citra: transformasi afin rotasi theta ~ U(-20, 20) derajat terhadap pusat plus geser tx, ty ~
U(-0,15, 0,15) x ukuran citra, diacak dengan seed per nama citra. Geometri ini dikalibrasi 31 Okt pada 70 citra set G
agar detektor petak melihat garis buatan seperti garis asli: terdeteksi 86% (sensitivitas pada garis asli 85%), kuartil
jumlah petak bergaris 4/5/6 (garis asli 3/5/7), luas isian median 16%. Rotasi murni dengan sudut donor uji baji
(luas 2-5%) ditolak karena garisnya terlalu tipis dan tidak terdeteksi sama sekali (0 dari 8 pada cek visual).
  G_asli   citra asli
  G_garis  area isian diisi garis replikasi tepi: citra ditransformasi dengan BORDER_REPLICATE, lalu hanya area isian
           (dengan tepi interpolasi seperti transformasi sungguhan) yang ditempel ke citra asli; isi citra tidak digeser.
  G_rata   area isian yang sama, tiap komponen diisi warna rata-rata garis yang digantikannya (kontrol oklusi dan
           warna, hanya tekstur garis yang hilang)
Kotak GT tidak diubah; porsi kotak GT yang tertutup isian dicatat di meta_garis.csv.
Cek manipulasi: detektor petak (cek_pinggiran_bergaris.nilai_array) dijalankan pada G_garis dan G_rata.

Analisis (PRA-REGISTRASI 31 Okt 2026 malam, dijalankan di workspace cloud dengan analyze_garis.py):
  Primer   subset G_tr (kelas dominan CGM/CMD/HEALTHY), G_garis - G_rata:
             (1) porsi massa keyakinan (conf >= 0,25) yang jatuh ke CBB; hipotesis jalan pintas: naik
             (2) rata-rata AP CGM/CMD/HEALTHY; hipotesis: turun
  Pendukung G_tr: porsi ke CBB/CBSD; persentase citra yang prediksi teratasnya CBB.
  Sekunder subset G_cb (kelas dominan CBB/CBSD), G_garis - G_rata: AP CBB, AP CBSD, porsi ke CBB/CBSD; hipotesis: naik.
  Deskriptif: G_garis - G_asli, G_rata - G_asli; sensitivitas pada citra yang G_garis-nya terdeteksi bergaris dan
           G_rata-nya tidak, serta pada citra yang kotak GT-nya tertutup isian < 30%.
  Model: verdict utama pada EXP-0 tiga seed; EXP-1 dan EXP-2 tiga seed dilaporkan dengan cara yang sama.
  Aturan keputusan pakem: "terbukti" bila tanda selisih sama di ketiga seed dan IK 95% (bootstrap klaster per
  clean_group_id, 1.000 iterasi, resampel sama untuk semua model dan varian) tidak memuat nol.

Pengaturan prediksi sama dengan uji_baji_hitam.py (conf 0,001, IoU NMS 0,7, max_det 300, imgsz 640, NMS seperti val()).

Keluaran (--out-dir, default stage_v2_garis/):
  preds_<model>_<varian>.csv, gt_cleanval.csv (set G saja), meta_garis.csv, run_info.json, SELESAI.txt

Pemakaian (.venv-cuda):
  python uji_pinggiran_bergaris.py                    # run penuh, sekitar 10 menit GPU (9 model)
  python uji_pinggiran_bergaris.py --cek-visual 8     # tanpa GPU/ultralytics: montase varian + cek manipulasi
  python uji_pinggiran_bergaris.py --batas 64 --out-dir <folder uji>   # uji cepat sebagian citra
"""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(ROOT))
from cek_pinggiran_bergaris import nilai_array  # noqa: E402
from uji_baji_hitam import NAMES, VAL_IMAGES, ada_baji, baca_gt, sha16, tutupan_gt  # noqa: E402

VARIAN = ["G_asli", "G_garis", "G_rata"]
META_BAJI = ROOT / "stage_v2_baji" / "meta_baji.csv"
PINGGIRAN_VAL = ROOT / "stage_v2_ablation" / "analysis" / "pinggiran" / "pinggiran_val.csv"
ROT_MAKS, GESER_MAKS = 20.0, 0.15   # dikalibrasi 31 Okt (lihat docstring)


def models_default() -> dict[str, Path]:
    m = {"exp0_s42": ROOT / "stage_8_ablation_clean_20261020" / "exp_0_baseline" / "weights" / "best.pt"}
    for s in (123, 2026):
        m[f"exp0_s{s}"] = ROOT / "stage_v2_ablation" / f"seed_{s}" / "exp_0_baseline" / "weights" / "best.pt"
    for e, nama in (("exp1", "exp_1_adaptive_sampling"), ("exp2", "exp_2_targeted_aug")):
        for s in (42, 123, 2026):
            m[f"{e}_s{s}"] = ROOT / "stage_v2_ablation" / f"seed_{s}" / nama / "weights" / "best.pt"
    return m


def geometri(image_id: str) -> tuple[float, float, float]:
    rnd = random.Random(f"garis-{image_id}")
    return rnd.uniform(-ROT_MAKS, ROT_MAKS), rnd.uniform(-GESER_MAKS, GESER_MAKS), rnd.uniform(-GESER_MAKS, GESER_MAKS)


def buat_varian(img: np.ndarray, theta: float, tx: float, ty: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mengembalikan dict varian -> citra BGR uint8 dan masker isian (bool, alpha < 0,5)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    m[0, 2] += tx * w
    m[1, 2] += ty * h
    a2 = cv2.warpAffine(np.ones((h, w), np.float32), m, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = a2[..., None]
    masker = a2 < 0.5
    rep = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    f = img.astype(np.float32)
    garis = np.clip(f * a + rep * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    # warna rata per sudut: rata-rata piksel garis (alpha < 0,5) di komponen tersambung area alpha < 1
    region = (a2 < 0.999).astype(np.uint8)
    n, lab = cv2.connectedComponents(region, connectivity=8)
    isi = np.zeros_like(f)
    global_mean = rep[masker].mean(axis=0) if masker.any() else f.reshape(-1, 3).mean(axis=0)
    for i in range(1, n):
        komp = lab == i
        inti = komp & masker
        isi[komp] = rep[inti].mean(axis=0) if inti.any() else global_mean
    rata = np.clip(f * a + isi * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    return {"G_asli": img, "G_garis": garis, "G_rata": rata}, masker


def daftar_set_g(batas: int | None) -> list[dict]:
    import pandas as pd
    meta = pd.read_csv(META_BAJI)
    ping = pd.read_csv(PINGGIRAN_VAL)[["image_id", "bergaris", "theta"]].rename(columns={"theta": "petak_asli"})
    g = meta[meta.set == "N"].merge(ping, on="image_id", how="inner")
    g = g[g.bergaris == 0].sort_values("image_id")
    rows = []
    for r in g.itertuples():
        th, tx, ty = geometri(r.image_id)
        rows.append({"image_id": r.image_id, "theta": th, "tx": tx, "ty": ty, "petak_asli": int(r.petak_asli)})
    if batas:
        rows = sorted(random.Random(0).sample(rows, min(batas, len(rows))), key=lambda r: r["image_id"])
    return rows


def cek_visual(args) -> None:
    out = Path(args.out_dir) / "_cek_visual"
    out.mkdir(parents=True, exist_ok=True)
    rows = daftar_set_g(None)
    print(f"Set G: {len(rows)} citra")
    pilih = random.Random(1).sample(rows, min(args.cek_visual, len(rows)))
    baris, stat = [], []
    for r in pilih:
        img = cv2.imread(str(VAL_IMAGES / r["image_id"]), cv2.IMREAD_COLOR)
        var, masker = buat_varian(img, r["theta"], r["tx"], r["ty"])
        det = {k: nilai_array(v) for k, v in var.items()}
        stat.append({k: det[k]["bergaris"] for k in var})
        print(f'{r["image_id"][:30]:30s} theta={r["theta"]:+6.2f} masker={100 * masker.mean():5.2f}% '
              f'petak garis asli/garis/rata = {det["G_asli"]["theta"]}/{det["G_garis"]["theta"]}/{det["G_rata"]["theta"]} '
              f'baji hitam terdeteksi = {[int(ada_baji(v)) for v in var.values()]}')
        sel = [cv2.resize(v, (256, 256), interpolation=cv2.INTER_AREA) for v in var.values()]
        for s, nama in zip(sel, var.keys()):
            cv2.putText(s, f'{nama} petak={det[nama]["theta"]}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        baris.append(np.hstack(sel))
    cv2.imwrite(str(out / "montase_garis.jpg"), np.vstack(baris), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("Terdeteksi bergaris (asli/garis/rata):", [sum(s[k] for s in stat) for k in VARIAN], "dari", len(stat))
    print(f"Montase: {out / 'montase_garis.jpg'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Uji jalan pintas pinggiran bergaris (inferensi saja, validasi bersih)")
    ap.add_argument("--out-dir", default=str(ROOT / "stage_v2_garis"))
    ap.add_argument("--models", nargs="*", default=None, help="nama=path best.pt; default EXP-0, EXP-1, EXP-2 tiga seed")
    ap.add_argument("--batas", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--cek-visual", type=int, default=0, help="buat montase N citra lalu berhenti (tanpa GPU)")
    ap.add_argument("--paksa", action="store_true")
    args = ap.parse_args()

    if args.cek_visual:
        cek_visual(args)
        return

    out = Path(args.out_dir)
    if (out / "SELESAI.txt").exists() and not args.paksa:
        print(f"Sudah selesai sebelumnya ({out / 'SELESAI.txt'}); dilewati. Pakai --paksa untuk mengulang.")
        return
    out.mkdir(parents=True, exist_ok=True)

    models = models_default()
    if args.models:
        models = {}
        for m in args.models:
            k, v = m.split("=", 1)
            models[k] = Path(v)
    hilang = [f"{k}: {v}" for k, v in models.items() if not Path(v).exists()]
    if hilang:
        raise FileNotFoundError("Bobot tidak ditemukan: " + "; ".join(hilang))

    t_mulai = time.time()
    rows = daftar_set_g(args.batas)
    print(f"Set G: {len(rows)} citra (set N uji baji, tidak bergaris menurut detektor)", flush=True)

    from predict_eleak_per_image import pasang_nms_seperti_val
    from ultralytics import YOLO
    import torch
    import ultralytics

    pasang_nms_seperti_val()
    yolo = {k: YOLO(str(v)) for k, v in models.items()}

    fh, wr = {}, {}
    for m in yolo:
        for v in VARIAN:
            fh[(m, v)] = open(out / f"preds_{m}_{v}.csv.tmp", "w", newline="", encoding="utf-8")
            wr[(m, v)] = csv.writer(fh[(m, v)])
            wr[(m, v)].writerow(["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    fgt = open(out / "gt_cleanval.csv.tmp", "w", newline="", encoding="utf-8")
    wgt = csv.writer(fgt)
    wgt.writerow(["image_id", "gt_id", "class_id", "x1", "y1", "x2", "y2", "img_w", "img_h"])
    fmeta = open(out / "meta_garis.csv.tmp", "w", newline="", encoding="utf-8")
    wmeta = csv.writer(fmeta)
    wmeta.writerow(["image_id", "kelas_dominan", "n_gt", "theta", "tx", "ty", "masker_pct", "gt_tutupan_maks",
                    "n_gt_tutupan_ge_0_3", "petak_asli", "petak_garis", "petak_rata", "garis_terdeteksi",
                    "rata_terdeteksi", "baji_hitam_garis", "baji_hitam_rata"])

    n_pred = {k: 0 for k in wr}
    t1 = time.time()
    print(f"Varian + prediksi ({len(yolo)} model x {len(VARIAN)} varian)", flush=True)
    for i in range(0, len(rows), args.batch):
        chunk = rows[i: i + args.batch]
        per_varian: dict[str, list[tuple[str, np.ndarray]]] = {v: [] for v in VARIAN}
        for r in chunk:
            p = VAL_IMAGES / r["image_id"]
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Gagal membaca {p}")
            h, w = img.shape[:2]
            var, masker = buat_varian(img, r["theta"], r["tx"], r["ty"])
            for v, arr in var.items():
                per_varian[v].append((r["image_id"], arr))
            gt = baca_gt(p, w, h)
            for gi, (c, x1, y1, x2, y2) in enumerate(gt):
                wgt.writerow([r["image_id"], gi, c, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", w, h])
            tut = tutupan_gt(gt, masker)
            kelas = [g[0] for g in gt]
            dom = NAMES[max(set(kelas), key=kelas.count)] if kelas else "(tanpa kotak)"
            dg, dr = nilai_array(var["G_garis"]), nilai_array(var["G_rata"])
            wmeta.writerow([r["image_id"], dom, len(gt), f'{r["theta"]:.2f}', f'{r["tx"]:.4f}', f'{r["ty"]:.4f}',
                            f"{100 * masker.mean():.3f}",
                            f"{max(tut):.3f}" if tut else "0.000", sum(t >= 0.3 for t in tut), r["petak_asli"],
                            dg["theta"], dr["theta"], dg["bergaris"], dr["bergaris"],
                            int(ada_baji(var["G_garis"])), int(ada_baji(var["G_rata"]))])
        for m, model in yolo.items():
            for v, items in per_varian.items():
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
        done = min(i + args.batch, len(rows))
        if done % 256 < args.batch or done == len(rows):
            print(f"  {done}/{len(rows)} citra ({time.time() - t1:.0f} dtk)", flush=True)

    for f in list(fh.values()) + [fgt, fmeta]:
        f.close()
    for p in out.glob("*.csv.tmp"):
        os.replace(p, p.with_suffix(""))
    run_info = {
        "tanggal": datetime.now().isoformat(timespec="seconds"),
        "skrip": "uji_pinggiran_bergaris.py", "skrip_sha16": sha16(Path(__file__)),
        "model": {k: {"path": str(v), "sha16": sha16(Path(v))} for k, v in models.items()},
        "varian": VARIAN, "n_citra": len(rows), "geometri": {"rot_maks_derajat": ROT_MAKS, "geser_maks_fraksi": GESER_MAKS,
                                                              "seed": "random.Random('garis-<image_id>')"}, "sumber_set": [str(META_BAJI), str(PINGGIRAN_VAL)],
        "prediksi": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
                     "nms": "seperti val() (multi_label=True, tanpa batas waktu)"},
        "jumlah_prediksi": {f"{m}|{v}": n for (m, v), n in n_pred.items()},
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

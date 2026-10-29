#!/usr/bin/env python3
"""
E-LEAK per citra (RQ1, Pakem v1.3) — simpan prediksi mentah + ground truth per citra
====================================================================================
Pengaturan prediksi mengikuti validasi Ultralytics (conf 0.001, NMS IoU 0.7, max_det 300, imgsz 640)
supaya mAP yang dihitung ulang di CPU bisa dicocokkan dengan angka val().

Perbaikan 29 Okt 2026: NMS predict() bawaan berbeda dari val() (multi_label=False) dan memakai batas waktu
2,0 + 0,05 x batch detik. Pada conf 0.001 batas itu terlampaui ("NMS time limit exceeded") sehingga sisa citra
dalam batch tidak mendapat prediksi sama sekali (49-179 dari 2.567 citra). Skrip ini kini memakai NMS seperti
val(): multi_label=True, torchvision.ops.nms, tanpa batas waktu. Opsi --nms-bawaan-predict mengembalikan perilaku lama.

Pemakaian (di .venv-cuda):
  python predict_eleak_per_image.py --weights <best.pt> --source <file .txt | folder images> \
      --out-preds <preds.csv> --out-gt <gt.csv>
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_source(src: str) -> list[str]:
    p = Path(src)
    if p.suffix.lower() == ".txt":
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(str(p / f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in IMG_EXT)


def label_path(img: str) -> Path:
    p = Path(img)
    parts = list(p.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def pasang_nms_seperti_val() -> None:
    """Samakan NMS predict() dengan val(): multi_label=True, tanpa batas waktu, torchvision.ops.nms."""
    import torchvision  # noqa: F401  (Ultralytics memilih torchvision.ops.nms bila modul ini sudah dimuat)
    from ultralytics.utils import nms as ul_nms

    asli = ul_nms.non_max_suppression

    def nms_seperti_val(*a, **k):
        k["multi_label"] = True
        k["max_time_img"] = 1e6
        return asli(*a, **k)

    ul_nms.non_max_suppression = nms_seperti_val
    print("NMS: seperti val() (multi_label=True, torchvision, tanpa batas waktu)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-preds", required=True)
    ap.add_argument("--out-gt", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--nms-bawaan-predict", action="store_true", help="Pakai NMS bawaan predict() (perilaku lama, tidak disarankan)")
    args = ap.parse_args()
    if not args.nms_bawaan_predict:
        pasang_nms_seperti_val()

    paths = read_source(args.source)
    model = YOLO(args.weights)
    Path(args.out_preds).parent.mkdir(parents=True, exist_ok=True)
    n_pred = n_gt = n_kosong = 0
    with open(args.out_preds, "w", newline="", encoding="utf-8") as fp, open(args.out_gt, "w", newline="", encoding="utf-8") as fg:
        wp = csv.writer(fp)
        wg = csv.writer(fg)
        wp.writerow(["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        wg.writerow(["image_id", "gt_id", "class_id", "x1", "y1", "x2", "y2", "img_w", "img_h"])
        for i in range(0, len(paths), args.batch):
            chunk = paths[i : i + args.batch]
            results = model.predict(chunk, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det,
                                    device=args.device, verbose=False)
            for img, r in zip(chunk, results):
                name = Path(img).name
                h, w = r.orig_shape[:2]
                b = r.boxes
                xyxy = b.xyxy.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(int)
                conf = b.conf.cpu().numpy()
                n_kosong += int(len(cls) == 0)
                for k in range(len(cls)):
                    wp.writerow([name, int(cls[k]), f"{conf[k]:.6f}", *(f"{v:.2f}" for v in xyxy[k])])
                    n_pred += 1
                lp = label_path(img)
                if lp.exists():
                    for gi, line in enumerate(l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()):
                        c, xc, yc, bw, bh = (float(x) for x in line.split()[:5])
                        wg.writerow([name, gi, int(c), f"{(xc - bw / 2) * w:.2f}", f"{(yc - bh / 2) * h:.2f}",
                                     f"{(xc + bw / 2) * w:.2f}", f"{(yc + bh / 2) * h:.2f}", w, h])
                        n_gt += 1
            print(f"{min(i + args.batch, len(paths))}/{len(paths)} citra", end="\r")
    print(f"\nSelesai: {len(paths)} citra, {n_pred:,} prediksi, {n_gt:,} objek GT, {n_kosong} citra tanpa prediksi")


if __name__ == "__main__":
    main()

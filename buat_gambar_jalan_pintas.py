"""Gambar 4.5 draf Bab 4: varian uji jalan pintas (baji hitam dan pinggiran bergaris).

Baris atas: satu citra validasi set N berkelas CBB (asli, baji hitam, baji warna rata).
Baris bawah: satu citra validasi set G berkelas CGM/CMD/HEALTHY (asli, pinggiran bergaris, warna rata).
Varian dibuat dengan fungsi yang sama dengan uji GPU (uji_baji_hitam.buat_varian,
uji_pinggiran_bergaris.buat_varian/geometri). Tidak memakai citra test.

Keluaran: _gambar_bab4/gbr4_7_jalan_pintas.jpg
Pakai: python buat_gambar_jalan_pintas.py [--n-id ID] [--g-id ID]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import uji_baji_hitam as ub  # noqa: E402
import uji_pinggiran_bergaris as ug  # noqa: E402

OUT = ROOT / "_gambar_bab4" / "gbr4_7_jalan_pintas.jpg"
T = 420


def font(size, bold=False):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold
              else ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((T, T))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-id", default=None)
    ap.add_argument("--g-id", default=None)
    a = ap.parse_args()
    meta = pd.read_csv(ROOT / "stage_v2_baji" / "meta_baji.csv")
    pg = pd.read_csv(ROOT / "stage_v2_ablation" / "analysis" / "pinggiran" / "pinggiran_val.csv")
    n_id = a.n_id or meta[(meta.set == "N") & (meta.kelas_dominan == "CBB")].sort_values("image_id").image_id.iloc[3]
    theta_n = float(meta.set_index("image_id").at[n_id, "theta"])
    cand = meta[(meta.set == "N") & meta.kelas_dominan.isin(["CGM", "CMD", "HEALTHY"])].merge(pg[["image_id", "bergaris"]], on="image_id")
    g_id = a.g_id or cand[cand.bergaris == 0].sort_values("image_id").image_id.iloc[5]

    img_n = cv2.imread(str(ub.VAL_IMAGES / n_id))
    vn, _ = ub.buat_varian(img_n, "N", theta_n)
    img_g = cv2.imread(str(ub.VAL_IMAGES / g_id))
    th, tx, ty = ug.geometri(g_id)
    vg, mg = ug.buat_varian(img_g, th, tx, ty)

    rows = [("Baji hitam (citra CBB)", [("Asli", vn["N_asli"]), ("Baji hitam", vn["N_baji_hitam"]),
                                        ("Baji warna rata (kontrol)", vn["N_baji_rata"])]),
            ("Pinggiran bergaris (citra " + str(meta.set_index("image_id").at[g_id, "kelas_dominan"]) + ")",
             [("Asli", vg["G_asli"]), ("Pinggiran bergaris", vg["G_garis"]), ("Warna rata (kontrol)", vg["G_rata"])])]
    pad, head, cap = 14, 40, 30
    W = 3 * T + 4 * pad
    H = 2 * (head + T + cap) + pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    y = pad // 2
    for judul, items in rows:
        d.text((pad, y + 8), judul, font=font(22, True), fill=(20, 20, 20))
        y += head
        for j, (nama, im) in enumerate(items):
            x = pad + j * (T + pad)
            canvas.paste(pil(im), (x, y))
            d.text((x, y + T + 4), nama, font=font(19), fill=(40, 40, 40))
        y += T + cap
    OUT.parent.mkdir(exist_ok=True)
    canvas.save(OUT, quality=90)
    print(f"{OUT}  N={n_id} (theta {theta_n:.2f})  G={g_id} (isian {100 * mg.mean():.1f}%)")


if __name__ == "__main__":
    main()

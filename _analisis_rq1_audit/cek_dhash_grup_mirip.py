#!/usr/bin/env python3
"""Cek dHash 256-bit antar anggota 8 grup nama file yang ditandai 'mirip' pada audit awal (jalankan dari folder SKRIPSI)."""
import os, re
import numpy as np
from PIL import Image

RF = re.compile(r"\.rf\.[0-9a-fA-F]{32}$")
TARGETS = ["image_130_jpg", "image_147_jpg", "image_170_jpg", "image_213_jpg",
           "image_221_jpg", "image_315_jpg", "image_458_jpg", "image_58_jpg"]


def dhash(p, n=16):
    a = np.asarray(Image.open(p).convert("L").resize((n + 1, n), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


files = {}
for sp in ("train", "valid", "test"):
    for f in os.listdir(f"cassava-v7/{sp}/images"):
        b = RF.sub("", os.path.splitext(f)[0])
        if b in TARGETS:
            files.setdefault(b, []).append((sp, f))
for b in TARGETS:
    hs = [(sp, dhash(f"cassava-v7/{sp}/images/{f}")) for sp, f in files[b]]
    cross, within = [], []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            d = int((hs[i][1] != hs[j][1]).sum())
            (cross if hs[i][0] != hs[j][0] else within).append(d)
    print(f"{b}: min dHash lintas split {min(cross)}/256, min dalam split {min(within) if within else '-'}/256")

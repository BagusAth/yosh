#!/usr/bin/env python3
"""
Cek struktur kelas pool latih+validasi bersih (Pakem v1.3.1) — test terkunci tidak disentuh
==========================================================================================
1. Kelas dominan per citra x ukuran grup foto (1 citra vs 2 salinan augmentasi).
2. Kelas dominan x split asal Roboflow v7 (train / valid / test).
3. Baji hitam di sudut (isian rotasi augmentasi) per asal split, sampel acak berseed.
Keluaran: _analisis_struktur_kelas/STRUKTUR_KELAS_REPORT.md + struktur_kelas.json

Pemakaian: python cek_struktur_kelas.py [--sample 400]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
ROOT = Path(__file__).resolve().parent


def load_groups() -> dict[str, str]:
    m = pd.read_csv(ROOT / "cassava-v7-stage-6" / "resplit_group_manifest.csv", encoding="utf-8-sig")
    g = {}
    for gid, mem in zip(m.clean_group_id, m.member_filenames):
        for fn in str(mem).split(";"):
            if fn.strip():
                g[fn.strip()] = str(gid)
    return g


def has_black_wedge(path: str, k: int = 24, frac: float = 0.30) -> tuple[bool, float]:
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    z = im.max(axis=2) <= 3
    corners = [z[:k, :k], z[:k, -k:], z[-k:, :k], z[-k:, -k:]]
    return max(c.mean() for c in corners) > frac, float(z.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()
    out = ROOT / "_analisis_struktur_kelas"
    out.mkdir(exist_ok=True)
    grp = load_groups()
    size = Counter(grp.values())
    orig = {}
    for sp in ("train", "valid", "test"):
        for f in os.listdir(ROOT / "cassava-v7" / sp / "images"):
            orig[f] = sp
    rows = []
    for sp in ("train", "valid"):
        ldir = ROOT / "cassava-v7-runtime-mirror" / sp / "labels"
        for f in os.listdir(ROOT / "cassava-v7-runtime-mirror" / sp / "images"):
            rows.append({"image_id": f, "split_bersih": sp, "img": str(ROOT / "cassava-v7-runtime-mirror" / sp / "images" / f)})
    img = pd.DataFrame(rows)
    # Kelas dari CSV yang sudah ada (lebih cepat daripada membaca 11 ribu label lewat jembatan)
    tr = pd.read_csv(ROOT / "stage_v2_difficulty_20261028" / "train_object_difficulty.csv", usecols=["image_id", "class_name"])
    va = pd.read_csv(ROOT / "stage_v2_eleak_sim" / "per_image" / "gt_cleanval.csv", usecols=["image_id", "class_id"])
    va["class_name"] = va.class_id.map(lambda c: NAMES[int(c)])
    obj = pd.concat([tr, va[["image_id", "class_name"]]])
    dom = obj.groupby("image_id").class_name.agg(lambda s: s.mode().iat[0])
    img["kelas"] = img.image_id.map(dom).fillna("(tanpa kotak)")
    img["ukuran_grup"] = img.image_id.map(lambda n: size.get(grp.get(n), 0))
    img["asal_roboflow"] = img.image_id.map(lambda n: orig.get(n, "?"))
    t1 = pd.crosstab(img.kelas, img.ukuran_grup, margins=True, margins_name="Total")
    t2 = pd.crosstab(img.kelas, img.asal_roboflow, margins=True, margins_name="Total")
    obj["ukuran_grup"] = obj.image_id.map(lambda n: size.get(grp.get(n), 0))
    t3 = pd.crosstab(obj.class_name, obj.ukuran_grup, margins=True, margins_name="Total")

    rnd = random.Random(42)
    wedge = {}
    for lab, mask in (("asal train", img.asal_roboflow == "train"), ("asal valid/test", img.asal_roboflow.isin(["valid", "test"]))):
        paths = img.loc[mask, "img"].tolist()
        s = rnd.sample(paths, min(args.sample, len(paths)))
        res = [has_black_wedge(p) for p in s]
        wedge[lab] = {"n": len(s), "baji_hitam_pct": round(100 * float(np.mean([r[0] for r in res])), 1),
                      "piksel_hitam_murni_pct": round(100 * float(np.mean([r[1] for r in res])), 2)}

    def md(t: pd.DataFrame) -> str:
        cols = [str(c) for c in t.columns]
        lines = ["| Kelas | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
        for idx, r in t.iterrows():
            lines.append(f"| {idx} | " + " | ".join(f"{int(v):,}" for v in r.values) + " |")
        return "\n".join(lines)

    rep = [
        "# Struktur kelas pool latih + validasi bersih", f"Tanggal: {datetime.now():%Y-%m-%d %H:%M}",
        f"Citra: {len(img):,} (test terkunci tidak disentuh). Kelas = kelas dominan per citra.", "",
        "## Kelas x ukuran grup foto (1 = tanpa salinan, 2 = dua salinan augmentasi)", md(t1), "",
        "## Kelas x split asal Roboflow v7", md(t2), "",
        "## Objek per kelas x ukuran grup", md(t3), "",
        f"## Baji hitam di sudut (isian rotasi), sampel acak {args.sample} citra per kelompok, seed 42", "",
        "| Kelompok | n | Citra dengan baji hitam | Rata-rata piksel hitam murni |", "| --- | --- | --- | --- |",
    ]
    for k, v in wedge.items():
        rep.append(f"| {k} | {v['n']} | {v['baji_hitam_pct']}% | {v['piksel_hitam_murni_pct']}% |")
    rep += ["", "Baji hitam = lebih dari 30% piksel di salah satu sudut 24x24 px bernilai RGB <= 3."]
    (out / "STRUKTUR_KELAS_REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    (out / "struktur_kelas.json").write_text(json.dumps({"kelas_x_ukuran_grup": t1.to_dict(), "kelas_x_asal": t2.to_dict(),
                                                           "baji_hitam": wedge}, indent=2, default=str), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()

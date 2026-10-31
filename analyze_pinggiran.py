#!/usr/bin/env python3
"""
Prevalensi pinggiran bergaris dan baji hitam (Pakem v1.4, Fase 5 A5, RQ4)
=========================================================================
Merangkum keluaran cek_pinggiran_bergaris.py (detektor petak, bebas geometri) pada seluruh citra validasi bersih dan
sampel acak 1.000 citra latih (seed 2031), lalu menghubungkannya dengan kelas dominan dan split asal Roboflow.

Masukan (stage_v2_ablation/analysis/pinggiran/):
  pinggiran_val.csv, pinggiran_train1000.csv  keluaran detektor (kolom "theta" = jumlah petak bergaris)
  baji_train1000.csv                           kriteria baji hitam uji_baji_hitam.py (sudut 24 px ATAU IoU rotasi >= 0,5)
  label_visual40.csv                           40 citra validasi acak tanpa baji hitam yang dilabeli visual (urutan baris
                                               = nomor montase label40_0/1.jpg)
Set W/N validasi diambil dari stage_v2_baji/meta_baji.csv (kriteria yang sama dengan uji baji).

Pemakaian: python analyze_pinggiran.py
Keluaran: stage_v2_ablation/analysis/pinggiran/ (PINGGIRAN_REPORT.md, pinggiran_results.json)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
D = ROOT / "stage_v2_ablation/analysis/pinggiran"
KELAS = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
# Label visual (31 Okt 2026) pada 40 citra validasi acak tanpa baji hitam; nomor = baris label_visual40.csv
LABEL_POSITIF = [0, 4, 7, 13, 14, 16, 17, 19, 20, 25, 34, 35, 38]
LABEL_RAGU = [29, 36]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def pct(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "pct": 100 * k / n if n else np.nan, "ci95": [lo, hi]}


def fmt(r: dict) -> str:
    return f"{r['k']:,}/{r['n']:,} = {r['pct']:.1f}% ({r['ci95'][0]:.1f}–{r['ci95'][1]:.1f})"


def asal_split() -> pd.Series:
    M = pd.read_csv(ROOT / "cassava-v7-stage-6/resplit_group_manifest.csv", encoding="utf-8-sig")
    M = M.assign(f=M.member_filenames.str.split(";")).explode("f")
    return M.set_index(M.f.str.strip()).original_splits


def main() -> None:
    res: dict = {"tanggal": datetime.now().isoformat(timespec="seconds")}
    asal = asal_split()

    # ---------------------------------------------------------------- kalibrasi
    V = pd.read_csv(D / "pinggiran_val.csv")
    L = pd.read_csv(D / "label_visual40.csv")[["image_id"]].reset_index().rename(columns={"index": "no"})
    L["label"] = np.where(L.no.isin(LABEL_POSITIF), 1, np.where(L.no.isin(LABEL_RAGU), -1, 0))
    L = L.merge(V[["image_id", "bergaris", "theta"]], on="image_id", how="left")
    Lk = L[L.label >= 0]
    tp = int(((Lk.label == 1) & (Lk.bergaris == 1)).sum()); fn = int(((Lk.label == 1) & (Lk.bergaris == 0)).sum())
    fp = int(((Lk.label == 0) & (Lk.bergaris == 1)).sum()); tn = int(((Lk.label == 0) & (Lk.bergaris == 0)).sum())
    ragu_pos = int(L[L.label == -1].bergaris.sum())
    sens = tp / (tp + fn); spes = tn / (tn + fp)
    res["kalibrasi"] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "ragu": len(LABEL_RAGU), "ragu_terdeteksi": ragu_pos,
                        "sensitivitas": sens, "spesifisitas": spes, "prevalensi_visual": pct(len(LABEL_POSITIF), 40 - len(LABEL_RAGU))}

    # ---------------------------------------------------------------- validasi
    meta = pd.read_csv(ROOT / "stage_v2_baji/meta_baji.csv")[["image_id", "set", "kelas_dominan"]]
    V = V.merge(meta, on="image_id", how="left")
    V["asal"] = V.image_id.map(asal).fillna("?")
    V["hitam"] = (V.set == "W").astype(int)
    # ---------------------------------------------------------------- latih
    T = pd.read_csv(D / "pinggiran_train1000.csv").merge(pd.read_csv(D / "baji_train1000.csv")[["image_id", "baji_hitam"]], on="image_id")
    O = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_object_difficulty.csv", usecols=["image_id", "class_name"])
    dom = O.groupby("image_id").class_name.agg(lambda s: s.mode().iat[0])
    T["kelas_dominan"] = T.image_id.map(dom)
    T["asal"] = T.image_id.map(asal).fillna("?")
    T["hitam"] = T.baji_hitam

    def ringkas(X: pd.DataFrame) -> dict:
        n = len(X); N = X[X.hitam == 0]
        r = {"n": n,
             "baji_hitam": pct(int(X.hitam.sum()), n),
             "bergaris_tanpa_baji": pct(int(N.bergaris.sum()), len(N)),
             "bergaris_dengan_baji": pct(int(X[X.hitam == 1].bergaris.sum()), int(X.hitam.sum())),
             "artefak_rotasi_apa_pun": pct(int(((X.hitam == 1) | (X.bergaris == 1)).sum()), n),
             "bersih_tanpa_artefak_terdeteksi": pct(int(((X.hitam == 0) & (X.bergaris == 0)).sum()), n)}
        r["per_kelas"] = {}
        for c in KELAS:
            Xc = X[X.kelas_dominan == c]; Nc = Xc[Xc.hitam == 0]
            r["per_kelas"][c] = {"n": len(Xc), "baji_hitam": pct(int(Xc.hitam.sum()), len(Xc)),
                                 "bergaris_tanpa_baji": pct(int(Nc.bergaris.sum()), len(Nc)),
                                 "artefak_rotasi_apa_pun": pct(int(((Xc.hitam == 1) | (Xc.bergaris == 1)).sum()), len(Xc))}
        r["per_asal"] = {}
        for a in ["train", "valid", "test"]:
            Xa = X[X.asal == a]; Na = Xa[Xa.hitam == 0]
            if len(Xa):
                r["per_asal"][a] = {"n": len(Xa), "baji_hitam": pct(int(Xa.hitam.sum()), len(Xa)),
                                    "bergaris_tanpa_baji": pct(int(Na.bergaris.sum()), len(Na)),
                                    "artefak_rotasi_apa_pun": pct(int(((Xa.hitam == 1) | (Xa.bergaris == 1)).sum()), len(Xa))}
        r["asal_x_kelas"] = {a: {c: int(((X.asal == a) & (X.kelas_dominan == c)).sum()) for c in KELAS} for a in ["train", "valid", "test"]}
        return r

    res["validasi"] = ringkas(V)
    res["latih_sampel1000"] = ringkas(T)
    # koreksi kasar untuk sensitivitas detektor (FP ~0 pada sampel kalibrasi)
    p_det = res["validasi"]["bergaris_tanpa_baji"]["pct"]
    res["validasi"]["bergaris_tanpa_baji_koreksi_sensitivitas"] = p_det / sens

    # ---------------------------------------------------------------- laporan
    k = res["kalibrasi"]
    Lr = ["# Prevalensi pinggiran bergaris dan baji hitam (Fase 5 A5)",
          f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · validasi bersih {len(V):,} citra · sampel latih acak {len(T):,} citra (seed 2031)", "",
          "Dua artefak augmentasi rotasi Roboflow: **baji hitam** (isian konstan hitam; kriteria uji baji: sudut 24 px atau IoU rotasi ≥ 0,5) "
          "dan **pinggiran bergaris** (isian replikasi tepi; detektor petak cek_pinggiran_bergaris.py). Persentase disertai IK 95% Wilson.", "",
          "## Kalibrasi detektor pinggiran bergaris", "",
          f"40 citra validasi acak tanpa baji hitam dilabeli visual: {len(LABEL_POSITIF)} positif, {len(LABEL_RAGU)} ragu, {40 - len(LABEL_POSITIF) - len(LABEL_RAGU)} negatif.",
          f"Pada 38 citra yang pasti: TP {k['tp']}, FN {k['fn']}, FP {k['fp']}, TN {k['tn']} → sensitivitas {100 * k['sensitivitas']:.0f}%, "
          f"spesifisitas {100 * k['spesifisitas']:.0f}%. Dari 2 citra ragu, {k['ragu_terdeteksi']} terdeteksi.",
          "Pada citra yang juga berbaji hitam, tepi baji bisa ikut terbaca sebagai garis; cek visual 12 citra W bergaris memperlihatkan "
          "sekitar 9 yang benar-benar bergaris. Angka \"bergaris dengan baji\" karena itu hanya indikatif.", ""]
    for nama, r in [("Validasi bersih", res["validasi"]), ("Sampel latih", res["latih_sampel1000"])]:
        Lr += [f"## {nama}", "",
               f"- Baji hitam: {fmt(r['baji_hitam'])}",
               f"- Pinggiran bergaris pada citra tanpa baji hitam: {fmt(r['bergaris_tanpa_baji'])}",
               f"- Pinggiran bergaris pada citra berbaji hitam (indikatif, rotasi ganda): {fmt(r['bergaris_dengan_baji'])}",
               f"- Artefak rotasi apa pun (baji hitam atau bergaris): {fmt(r['artefak_rotasi_apa_pun'])}",
               f"- Tanpa artefak yang terdeteksi: {fmt(r['bersih_tanpa_artefak_terdeteksi'])}", ""]
        if nama == "Validasi bersih":
            Lr += [f"Dikoreksi untuk sensitivitas detektor (~{100 * sens:.0f}%), pinggiran bergaris pada citra tanpa baji sekitar "
                   f"{res['validasi']['bergaris_tanpa_baji_koreksi_sensitivitas']:.0f}%.", ""]
        Lr += ["| Kelas dominan | n | Baji hitam | Bergaris (tanpa baji) | Artefak apa pun |", "| --- | --- | --- | --- | --- |"]
        for c in KELAS:
            q = r["per_kelas"][c]
            Lr.append(f"| {c} | {q['n']:,} | {q['baji_hitam']['pct']:.1f}% | {q['bergaris_tanpa_baji']['pct']:.1f}% "
                      f"({q['bergaris_tanpa_baji']['k']}/{q['bergaris_tanpa_baji']['n']}) | {q['artefak_rotasi_apa_pun']['pct']:.1f}% |")
        Lr += ["", "| Split asal Roboflow | n | Baji hitam | Bergaris (tanpa baji) | Artefak apa pun |", "| --- | --- | --- | --- | --- |"]
        for a, q in r["per_asal"].items():
            Lr.append(f"| {a} | {q['n']:,} | {q['baji_hitam']['pct']:.1f}% | {q['bergaris_tanpa_baji']['pct']:.1f}% "
                      f"({q['bergaris_tanpa_baji']['k']}/{q['bergaris_tanpa_baji']['n']}) | {q['artefak_rotasi_apa_pun']['pct']:.1f}% |")
        Lr += ["", "Komposisi kelas per split asal (jumlah citra): " +
               "; ".join(f"{a}: " + ", ".join(f"{c} {n}" for c, n in d.items()) for a, d in r["asal_x_kelas"].items()), ""]
    (D / "PINGGIRAN_REPORT.md").write_text("\n".join(Lr) + "\n", encoding="utf-8")
    (D / "pinggiran_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n".join(Lr))


if __name__ == "__main__":
    main()

"""Fase 6: ringkasan test set dibanding validasi (sesuai PRA_REGISTRASI_FASE6.md).

Membaca hasil yang sudah dihitung skrip lain (tidak menghitung ulang metrik):
  validasi: stage_v2_ablation/analysis/ablation_results.json, stage_v2_conv60/analysis/conv_results.json
  test    : stage_v2_fase6/analysis/ablation_results.json, point.json, conv/conv_results.json
            stage_v2_fase6/per_image/gt_test.csv
Keluaran: stage_v2_fase6/analysis/FASE6_RINGKASAN.md dan fase6_ringkasan.json

Pakai: python analyze_fase6.py [--uji-tiruan DIR]   (uji-tiruan: buat berkas tiruan di DIR lalu ringkas)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
UTAMA = {"EXP-1 − EXP-0R": "RQ1 inti", "EXP-2 − EXP-0": "RQ2 inti"}
LABS = ["EXP-0", "EXP-0R", "EXP-1", "EXP-2", "EXP-3"]


def ringkas(p: dict) -> dict:
    av = json.loads(p["abl_val"].read_text(encoding="utf-8"))
    at = json.loads(p["abl_test"].read_text(encoding="utf-8"))
    pt = json.loads(p["point_test"].read_text(encoding="utf-8"))
    cv = json.loads(p["conv_val"].read_text(encoding="utf-8")) if p["conv_val"].exists() else None
    ct = json.loads(p["conv_test"].read_text(encoding="utf-8")) if p["conv_test"].exists() else None
    G = pd.read_csv(p["gt_test"])
    out = {"waktu": datetime.now().isoformat(timespec="seconds"), "kontras": {}, "conv": {}, "model": {}}
    for name in at:
        v, t = av.get(name), at[name]
        out["kontras"][name] = {"val": v, "test": t,
                                "verdict_sama": bool(v and v["verdict"].split(" (")[0] == t["verdict"].split(" (")[0])}
    rec = pt["recomputed"]
    for lab in LABS:
        vals = [rec[k]["mAP50_95"] * 100 for k in rec if k.split("|")[1] == lab]
        vv = [pt["val"][k]["mAP50_95"] * 100 for k in pt.get("val", {}) if k.split("|")[1] == lab]
        out["model"][lab] = {"mean": float(np.mean(vals)) if vals else None, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                             "n_seed": len(vals), "val_mean": float(np.mean(vv)) if vv else None}
    if ct:
        for name, r in ct["contrasts"].items():
            out["conv"][name] = {"test": r, "val": (cv or {}).get("contrasts", {}).get(name)}
    out["komposisi"] = {"n_citra": int(G.image_id.nunique()), "n_objek": int(len(G)),
                        "per_kelas": {CLASS_NAMES[c]: int((G.class_id == c).sum()) for c in range(5)}}
    return out


def fk(r, key="mean"):
    if not r:
        return "–"
    m = r.get(key, r.get("diff"))
    lo, hi = r["ci95"]
    return f"{m:+.2f} ({lo:+.2f} s.d. {hi:+.2f})"


def tulis(o: dict, out_dir: Path, tiruan: bool) -> None:
    L = ["# Fase 6 — test set dibanding validasi", ""]
    if tiruan:
        L += ["**DATA TIRUAN — hanya untuk menguji skrip.**", ""]
    k = o["komposisi"]
    L += [f"Dibuat {o['waktu']}. Test set: {k['n_citra']:,} citra, {k['n_objek']:,} objek GT ("
          + ", ".join(f"{c} {n:,}" for c, n in k["per_kelas"].items()) + ").", "",
          "Aturan (pra-registrasi): verdict test dilaporkan apa adanya; tidak ada pemilihan ulang atau penyetelan. "
          "Validasi tetap bukti utama; test sebagai konfirmasi.", "",
          "## mAP@50–95 per strategi (rata-rata seed, hitung ulang; val() Ultralytics sebagai pendamping)", "",
          "| Strategi | Test (rata-rata ± SD) | Seed | val() test |", "| --- | --- | --- | --- |"]
    for lab, r in o["model"].items():
        if r["mean"] is None:
            continue
        sd = f" ± {r['sd']:.2f}" if r["sd"] is not None else ""
        vm = f"{r['val_mean']:.2f}" if r["val_mean"] is not None else "–"
        L.append(f"| {lab} | {r['mean']:.2f}{sd} | {r['n_seed']} | {vm} |")
    L += ["", "## Kontras (poin mAP@50–95; IK 95% bootstrap klaster per grup foto)", "",
          "| Kontras | Peran | Validasi | Verdict validasi | Test | Verdict test | Sama? |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in o["kontras"].items():
        v, t = r["val"], r["test"]
        L.append(f"| {name} | {UTAMA.get(name, 'pendukung')} | {fk(v)} | {v['verdict'] if v else '–'} | {fk(t)} | "
                 f"{t['verdict']} | {'ya' if r['verdict_sama'] else 'TIDAK'} |")
    if o["conv"]:
        L += ["", "## CONV (seed 42; deskriptif, tanpa aturan keputusan tiga seed)", "",
              "| Kontras | Validasi | Test |", "| --- | --- | --- |"]
        for name, r in o["conv"].items():
            L.append(f"| {name} | {fk(r['val'], 'diff')} | {fk(r['test'], 'diff')} |")
    L += ["", "Rincian: ABLATION_REPORT.md (test), conv/CONV_REPORT.md, ukuran/UKURAN_REPORT.md, error/ERROR_REPORT.md."]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FASE6_RINGKASAN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "fase6_ringkasan.json").write_text(json.dumps(o, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


def tiruan(d: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = ["EXP-0R − EXP-0", "EXP-1 − EXP-0R", "EXP-1 − EXP-0", "EXP-2 − EXP-0", "EXP-3 − EXP-0"]

    def abl(shift):
        r = {}
        for n in names:
            m = float(rng.normal(shift, 0.5))
            r[n] = {"per_seed": {}, "mean": m, "ci95": [m - 0.6, m + 0.6],
                    "verdict": "menurunkan secara konsisten" if m + 0.6 < 0 else "tidak terbukti berbeda"}
        return r
    (d / "abl_val.json").write_text(json.dumps(abl(-1.0)))
    (d / "abl_test.json").write_text(json.dumps(abl(-0.8)))
    rec = {f"{s}|{lab}": {"mAP50_95": 0.25 + rng.normal(0, .005)} for s in (42, 123, 2026) for lab in LABS}
    val = {k: {"mAP50_95": v["mAP50_95"] + 0.005} for k, v in rec.items()}
    (d / "point.json").write_text(json.dumps({"recomputed": rec, "val": val, "n_images": 1282}))
    conv = {"contrasts": {"EXP-1 − EXP-0 pada 30ep": {"diff": -1.3, "ci95": [-2, -.6]}}}
    (d / "conv_val.json").write_text(json.dumps(conv))
    (d / "conv_test.json").write_text(json.dumps(conv))
    pd.DataFrame({"image_id": [f"i{i % 1282}" for i in range(3000)], "class_id": rng.integers(0, 5, 3000)}).to_csv(d / "gt.csv", index=False)
    return {"abl_val": d / "abl_val.json", "abl_test": d / "abl_test.json", "point_test": d / "point.json",
            "conv_val": d / "conv_val.json", "conv_test": d / "conv_test.json", "gt_test": d / "gt.csv", "out": d / "analysis"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uji-tiruan", type=Path, default=None)
    a = ap.parse_args()
    if a.uji_tiruan:
        p = tiruan(a.uji_tiruan)
        tulis(ringkas(p), p["out"], True)
        return
    f6 = ROOT / "stage_v2_fase6"
    p = {"abl_val": ROOT / "stage_v2_ablation/analysis/ablation_results.json",
         "abl_test": f6 / "analysis/ablation_results.json", "point_test": f6 / "analysis/point.json",
         "conv_val": ROOT / "stage_v2_conv60/analysis/conv_results.json",
         "conv_test": f6 / "analysis/conv/conv_results.json", "gt_test": f6 / "per_image/gt_test.csv", "out": f6 / "analysis"}
    tulis(ringkas(p), p["out"], False)


if __name__ == "__main__":
    main()

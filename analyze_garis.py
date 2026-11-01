#!/usr/bin/env python3
"""
Analisis uji jalan pintas pinggiran bergaris (Pakem v1.4.1) - keluaran uji_pinggiran_bergaris.py
================================================================================================
Pra-registrasi (docstring uji_pinggiran_bergaris.py, 31 Okt 2026 malam):
  Primer    subset G_tr (kelas dominan CGM/CMD/HEALTHY), G_garis - G_rata: porsi massa keyakinan (conf >= 0,25) ke CBB
            (hipotesis jalan pintas: naik) dan rata-rata AP CGM/CMD/HEALTHY (hipotesis: turun).
  Pendukung G_tr: porsi ke CBB/CBSD; persentase citra yang prediksi teratasnya CBB (deskriptif).
  Sekunder  subset G_cb (kelas dominan CBB/CBSD), G_garis - G_rata: AP CBB, AP CBSD, porsi ke CBB/CBSD (hipotesis: naik).
  Deskriptif G_garis - G_asli dan G_rata - G_asli; sensitivitas (estimasi titik) pada citra yang G_garis-nya terdeteksi
            bergaris, G_rata-nya tidak, dan kotak GT-nya tertutup isian < 30%.
  Verdict utama pada EXP-0 tiga seed; EXP-1 dan EXP-2 dilaporkan dengan cara yang sama, plus selisih efek primer
  terhadap EXP-0 (berpasangan per seed, resampel bootstrap sama).
  Aturan keputusan: "terbukti" bila tanda selisih sama di ketiga seed dan IK 95% (bootstrap klaster per grup foto,
  1.000 iterasi) tidak memuat nol.

Pemakaian: python analyze_garis.py prep ; python analyze_garis.py boot --seed 0 --iters 250 (seed 0-3) ;
           python analyze_garis.py report
Keluaran: stage_v2_garis/analysis/ (GARIS_REPORT.md, garis_results.json, boot_seed*.csv, cache/)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_ablation import split_by_class  # noqa: E402
from analyze_baji import metrik as metrik_baji, prepare_df  # noqa: E402
from analyze_eleak_per_image import CLASS_NAMES, load_groups  # noqa: E402

VARIAN = ["G_asli", "G_garis", "G_rata"]
GRUP = {"EXP-0": "exp0", "EXP-1": "exp1", "EXP-2": "exp2"}
SEEDS = ["s42", "s123", "s2026"]
SUBSETS = {
    "G_tr": lambda m: m.kelas_dominan.isin(["CGM", "CMD", "HEALTHY"]),
    "G_cb": lambda m: m.kelas_dominan.isin(["CBB", "CBSD"]),
    "G_tr_sens": lambda m: m.kelas_dominan.isin(["CGM", "CMD", "HEALTHY"]) & (m.garis_terdeteksi == 1)
    & (m.rata_terdeteksi == 0) & (m.gt_tutupan_maks < 0.3),
    "G_cb_sens": lambda m: m.kelas_dominan.isin(["CBB", "CBSD"]) & (m.garis_terdeteksi == 1)
    & (m.rata_terdeteksi == 0) & (m.gt_tutupan_maks < 0.3),
}
BOOT_SUBSETS = ["G_tr", "G_cb"]
CONTRASTS = [
    # (nama, subset, {varian: koef}, metrik, peran)
    ("Garis − rata: porsi ke CBB", "G_tr", {"G_garis": 1, "G_rata": -1}, "porsi_CBB", "primer"),
    ("Garis − rata: rata-rata AP CGM/CMD/HEALTHY", "G_tr", {"G_garis": 1, "G_rata": -1}, "map_train", "primer"),
    ("Garis − rata: porsi ke CBB/CBSD", "G_tr", {"G_garis": 1, "G_rata": -1}, "porsi_vt", "pendukung"),
    ("Garis − rata: AP CBB", "G_cb", {"G_garis": 1, "G_rata": -1}, "ap_CBB", "sekunder"),
    ("Garis − rata: AP CBSD", "G_cb", {"G_garis": 1, "G_rata": -1}, "ap_CBSD", "sekunder"),
    ("Garis − rata: porsi ke CBB/CBSD (citra CBB/CBSD)", "G_cb", {"G_garis": 1, "G_rata": -1}, "porsi_vt", "sekunder"),
    ("Garis − asli: porsi ke CBB", "G_tr", {"G_garis": 1, "G_asli": -1}, "porsi_CBB", "deskriptif"),
    ("Rata − asli: porsi ke CBB", "G_tr", {"G_rata": 1, "G_asli": -1}, "porsi_CBB", "deskriptif"),
    ("Garis − asli: rata-rata AP CGM/CMD/HEALTHY", "G_tr", {"G_garis": 1, "G_asli": -1}, "map_train", "deskriptif"),
    ("Rata − asli: rata-rata AP CGM/CMD/HEALTHY", "G_tr", {"G_rata": 1, "G_asli": -1}, "map_train", "deskriptif"),
    ("Garis − asli: AP CBB", "G_cb", {"G_garis": 1, "G_asli": -1}, "ap_CBB", "deskriptif"),
    ("Rata − asli: AP CBB", "G_cb", {"G_rata": 1, "G_asli": -1}, "ap_CBB", "deskriptif"),
]
SENS = [("G_tr_sens", "porsi_CBB"), ("G_tr_sens", "map_train"), ("G_cb_sens", "ap_CBB"), ("G_cb_sens", "porsi_vt")]


def metrik(sp: dict, z: dict, w: np.ndarray) -> dict:
    r = metrik_baji(sp, z, w)
    tot = z["massa"].sum(1)
    ok = (tot > 0) & (w > 0)
    ww = w[ok].astype(float)
    r["porsi_CBB"] = float((z["massa"][ok][:, 0] / tot[ok] * ww).sum() / ww.sum() * 100)
    return r


def verdict(per: list[float], lo: float, hi: float) -> str:
    if all(d > 0 for d in per) and lo > 0:
        return "naik, terbukti"
    if all(d < 0 for d in per) and hi < 0:
        return "turun, terbukti"
    return "tidak terbukti berbeda"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "boot", "report"])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dir", default="stage_v2_garis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--grup", default="EXP-0,EXP-1,EXP-2", help="grup model yang dianalisis")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    bd = root / args.dir
    out = bd / "analysis"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(bd / "meta_garis.csv")
    grup = [g for g in args.grup.split(",") if g]
    models = [f"{GRUP[g]}_{s}" for g in grup for s in SEEDS]
    imgs_of = {sub: sorted(meta[f(meta)].image_id) for sub, f in SUBSETS.items()}

    if args.mode == "prep":
        G = pd.read_csv(bd / "gt_cleanval.csv")
        for mdl in models:
            for v in VARIAN:
                P = None
                for sub, imgs in imgs_of.items():
                    dst = cache / f"{sub}__{mdl}__{v}.npz"
                    if dst.exists():
                        continue
                    if P is None:
                        P = pd.read_csv(bd / f"preds_{mdl}_{v}.csv")
                    s = set(imgs)
                    np.savez_compressed(dst, **prepare_df(P[P.image_id.isin(s)], G[G.image_id.isin(s)], imgs))
            print(f"cache {mdl}", flush=True)
        print({k: len(v) for k, v in imgs_of.items()})
        return

    def load(sub):
        d = {}
        for mdl in models:
            for v in VARIAN:
                z = dict(np.load(cache / f"{sub}__{mdl}__{v}.npz"))
                d[(mdl, v)] = (split_by_class(z), z)
        return d

    groups = load_groups(root)
    if args.mode == "boot":
        data = {sub: load(sub) for sub in BOOT_SUBSETS}
        rng = np.random.default_rng(8000 + args.seed)
        gi = {}
        for sub in BOOT_SUBSETS:
            g = np.array([groups.get(n, n) for n in imgs_of[sub]])
            u, inv = np.unique(g, return_inverse=True)
            gi[sub] = (len(u), inv)
        rows = []
        for _ in range(args.iters):
            r = {}
            for sub in BOOT_SUBSETS:
                nu, inv = gi[sub]
                w = np.bincount(rng.integers(0, nu, nu), minlength=nu)[inv]
                for (mdl, v), (sp, z) in data[sub].items():
                    for k, val in metrik(sp, z, w).items():
                        r[f"{sub}|{mdl}|{v}|{k}"] = val
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / f"boot_seed{args.seed}.csv", index=False)
        print(f"boot seed {args.seed}: {len(rows)} iterasi")
        return

    # ------------------------------------------------------------------ report
    point, top1 = {}, {}
    for sub in SUBSETS:
        if not imgs_of[sub]:
            continue
        for (mdl, v), (sp, z) in load(sub).items():
            point[f"{sub}|{mdl}|{v}"] = metrik(sp, z, np.ones(len(z["images"]), dtype=np.int64))
            top1[f"{sub}|{mdl}|{v}"] = z["top1"]
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "boot_seed*.csv")))], ignore_index=True)
    res = {"tanggal": datetime.now().isoformat(timespec="seconds"), "n_citra": {k: len(v) for k, v in imgs_of.items()},
           "point": point, "contrasts": {}, "selisih_efek_vs_exp0": {}, "sensitivitas": {}, "top1_CBB_pct": {}}
    man = {"masker_pct_median": float(meta.masker_pct.median()), "masker_pct_p90": float(meta.masker_pct.quantile(0.9)),
           "garis_terdeteksi_pct": float(meta.garis_terdeteksi.mean() * 100), "rata_terdeteksi_pct": float(meta.rata_terdeteksi.mean() * 100),
           "petak_garis_q": [float(x) for x in meta[meta.garis_terdeteksi == 1].petak_garis.quantile([0.25, 0.5, 0.75])],
           "gt_tutupan_ge_0_3_citra_pct": float((meta.gt_tutupan_maks >= 0.3).mean() * 100)}
    res["cek_manipulasi"] = man
    L = ["# Uji jalan pintas pinggiran bergaris — hasil",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · model {', '.join(grup)} × seed 42/123/2026 · bootstrap klaster per grup {len(B)} iterasi", "",
         "## Himpunan citra dan cek manipulasi", "",
         f"- Set G: {len(meta):,} citra validasi tanpa baji hitam dan tanpa garis terdeteksi. Subset: " +
         ", ".join(f"{k} {len(v):,}" for k, v in imgs_of.items()) + ".",
         f"- Luas isian median {man['masker_pct_median']:.1f}% (p90 {man['masker_pct_p90']:.1f}%); citra dengan kotak GT tertutup ≥ 30%: "
         f"{man['gt_tutupan_ge_0_3_citra_pct']:.1f}%.",
         f"- Detektor petak: G_garis terdeteksi bergaris {man['garis_terdeteksi_pct']:.1f}%, G_rata {man['rata_terdeteksi_pct']:.1f}%; "
         f"kuartil petak bergaris pada G_garis yang terdeteksi {man['petak_garis_q'][0]:.0f}/{man['petak_garis_q'][1]:.0f}/{man['petak_garis_q'][2]:.0f} "
         "(garis asli di validasi: 3/5/7).", ""]
    mets = {"G_tr": ["porsi_CBB", "porsi_vt", "map_train", "ap_CGM", "ap_CMD", "ap_HEALTHY", "map"],
            "G_cb": ["porsi_CBB", "porsi_vt", "ap_CBB", "ap_CBSD", "map"]}
    L += ["## Nilai per varian (rata-rata tiga seed)", ""]
    for g in grup:
        ms = [f"{GRUP[g]}_{s}" for s in SEEDS]
        for sub in BOOT_SUBSETS:
            L += [f"**{g}, {sub}** ({len(imgs_of[sub]):,} citra)", "", "| Varian | " + " | ".join(mets[sub]) + " |",
                  "| --- |" + " --- |" * len(mets[sub])]
            for v in VARIAN:
                L.append(f"| {v} | " + " | ".join(f"{np.nanmean([point[f'{sub}|{m}|{v}'][k] for m in ms]):.2f}" for k in mets[sub]) + " |")
            L.append("")
    L += ["porsi_CBB = rata-rata per citra dari bagian massa keyakinan (prediksi conf ≥ 0,25) yang jatuh ke CBB; porsi_vt = ke CBB/CBSD; "
          "map_train = rata-rata AP CGM/CMD/HEALTHY. AP dalam poin persen (AP@50–95).", "",
          "## Kontras (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)", "",
          "| Model | Peran | Kontras | Subset | Per seed | Rata-rata | IK 95% | Verdict |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in grup:
        ms = [f"{GRUP[g]}_{s}" for s in SEEDS]
        for name, sub, coef, k, peran in CONTRASTS:
            per = [sum(c * point[f"{sub}|{m}|{v}"][k] for v, c in coef.items()) for m in ms]
            boot = np.mean([sum(c * B[f"{sub}|{m}|{v}|{k}"].to_numpy() for v, c in coef.items()) for m in ms], axis=0)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            vd = verdict(per, lo, hi)
            res["contrasts"][f"{g}|{name}"] = {"subset": sub, "metrik": k, "peran": peran, "per_seed": per,
                                               "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)], "verdict": vd}
            L.append(f"| {g} | {peran} | {name} | {sub} | " + " / ".join(f"{d:+.2f}" for d in per) +
                     f" | {np.mean(per):+.2f} | {lo:+.2f} s.d. {hi:+.2f} | {vd} |")
    if "EXP-0" in grup and len(grup) > 1:
        L += ["", "## Selisih efek garis − rata terhadap EXP-0 (berpasangan per seed, resampel sama)", "",
              "| Kontras | Subset | Metrik | Per seed | Rata-rata | IK 95% | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for g in [x for x in grup if x != "EXP-0"]:
            for sub, k in [("G_tr", "porsi_CBB"), ("G_tr", "map_train"), ("G_cb", "ap_CBB"), ("G_cb", "porsi_vt")]:
                per, boots = [], []
                for s in SEEDS:
                    e = lambda pre, src: (src(f"{sub}|{pre}_{s}|G_garis", k) - src(f"{sub}|{pre}_{s}|G_rata", k))
                    pt = lambda key, kk: point[key][kk]
                    bt = lambda key, kk: B[f"{key}|{kk}"].to_numpy()
                    per.append(e(GRUP[g], pt) - e("exp0", pt))
                    boots.append(e(GRUP[g], bt) - e("exp0", bt))
                lo, hi = np.nanpercentile(np.mean(boots, axis=0), [2.5, 97.5])
                vd = verdict(per, lo, hi)
                res["selisih_efek_vs_exp0"][f"{g} − EXP-0|{sub}|{k}"] = {"per_seed": per, "mean": float(np.mean(per)),
                                                                          "ci95": [float(lo), float(hi)], "verdict": vd}
                L.append(f"| {g} − EXP-0 | {sub} | {k} | " + " / ".join(f"{d:+.2f}" for d in per) +
                         f" | {np.mean(per):+.2f} | {lo:+.2f} s.d. {hi:+.2f} | {vd} |")
    L += ["", "## Prediksi teratas CBB per citra (deskriptif, rata-rata tiga seed)", "", "| Model | Subset | Varian | % citra |", "| --- | --- | --- | --- |"]
    for g in grup:
        ms = [f"{GRUP[g]}_{s}" for s in SEEDS]
        for sub in BOOT_SUBSETS:
            for v in VARIAN:
                pct = float(np.mean([np.mean(top1[f"{sub}|{m}|{v}"] == 0) * 100 for m in ms]))
                res["top1_CBB_pct"][f"{g}|{sub}|{v}"] = pct
                L.append(f"| {g} | {sub} | {v} | {pct:.1f} |")
    L += ["", "## Sensitivitas garis − rata (estimasi titik, rata-rata tiga seed)", "", "| Model | Subset | Metrik | Selisih |", "| --- | --- | --- | --- |"]
    for g in grup:
        ms = [f"{GRUP[g]}_{s}" for s in SEEDS]
        for sub, k in SENS:
            if not imgs_of[sub]:
                continue
            d = float(np.mean([point[f"{sub}|{m}|G_garis"][k] - point[f"{sub}|{m}|G_rata"][k] for m in ms]))
            res["sensitivitas"][f"{g}|{sub}|{k}"] = d
            L.append(f"| {g} | {sub} ({len(imgs_of[sub]):,}) | {k} | {d:+.2f} |")
    L += ["", "Catatan: kotak GT tidak diubah; garis dan warna rata menutup area yang sama dengan warna rata-rata yang sama, "
          "sehingga kontras garis − rata hanya menguji tekstur garis. IK dari bootstrap klaster per grup foto (resampel sama "
          "untuk semua model dan varian), dirata-rata antar seed."]
    (out / "GARIS_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "garis_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analisis uji jalan pintas baji hitam (Pakem v1.3.2) - keluaran uji_baji_hitam.py
=================================================================================
Pra-registrasi (MALAM_INI.md, 30 Okt 2026):
  Primer   set N, citra berkelas dominan CBB/CBSD: AP@50-95 CBB dan CBSD, serta porsi massa keyakinan (conf >= 0,25)
           yang jatuh ke CGM/CMD/HEALTHY; kontras N_baji_hitam - N_baji_rata.
  Sekunder set W (citra berbaji): W_asli - W_rata dan W_asli - W_inpaint; AP per kelas dan porsi ke CBB/CBSD.
  Keputusan: "terbukti" bila tanda selisih sama di ketiga seed (EXP-0 seed 42/123/2026) dan IK 95% (bootstrap klaster
  per grup foto, 1.000 iterasi, resampel yang sama untuk semua model dan varian) tidak memuat nol.
Tambahan deskriptif: efek total (hitam - asli), efek oklusi (rata - asli), AP per kelas di seluruh set N, perpindahan kelas
teratas per citra, dan analisis sensitivitas (citra N yang bajinya terdeteksi; citra W dengan estimasi sudut andal).

Pemakaian: python analyze_baji.py prep ; python analyze_baji.py boot --seed 0 --iters 250 ; python analyze_baji.py report
Keluaran: stage_v2_baji/analysis/ (BAJI_REPORT.md, baji_results.json, boot_seed*.csv, cache/)
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
from analyze_ablation import fast_map, split_by_class  # noqa: E402
from analyze_eleak_per_image import CLASS_NAMES, iou_matrix, load_groups, match_predictions  # noqa: E402

MODELS = ["exp0_s42", "exp0_s123", "exp0_s2026"]
VAR = {"N": ["N_asli", "N_baji_hitam", "N_baji_rata"], "W": ["W_asli", "W_rata", "W_inpaint"]}
TRAIN_CLS = [2, 3, 4]   # CGM, CMD, HEALTHY (asal train Roboflow, sering berbaji)
VT_CLS = [0, 1]         # CBB, CBSD (asal valid/test, hampir tanpa baji)
TAU = 0.25
# subset: (nama, set, filter meta)
SUBSETS = {
    "N_cb": ("N", lambda m: m.kelas_dominan.isin(["CBB", "CBSD"])),
    "N_semua": ("N", lambda m: m.kelas_dominan.notna()),
    "W_semua": ("W", lambda m: m.kelas_dominan.notna()),
    "N_cb_sens": ("N", lambda m: m.kelas_dominan.isin(["CBB", "CBSD"]) & (m.baji_sesudah_hitam_atau_asli == 1)),
    "W_sens": ("W", lambda m: m.iou_estimasi >= 0.5),
}
BOOT_SUBSETS = ["N_cb", "W_semua"]
CONTRASTS = [
    # (nama, subset, {varian: koef}, metrik, peran)
    ("Hitam − rata: AP CBB", "N_cb", {"N_baji_hitam": 1, "N_baji_rata": -1}, "ap_CBB", "primer"),
    ("Hitam − rata: AP CBSD", "N_cb", {"N_baji_hitam": 1, "N_baji_rata": -1}, "ap_CBSD", "primer"),
    ("Hitam − rata: porsi ke CGM/CMD/HEALTHY", "N_cb", {"N_baji_hitam": 1, "N_baji_rata": -1}, "porsi_train", "primer"),
    ("Hitam − rata: mAP subset", "N_cb", {"N_baji_hitam": 1, "N_baji_rata": -1}, "map", "pendukung"),
    ("Hitam − asli: AP CBB (efek total)", "N_cb", {"N_baji_hitam": 1, "N_asli": -1}, "ap_CBB", "deskriptif"),
    ("Hitam − asli: AP CBSD (efek total)", "N_cb", {"N_baji_hitam": 1, "N_asli": -1}, "ap_CBSD", "deskriptif"),
    ("Hitam − asli: porsi ke CGM/CMD/HEALTHY", "N_cb", {"N_baji_hitam": 1, "N_asli": -1}, "porsi_train", "deskriptif"),
    ("Rata − asli: AP CBB (oklusi)", "N_cb", {"N_baji_rata": 1, "N_asli": -1}, "ap_CBB", "deskriptif"),
    ("Rata − asli: AP CBSD (oklusi)", "N_cb", {"N_baji_rata": 1, "N_asli": -1}, "ap_CBSD", "deskriptif"),
    ("Rata − asli: porsi ke CGM/CMD/HEALTHY", "N_cb", {"N_baji_rata": 1, "N_asli": -1}, "porsi_train", "deskriptif"),
    ("Asli − rata: porsi ke CBB/CBSD", "W_semua", {"W_asli": 1, "W_rata": -1}, "porsi_vt", "sekunder"),
    ("Asli − inpaint: porsi ke CBB/CBSD", "W_semua", {"W_asli": 1, "W_inpaint": -1}, "porsi_vt", "sekunder"),
    ("Asli − rata: rata-rata AP CGM/CMD/HEALTHY", "W_semua", {"W_asli": 1, "W_rata": -1}, "map_train", "sekunder"),
    ("Asli − inpaint: rata-rata AP CGM/CMD/HEALTHY", "W_semua", {"W_asli": 1, "W_inpaint": -1}, "map_train", "sekunder"),
    ("Asli − rata: AP CGM", "W_semua", {"W_asli": 1, "W_rata": -1}, "ap_CGM", "sekunder"),
    ("Asli − rata: AP CMD", "W_semua", {"W_asli": 1, "W_rata": -1}, "ap_CMD", "sekunder"),
    ("Asli − rata: AP HEALTHY", "W_semua", {"W_asli": 1, "W_rata": -1}, "ap_HEALTHY", "sekunder"),
    ("Asli − inpaint: AP CGM", "W_semua", {"W_asli": 1, "W_inpaint": -1}, "ap_CGM", "sekunder"),
    ("Asli − inpaint: AP CMD", "W_semua", {"W_asli": 1, "W_inpaint": -1}, "ap_CMD", "sekunder"),
    ("Asli − inpaint: AP HEALTHY", "W_semua", {"W_asli": 1, "W_inpaint": -1}, "ap_HEALTHY", "sekunder"),
]


def prepare_df(P: pd.DataFrame, G: pd.DataFrame, images: list[str]) -> dict:
    """Sama dengan analyze_eleak_per_image.prepare_run, tetapi untuk daftar citra tetap dan DataFrame."""
    pg = P.groupby("image_id").indices
    gg = G.groupby("image_id").indices
    Pb = P[["x1", "y1", "x2", "y2"]].to_numpy(float); Pc = P.class_id.to_numpy(int); Pf = P.confidence.to_numpy(float)
    Gb = G[["x1", "y1", "x2", "y2"]].to_numpy(float); Gc = G.class_id.to_numpy(int)
    nc = len(CLASS_NAMES)
    det_img, det_cls, det_conf, det_tp = [], [], [], []
    gt_counts = np.zeros((len(images), nc), dtype=np.int64)
    massa = np.zeros((len(images), nc))          # jumlah conf >= TAU per kelas
    top1 = np.full(len(images), -1, dtype=np.int64)
    for k, im in enumerate(images):
        gi = gg.get(im, np.array([], dtype=int)); pi = pg.get(im, np.array([], dtype=int))
        gcls = Gc[gi]; pcls = Pc[pi]; pconf = Pf[pi]
        for c in gcls:
            gt_counts[k, c] += 1
        cor = match_predictions(gcls, pcls, iou_matrix(Gb[gi], Pb[pi]))
        det_img.append(np.full(len(pi), k)); det_cls.append(pcls); det_conf.append(pconf); det_tp.append(cor)
        hi = pconf >= TAU
        np.add.at(massa[k], pcls[hi], pconf[hi])
        if len(pi):
            top1[k] = pcls[np.argmax(pconf)]
    det_img = np.concatenate(det_img); det_cls = np.concatenate(det_cls)
    det_conf = np.concatenate(det_conf); det_tp = np.concatenate(det_tp)
    order = np.argsort(-det_conf, kind="stable")
    return {"images": np.array(images), "det_img": det_img[order], "det_cls": det_cls[order], "det_conf": det_conf[order],
            "det_tp": det_tp[order], "gt_counts": gt_counts, "massa": massa, "top1": top1}


def metrik(sp: dict, z: dict, w: np.ndarray) -> dict:
    m50, m, pc = fast_map(sp, w, per_class=True)
    r = {"map": m * 100, "map50": m50 * 100}
    for c, nm in enumerate(CLASS_NAMES):
        r[f"ap_{nm}"] = float(np.nanmean(pc[c]) * 100) if not np.all(np.isnan(pc[c])) else np.nan
    tot = z["massa"].sum(1)
    ok = (tot > 0) & (w > 0)
    ww = w[ok].astype(float)
    r["map_train"] = float(np.nanmean([r["ap_CGM"], r["ap_CMD"], r["ap_HEALTHY"]]))
    r["porsi_train"] = float((z["massa"][ok][:, TRAIN_CLS].sum(1) / tot[ok] * ww).sum() / ww.sum() * 100)
    r["porsi_vt"] = float((z["massa"][ok][:, VT_CLS].sum(1) / tot[ok] * ww).sum() / ww.sum() * 100)
    return r


def subset_images(meta: pd.DataFrame, name: str) -> list[str]:
    s, f = SUBSETS[name]
    m = meta[meta.set == s]
    return sorted(m[f(m)].image_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "boot", "report"])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dir", default="stage_v2_baji")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=250)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    bd = root / args.dir
    out = bd / "analysis"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(bd / "meta_baji.csv")

    if args.mode == "prep":
        G = pd.read_csv(bd / "gt_cleanval.csv")
        for sub in SUBSETS:
            imgs = subset_images(meta, sub)
            s = SUBSETS[sub][0]
            Gs = G[G.image_id.isin(set(imgs))]
            for mdl in MODELS:
                for v in VAR[s]:
                    dst = cache / f"{sub}__{mdl}__{v}.npz"
                    if dst.exists():
                        continue
                    P = pd.read_csv(bd / f"preds_{mdl}_{v}.csv")
                    P = P[P.image_id.isin(set(imgs))]
                    np.savez_compressed(dst, **prepare_df(P, Gs, imgs))
            print(f"{sub}: {len(imgs)} citra")
        return

    def load(sub):
        s = SUBSETS[sub][0]
        d = {}
        for mdl in MODELS:
            for v in VAR[s]:
                z = dict(np.load(cache / f"{sub}__{mdl}__{v}.npz"))
                d[(mdl, v)] = (split_by_class(z), z)
        return d

    groups = load_groups(root)
    if args.mode == "boot":
        rows = []
        data = {sub: load(sub) for sub in BOOT_SUBSETS}
        rng = np.random.default_rng(7000 + args.seed)
        gi = {}
        for sub in BOOT_SUBSETS:
            imgs = next(iter(data[sub].values()))[1]["images"]
            g = np.array([groups.get(n, n) for n in imgs])
            u, inv = np.unique(g, return_inverse=True)
            gi[sub] = (len(u), inv)
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
        for (mdl, v), (sp, z) in load(sub).items():
            point[f"{sub}|{mdl}|{v}"] = metrik(sp, z, np.ones(len(z["images"]), dtype=np.int64))
            top1[f"{sub}|{mdl}|{v}"] = z["top1"]
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "boot_seed*.csv")))], ignore_index=True)
    for sub in BOOT_SUBSETS:  # metrik turunan untuk file boot lama yang belum memuat map_train
        for mdl in MODELS:
            for v in VAR[SUBSETS[sub][0]]:
                k = f"{sub}|{mdl}|{v}|"
                if k + "map_train" not in B:
                    B[k + "map_train"] = B[[k + "ap_CGM", k + "ap_CMD", k + "ap_HEALTHY"]].mean(axis=1)
    G = pd.read_csv(bd / "gt_cleanval.csv")

    n_img = {sub: len(subset_images(meta, sub)) for sub in SUBSETS}
    res = {"n_citra": n_img, "point": point, "contrasts": {}, "sensitivitas": {}}
    L = ["# Uji jalan pintas baji hitam — hasil",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · model EXP-0 seed 42/123/2026 · bootstrap klaster per grup {len(B)} iterasi",
         "", "## Himpunan citra", "",
         f"- Set N (tanpa baji): {int((meta.set == 'N').sum()):,} citra; subset primer N berkelas dominan CBB/CBSD: {n_img['N_cb']:,} citra.",
         f"- Set W (berbaji): {int((meta.set == 'W').sum()):,} citra; {int(((meta.set == 'W') & (meta.kriteria_lama_sudut24 == 0)).sum())} di antaranya hanya terdeteksi lewat estimasi sudut.",
         f"- Kelas dominan set W: " + ", ".join(f"{k} {v}" for k, v in meta[meta.set == 'W'].kelas_dominan.value_counts().items()),
         f"- Kelas dominan set N: " + ", ".join(f"{k} {v}" for k, v in meta[meta.set == 'N'].kelas_dominan.value_counts().items()),
         "", "Objek GT per kelas: " + "; ".join(
             f"{sub} " + ", ".join(f"{CLASS_NAMES[c]} {n}" for c, n in G[G.image_id.isin(set(subset_images(meta, sub)))].class_id.value_counts().sort_index().items())
             for sub in ["N_cb", "W_semua"]) + ". AP CBB di set W tidak dilaporkan (hanya 2 objek).",
         "", "## Nilai per varian (rata-rata tiga seed)", ""]
    for sub, mets in [("N_cb", ["map", "ap_CBB", "ap_CBSD", "porsi_train", "porsi_vt"]),
                      ("W_semua", ["map_train", "ap_CGM", "ap_CMD", "ap_HEALTHY", "ap_CBSD", "porsi_vt", "porsi_train"]),
                      ("N_semua", ["map"] + [f"ap_{c}" for c in CLASS_NAMES] + ["porsi_vt"])]:
        s = SUBSETS[sub][0]
        L += [f"**{sub}** ({n_img[sub]:,} citra)", "", "| Varian | " + " | ".join(mets) + " |", "| --- |" + " --- |" * len(mets)]
        for v in VAR[s]:
            L.append(f"| {v} | " + " | ".join(f"{np.nanmean([point[f'{sub}|{m}|{v}'][k] for m in MODELS]):.2f}" for k in mets) + " |")
        L.append("")
    L += ["Porsi = rata-rata per citra dari bagian massa keyakinan (prediksi conf ≥ 0,25) yang jatuh ke kelompok kelas tersebut; "
          "porsi_train = CGM/CMD/HEALTHY, porsi_vt = CBB/CBSD. AP dalam poin persen (AP@50–95); map_train = rata-rata AP CGM/CMD/HEALTHY.", "",
          "## Kontras (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)", "",
          "| Peran | Kontras | Subset | Per seed | Rata-rata | IK 95% | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, sub, coef, k, peran in CONTRASTS:
        per = [sum(c * point[f"{sub}|{m}|{v}"][k] for v, c in coef.items()) for m in MODELS]
        boot = np.mean([sum(c * B[f"{sub}|{m}|{v}|{k}"] for v, c in coef.items()) for m in MODELS], axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        mean = float(np.mean(per))
        if all(d > 0 for d in per) and lo > 0:
            vd = "naik, terbukti"
        elif all(d < 0 for d in per) and hi < 0:
            vd = "turun, terbukti"
        else:
            vd = "tidak terbukti berbeda"
        res["contrasts"][name] = {"subset": sub, "metrik": k, "peran": peran, "per_seed": per, "mean": mean,
                                  "ci95": [float(lo), float(hi)], "verdict": vd}
        L.append(f"| {peran} | {name} | {sub} | " + " / ".join(f"{d:+.2f}" for d in per) + f" | {mean:+.2f} | {lo:+.2f} s.d. {hi:+.2f} | {vd} |")

    # Perpindahan kelas teratas per citra (deskriptif)
    L += ["", "## Perpindahan kelas teratas per citra (deskriptif, rata-rata tiga seed)", "",
          "Persentase citra yang prediksi berkeyakinan tertinggi berada di kelompok CGM/CMD/HEALTHY.", "",
          "| Subset | Varian | % citra |", "| --- | --- | --- |"]
    for sub in ["N_cb", "W_semua"]:
        for v in VAR[SUBSETS[sub][0]]:
            pct = np.mean([np.mean(np.isin(top1[f"{sub}|{m}|{v}"], TRAIN_CLS)) * 100 for m in MODELS])
            res.setdefault("top1_train_pct", {})[f"{sub}|{v}"] = float(pct)
            L.append(f"| {sub} | {v} | {pct:.1f} |")

    # Sensitivitas (estimasi titik saja)
    L += ["", "## Sensitivitas (estimasi titik, rata-rata tiga seed)", "", "| Subset | Kontras | Metrik | Selisih |", "| --- | --- | --- | --- |"]
    for sub, coef, k in [("N_cb_sens", {"N_baji_hitam": 1, "N_baji_rata": -1}, "ap_CBB"),
                         ("N_cb_sens", {"N_baji_hitam": 1, "N_baji_rata": -1}, "ap_CBSD"),
                         ("N_cb_sens", {"N_baji_hitam": 1, "N_baji_rata": -1}, "porsi_train"),
                         ("W_sens", {"W_asli": 1, "W_rata": -1}, "porsi_vt"),
                         ("W_sens", {"W_asli": 1, "W_inpaint": -1}, "porsi_vt")]:
        d = float(np.mean([sum(c * point[f"{sub}|{m}|{v}"][k] for v, c in coef.items()) for m in MODELS]))
        nm = " − ".join(coef)
        res["sensitivitas"][f"{sub}|{nm}|{k}"] = d
        L.append(f"| {sub} ({n_img[sub]:,}) | {nm} | {k} | {d:+.2f} |")
    L += ["", "Catatan: kotak GT tidak diubah oleh intervensi; baji dan warna rata menutup area yang sama, sehingga kontras hitam − rata "
          "mengendalikan oklusi. IK berasal dari bootstrap klaster per grup foto (resampel sama untuk semua model dan varian), "
          "dirata-rata antar seed."]
    (out / "BAJI_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "baji_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

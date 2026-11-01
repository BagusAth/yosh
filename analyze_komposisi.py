#!/usr/bin/env python3
"""
Komposisi sampel yang ditekankan EXP-1 (Pakem v1.4.2, Fase 5 A3, RQ2/RQ4)
=========================================================================
EXP-1 menarik citra latih dengan peluang P(k) = (1 - lambda)/N + lambda * D_k / sum(D), lambda = 0,70
(stage_8_2_build_difficulty_manifest.py; kolom p_sampling di train_image_difficulty_manifest.csv). Bobot relatif terhadap
sampling uniform: w_k = N * P(k). Rasio paparan sebuah kelompok = rata-rata w di kelompok itu (1 = sama dengan uniform).
Paparan objek = bobot citra tempat objek itu berada.

Untuk objek, rasio dinormalkan terhadap rata-rata bobot semua objek (= porsi EXP-1 / porsi uniform), karena citra
berobjek banyak berbobot lebih tinggi sehingga EXP-1 menarik sekitar 11% lebih banyak objek per epoch secara total.
Kelompok yang dibandingkan: kelas dominan citra, split asal Roboflow, citra dengan target kecil, desil skor kesulitan;
objek per kelas, ukuran, dan taksonomi OOF (TP, kotak meleset, salah kelas, tidak terdeteksi); objek yang OOF-nya
lebih yakin pada kelas lain (wrong_class, indikasi label bermasalah); artefak rotasi (baji hitam, pinggiran bergaris)
pada sampel 1.000 citra latih (seed 2031, dari A5).
IK 95% rasio paparan: bootstrap klaster per grup foto (1.000 iterasi).

Pemakaian: python analyze_komposisi.py
Keluaran: stage_v2_ablation/analysis/komposisi/ (KOMPOSISI_REPORT.md, komposisi_results.json)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_eleak_per_image import load_groups  # noqa: E402

KELAS = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
TAKS = {"True_Positive": "TP", "Localization_Error": "kotak meleset", "Classification_Confusion": "salah kelas",
        "Localization_and_Class_Error": "meleset + salah kelas", "False_Negative_Missed": "tidak terdeteksi"}
ITERS = 1000


def main() -> None:
    out = ROOT / "stage_v2_ablation/analysis/komposisi"
    out.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv")
    O = pd.read_csv(ROOT / "stage_v2_difficulty_20261028/train_object_difficulty.csv")
    N = len(M)
    M["w"] = M.p_sampling / M.p_sampling.sum() * N
    man = pd.read_csv(ROOT / "cassava-v7-stage-6/resplit_group_manifest.csv", encoding="utf-8-sig")
    man = man.assign(f=man.member_filenames.str.split(";")).explode("f")
    asal = man.set_index(man.f.str.strip()).original_splits
    M["asal"] = M.image_id.map(asal).fillna("?")
    dom = O.groupby("image_id").class_name.agg(lambda s: s.mode().iat[0])
    M["kelas"] = M.image_id.map(dom).fillna("(tanpa kotak)")
    M["n_obj"] = M.image_id.map(O.groupby("image_id").size()).fillna(0).astype(int)
    M["desil_D"] = pd.qcut(M.d_image.rank(method="first"), 10, labels=False) + 1
    O = O.merge(M[["image_id", "w"]], on="image_id")
    groups = load_groups(ROOT)
    gid = M.image_id.map(lambda n: groups.get(n, n)).to_numpy()
    uniq, inv = np.unique(gid, return_inverse=True)
    rng = np.random.default_rng(9100)
    boot_w = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv] for _ in range(ITERS)])
    img_idx = pd.Series(np.arange(N), index=M.image_id)

    def rasio_citra(mask: np.ndarray) -> dict:
        """Rata-rata w di kelompok (citra), dengan IK bootstrap klaster."""
        w = M.w.to_numpy()
        pt = float(w[mask].mean()) if mask.any() else np.nan
        bw = boot_w[:, mask]
        bs = (bw * w[mask]).sum(1) / np.maximum(bw.sum(1), 1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"n": int(mask.sum()), "porsi_uniform_pct": float(mask.mean() * 100),
                "porsi_exp1_pct": float(w[mask].sum() / w.sum() * 100), "rasio": pt, "ci95": [float(lo), float(hi)]}

    ii_all = img_idx[O.image_id].to_numpy()
    cnt_all = np.bincount(ii_all, minlength=N)
    wsum_all = np.bincount(ii_all, weights=O.w.to_numpy(), minlength=N)
    w_obj_all = float(O.w.mean())
    bs_all = (boot_w @ wsum_all) / (boot_w @ cnt_all)

    def rasio_objek(mask: np.ndarray) -> dict:
        """Rasio paparan relatif objek = rata-rata w kelompok / rata-rata w semua objek."""
        ii = img_idx[O.image_id[mask]].to_numpy()
        w = O.w.to_numpy()[mask]
        pt = float(w.mean() / w_obj_all) if len(w) else np.nan
        cnt = np.bincount(ii, minlength=N)          # objek kelompok per citra
        wsum = np.bincount(ii, weights=w, minlength=N)
        bs = (boot_w @ wsum) / np.maximum(boot_w @ cnt, 1) / bs_all
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"n": int(mask.sum()), "porsi_uniform_pct": float(mask.mean() * 100),
                "porsi_exp1_pct": float(w.sum() / O.w.sum() * 100), "rasio": pt, "ci95": [float(lo), float(hi)]}

    res: dict = {"tanggal": datetime.now().isoformat(timespec="seconds"), "N": N}
    w = M.w.to_numpy()
    res["sebaran_bobot"] = {"min": float(w.min()), "p10": float(np.percentile(w, 10)), "median": float(np.median(w)),
                            "p90": float(np.percentile(w, 90)), "maks": float(w.max()),
                            "ess_relatif": float(w.sum() ** 2 / (w ** 2).sum() / N),
                            "massa_10pct_teratas_pct": float(np.sort(w)[::-1][: N // 10].sum() / w.sum() * 100),
                            "massa_10pct_terbawah_pct": float(np.sort(w)[: N // 10].sum() / w.sum() * 100)}
    res["citra"] = {
        "kelas": {k: rasio_citra((M.kelas == k).to_numpy()) for k in KELAS},
        "asal": {a: rasio_citra((M.asal == a).to_numpy()) for a in ["train", "valid", "test"]},
        "target_kecil": {str(v): rasio_citra((M.has_target_small == v).to_numpy()) for v in [1, 0]},
        "jumlah_objek": {b: rasio_citra(m.to_numpy()) for b, m in
                         [("1", M.n_obj == 1), ("2-3", M.n_obj.between(2, 3)), ("4-6", M.n_obj.between(4, 6)), ("7+", M.n_obj >= 7)]},
        "desil_D": {int(d): rasio_citra((M.desil_D == d).to_numpy()) for d in range(1, 11)},
    }
    res["objek_per_epoch_relatif"] = w_obj_all
    res["objek"] = {
        "ukuran": {s: rasio_objek((O.size_cat == s).to_numpy()) for s in ["Small", "Medium", "Large"]},
        "kelas": {k: rasio_objek((O.class_name == k).to_numpy()) for k in KELAS},
        "taksonomi_oof": {TAKS[t]: rasio_objek((O.taxonomy == t).to_numpy()) for t in TAKS},
        "wrong_class": {str(v): rasio_objek((O.wrong_class == v).to_numpy()) for v in [1, 0]},
        "kelas_x_ukuran": {f"{k}|{s}": rasio_objek(((O.class_name == k) & (O.size_cat == s)).to_numpy())
                           for k in KELAS for s in ["Small", "Medium", "Large"]},
    }
    # artefak rotasi pada sampel 1.000 citra latih (A5); tanpa bootstrap klaster (sampel acak citra)
    P = pd.read_csv(ROOT / "stage_v2_ablation/analysis/pinggiran/pinggiran_train1000.csv")
    B = pd.read_csv(ROOT / "stage_v2_ablation/analysis/pinggiran/baji_train1000.csv")
    S = P.merge(B[["image_id", "baji_hitam"]], on="image_id").merge(M[["image_id", "w", "kelas", "d_image"]], on="image_id")
    S["garis"] = ((S.baji_hitam == 0) & (S.bergaris == 1)).astype(int)
    S["bersih"] = ((S.baji_hitam == 0) & (S.bergaris == 0)).astype(int)
    rb = np.random.default_rng(9200)
    art = {}
    for nama in ["baji_hitam", "garis", "bersih"]:
        m = S[nama].to_numpy() == 1
        ws = S.w.to_numpy()
        bs = []
        for _ in range(ITERS):
            ix = rb.integers(0, len(S), len(S))
            mm = m[ix]
            bs.append(ws[ix][mm].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        art[nama] = {"n": int(m.sum()), "porsi_uniform_pct": float(m.mean() * 100), "porsi_exp1_pct": float(ws[m].sum() / ws.sum() * 100),
                     "rasio": float(ws[m].mean()), "ci95": [float(lo), float(hi)], "d_image_median": float(S.d_image[m].median())}
    art["per_kelas"] = {k: {nm: float(S[(S.kelas == k) & (S[nm] == 1)].w.mean()) if ((S.kelas == k) & (S[nm] == 1)).any() else None
                            for nm in ["baji_hitam", "garis", "bersih"]} for k in KELAS}
    res["artefak_sampel1000"] = art

    # ---------------------------------------------------------------- laporan
    def baris(nama, r):
        return (f"| {nama} | {r['n']:,} | {r['porsi_uniform_pct']:.1f}% | {r['porsi_exp1_pct']:.1f}% | {r['rasio']:.3f} "
                f"({r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}) |")
    hdr = ["| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |", "| --- | --- | --- | --- | --- |"]
    sb = res["sebaran_bobot"]
    L = ["# Komposisi sampel yang ditekankan EXP-1 (Fase 5 A3)",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · {N:,} citra latih · bobot relatif w = N × P(k), λ = 0,70", "",
         f"Sebaran bobot: min {sb['min']:.2f}, p10 {sb['p10']:.2f}, median {sb['median']:.2f}, p90 {sb['p90']:.2f}, maks {sb['maks']:.2f}; "
         f"ESS relatif {sb['ess_relatif']:.3f}. 10% citra berbobot tertinggi menerima {sb['massa_10pct_teratas_pct']:.1f}% tarikan "
         f"(uniform 10%); 10% terendah {sb['massa_10pct_terbawah_pct']:.1f}%.", "",
         "Rasio paparan citra = rata-rata w di kelompok (1 = sama dengan uniform). Objek mewarisi bobot citranya; rasio objek "
         f"dinormalkan terhadap rata-rata bobot semua objek (= porsi EXP-1 / porsi uniform). Secara total EXP-1 menarik "
         f"{(w_obj_all - 1) * 100:.1f}% lebih banyak objek per epoch daripada uniform karena citra berobjek banyak berbobot lebih tinggi.", ""]
    for judul, d in [("Citra per kelas dominan", res["citra"]["kelas"]), ("Citra per split asal Roboflow", res["citra"]["asal"]),
                     ("Citra dengan target kecil (1) dan tanpa (0)", res["citra"]["target_kecil"]),
                     ("Citra per jumlah objek", res["citra"]["jumlah_objek"]),
                     ("Objek per ukuran", res["objek"]["ukuran"]), ("Objek per kelas", res["objek"]["kelas"]),
                     ("Objek per taksonomi OOF", res["objek"]["taksonomi_oof"]),
                     ("Objek dengan OOF lebih yakin pada kelas lain (1) dan lainnya (0)", res["objek"]["wrong_class"])]:
        L += [f"## {judul}", ""] + hdr + [baris(k, v) for k, v in d.items()] + [""]
    L += ["## Objek per kelas × ukuran (rasio paparan)", "", "| Kelas | Kecil | Sedang | Besar |", "| --- | --- | --- | --- |"]
    for k in KELAS:
        L.append(f"| {k} | " + " | ".join(f"{res['objek']['kelas_x_ukuran'][f'{k}|{s}']['rasio']:.3f} (n {res['objek']['kelas_x_ukuran'][f'{k}|{s}']['n']:,})"
                                         for s in ["Small", "Medium", "Large"]) + " |")
    L += ["", "## Artefak rotasi (sampel acak 1.000 citra latih, IK bootstrap citra)", "",
          "| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) | Median D_image |", "| --- | --- | --- | --- | --- | --- |"]
    for nm, lab in [("baji_hitam", "baji hitam"), ("garis", "bergaris, tanpa baji"), ("bersih", "tanpa artefak terdeteksi")]:
        r = art[nm]
        L.append(f"| {lab} | {r['n']} | {r['porsi_uniform_pct']:.1f}% | {r['porsi_exp1_pct']:.1f}% | {r['rasio']:.3f} "
                 f"({r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}) | {r['d_image_median']:.3f} |")
    L += ["", "Rasio per kelas (baji hitam / bergaris / bersih): " +
          "; ".join(f"{k} " + " / ".join("–" if art['per_kelas'][k][nm] is None else f"{art['per_kelas'][k][nm]:.2f}"
                                          for nm in ["baji_hitam", "garis", "bersih"]) for k in KELAS), "",
          "## Desil skor kesulitan D_image (rasio paparan)", "",
          " · ".join(f"D{d}: {res['citra']['desil_D'][d]['rasio']:.2f}" for d in range(1, 11))]
    (out / "KOMPOSISI_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "komposisi_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

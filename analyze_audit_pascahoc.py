"""Analisis PASCA-HOC audit label. Disetujui pengguna 4 November 2026 pukul 20.09; spesifikasi ini ditulis
sebelum skrip dijalankan.

Bukan bagian rencana terkunci analyze_audit_label.py. Hasil utama audit (22% vs 72%) sudah diketahui saat skrip
ini ditulis, sehingga SEMUA keluaran bersifat eksploratif pasca-hoc dan harus dilaporkan dengan label itu.
Tidak ada uji hipotesis konfirmatori di sini.

Data: audit_label_150/analysis/audit_per_citra.csv (isian + strata + d_image), _kunci/audit_key.csv,
stage_v2_difficulty_20261028/{train_image_difficulty_manifest,train_image_label_quality,train_object_difficulty}.csv,
cassava-v7-stage-6/resplit_group_manifest.csv (asal split Roboflow), sampel artefak 1.000 citra latih
(stage_v2_ablation/analysis/pinggiran/{baji_train1000,pinggiran_train1000}.csv), citra cassava-v7-runtime-mirror/train.

P1 Komposisi (deskriptif): strata tinggi (100), acak (50), populasi latih (8.980; artefak dari sampel 1.000 citra).
   kelas dominan (modus kelas kotak), jumlah kotak (median [Q1, Q3], % 1 kotak, % >= 7), ukuran objek OOF
   (% Small/Medium/Large; % citra dengan >= 1 objek kecil), taksonomi OOF per objek (% tiap kategori; % citra dengan
   >= 1 objek wrong_class), asal split Roboflow, artefak rotasi (baji hitam = uji_baji_hitam.ada_baji; pinggiran
   bergaris = cek_pinggiran_bergaris.nilai_citra, dihitung pada citra tanpa baji; artefak apa pun), sinyal kualitas
   label OOF (% citra n_suspect_missing/swap/badloc > 0; median label_quality).
P2 Y_label per kelas dominan di tiap strata + selisih tertimbang Mantel-Haenszel tinggi - acak (bobot n1*n2/(n1+n2),
   hanya kelas yang ada di kedua strata); IK 95% bootstrap bertingkat per strata, 2.000 kali.
P3 Hubungan D_image dengan Y_label di strata acak: AUC (IK bootstrap 2.000) dan proporsi per pita P4.
P4 Paparan EXP-1 terhadap citra bermasalah label tanpa asumsi monoton (estimator berbasis desain).
   Populasi latih dibagi: Puncak = D_image >= min D_image strata tinggi (diwakili strata tinggi);
   Sisa = lainnya (diwakili strata acak).
   E1 (utama pasca-hoc): Sisa dibagi 4 pita menurut kuartil D_image populasi Sisa; p_pita = proporsi Y_label citra
       acak di pita itu (pita tanpa citra acak memakai proporsi acak keseluruhan); Puncak p = proporsi strata tinggi.
       Porsi uniform U = sum_b (N_b/N) p_b; porsi EXP-1 W = sum_b P_b p_b, P_b = jumlah p_sampling ternormalisasi di pita b.
       Rasio paparan = W / U.
   E2: logistik Y_label ~ D_image pada citra acak saja untuk Sisa; Puncak = proporsi strata tinggi.
   E3: Sisa satu pita (p konstan = proporsi acak) + Puncak (tanpa gradien di Sisa).
   E1-S1: E1 tanpa baris "Kurang yakin".
   IK 95%: bootstrap bertingkat (citra tinggi dan acak diresampel terpisah), 2.000 kali, persentil; batas pita tetap.
   Penafsiran (ditetapkan sebelum dijalankan): rasio > 1 dengan batas bawah IK > 1 = EXP-1 menarik citra bermasalah
   label lebih sering daripada uniform; batas atas IK < 1 = lebih jarang; selain itu belum dapat ditentukan.

Keluaran: audit_label_150/analysis_pascahoc/ (PASCAHOC_REPORT.md, pascahoc_results.json, fitur_citra_audit.csv)

Pakai
  python analyze_audit_pascahoc.py fitur      # hitung artefak 150 citra audit (bisa diulang; melanjutkan)
  python analyze_audit_pascahoc.py analisis   # P1-P4 dan laporan
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUD = ROOT / "audit_label_150"
DD = ROOT / "stage_v2_difficulty_20261028"
PG = ROOT / "stage_v2_ablation" / "analysis" / "pinggiran"
OUT = AUD / "analysis_pascahoc"
IMG = ROOT / "cassava-v7-runtime-mirror" / "train" / "images"
N_BOOT, SEED = 2000, 20261104
TAKS = ["True_Positive", "False_Negative_Missed", "Localization_Error", "Classification_Confusion",
        "Localization_and_Class_Error"]
KELAS = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY", "(tanpa kotak)"]


def fitur() -> None:
    import cv2
    sys.path.insert(0, str(ROOT))
    from uji_baji_hitam import ada_baji
    from cek_pinggiran_bergaris import nilai_citra
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "fitur_citra_audit.csv"
    key = pd.read_csv(AUD / "_kunci" / "audit_key.csv")
    done = set(pd.read_csv(f).image_id) if f.exists() else set()
    rows = []
    for iid in key.image_id:
        if iid in done:
            continue
        p = IMG / iid
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rows.append({"image_id": iid, "baji_hitam": int(ada_baji(img)), "bergaris": int(nilai_citra(str(p))["bergaris"])})
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
            rows = []
    if rows:
        pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
    print("fitur:", len(pd.read_csv(f)), "citra")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logit_fit(x, y, iters=50):
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)]); y = np.asarray(y, float); b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(2), X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def pct(mask):
    mask = np.asarray(mask, bool)
    return {"k": int(mask.sum()), "n": int(len(mask)), "pct": float(100 * mask.mean()) if len(mask) else float("nan")}


def komposisi(ids, obj, lq, asal, art=None):
    o = obj[obj.image_id.isin(ids)]
    q = lq.set_index("image_id").loc[list(ids)]
    dom = o.groupby("image_id").class_name.agg(lambda s: s.mode().iat[0]).reindex(list(ids)).fillna("(tanpa kotak)")
    ngt = q.n_gt.to_numpy()
    r = {"n_citra": len(ids), "n_objek": int(len(o)),
         "kelas_dominan": {c: pct(dom.to_numpy() == c) for c in KELAS},
         "jumlah_kotak": {"median": float(np.median(ngt)), "q1": float(np.percentile(ngt, 25)), "q3": float(np.percentile(ngt, 75)),
                          "satu_kotak": pct(ngt == 1), "tujuh_atau_lebih": pct(ngt >= 7)},
         "ukuran_objek": {s: pct(o.size_cat.to_numpy() == s) for s in ("Small", "Medium", "Large")},
         "citra_ada_objek_kecil": pct(pd.Series(list(ids)).isin(o[o.size_cat == "Small"].image_id).to_numpy()),
         "taksonomi_objek": {t: pct(o.taxonomy.to_numpy() == t) for t in TAKS},
         "citra_ada_wrong_class": pct(pd.Series(list(ids)).isin(o[o.wrong_class == 1].image_id).to_numpy()),
         "asal_roboflow": {a: pct(asal.reindex(list(ids)).fillna("?").to_numpy() == a) for a in sorted(asal.reindex(list(ids)).fillna("?").unique())},
         "sinyal_lq": {"missing>0": pct(q.n_suspect_missing.to_numpy() > 0), "swap>0": pct(q.n_suspect_swap.to_numpy() > 0),
                       "badloc>0": pct(q.n_suspect_badloc.to_numpy() > 0), "median_label_quality": float(q.label_quality.median())}}
    if art is not None:
        a = art.set_index("image_id").reindex(list(ids)).dropna()
        tb = a[a.baji_hitam == 0]
        r["artefak"] = {"n": int(len(a)), "baji_hitam": pct(a.baji_hitam == 1), "bergaris_tanpa_baji": pct(tb.bergaris == 1),
                        "artefak_apa_pun": pct((a.baji_hitam == 1) | (a.bergaris == 1))}
    return r


def mh(d):
    num = den = 0.0
    for c, g in d.groupby("kelas"):
        t, a = g[g.strata == "tinggi"].y_label, g[g.strata == "acak"].y_label
        if len(t) and len(a):
            w = len(t) * len(a) / (len(t) + len(a)); num += w * (t.mean() - a.mean()); den += w
    return num / den if den else float("nan")


def paparan(d, man, top_min, varian):
    x = man.d_image.to_numpy(float); w = man.p_sampling.to_numpy(float); w = w / w.sum(); N = len(x)
    puncak = x >= top_min
    sisa_x = x[~puncak]
    q = np.quantile(sisa_x, [0.25, 0.5, 0.75])
    pita = np.where(puncak, 4, np.searchsorted(q, x, side="right"))  # 0..3 sisa, 4 puncak
    t, a = d[d.strata == "tinggi"], d[d.strata == "acak"]
    p_top = t.y_label.mean()
    if varian == "E1":
        pa = np.searchsorted(q, a.d_image.to_numpy(float), side="right")
        pb = np.array([a.y_label.to_numpy()[pa == b].mean() if (pa == b).any() else a.y_label.mean() for b in range(4)] + [p_top])
        p = pb[pita]
    elif varian == "E2":
        b = logit_fit(a.d_image, a.y_label); p = 1 / (1 + np.exp(-np.clip(b[0] + b[1] * x, -30, 30))); p[puncak] = p_top
    elif varian == "E3":
        p = np.where(puncak, p_top, a.y_label.mean())
    U, W = float(p.mean()), float((w * p).sum())
    return W / U, U, W, pita, q


def analisis() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(AUD / "analysis" / "audit_per_citra.csv")
    man = pd.read_csv(DD / "train_image_difficulty_manifest.csv")
    lq = pd.read_csv(DD / "train_image_label_quality.csv")
    obj = pd.read_csv(DD / "train_object_difficulty.csv")
    M = pd.read_csv(ROOT / "cassava-v7-stage-6" / "resplit_group_manifest.csv", encoding="utf-8-sig")
    M = M.assign(f=M.member_filenames.str.split(";")).explode("f")
    asal = pd.Series(M.original_splits.to_numpy(), index=M.f.str.strip().to_numpy())
    asal = asal[~asal.index.duplicated()]
    art_a = pd.read_csv(OUT / "fitur_citra_audit.csv")
    art_p = pd.read_csv(PG / "pinggiran_train1000.csv")[["image_id", "bergaris"]].merge(
        pd.read_csv(PG / "baji_train1000.csv")[["image_id", "baji_hitam"]], on="image_id")
    assert len(art_a) == 150, "jalankan 'fitur' dulu sampai 150 citra"
    dom = obj.groupby("image_id").class_name.agg(lambda s: s.mode().iat[0])
    d["kelas"] = d.image_id.map(dom).fillna("(tanpa kotak)")
    res = {"waktu": dt.datetime.now().isoformat(timespec="seconds"), "label": "PASCA-HOC (eksploratif)"}
    T, A = d[d.strata == "tinggi"], d[d.strata == "acak"]
    res["P1"] = {"tinggi": komposisi(T.image_id, obj, lq, asal, art_a), "acak": komposisi(A.image_id, obj, lq, asal, art_a),
                 "populasi": komposisi(man.image_id, obj, lq, asal, art_p)}
    rng = np.random.default_rng(SEED)
    # P2
    per = {}
    for c in KELAS:
        g = d[d.kelas == c]
        per[c] = {s: {"k": int(g[g.strata == s].y_label.sum()), "n": int((g.strata == s).sum())} for s in ("tinggi", "acak")}
    bs = []
    for _ in range(N_BOOT):
        s = pd.concat([T.iloc[rng.integers(0, len(T), len(T))], A.iloc[rng.integers(0, len(A), len(A))]])
        bs.append(mh(s))
    res["P2"] = {"per_kelas": per, "selisih_MH": mh(d), "ik": ci(bs)}
    # P3
    top_min = float(T.d_image.min())
    bs = []
    for _ in range(N_BOOT):
        s = A.iloc[rng.integers(0, len(A), len(A))]
        bs.append(auc(s.d_image, s.y_label))
    res["P3"] = {"auc_acak": auc(A.d_image, A.y_label), "ik": ci(bs)}
    # P4
    out4 = {"puncak_min_d": top_min}
    r1, U1, W1, pita, q = paparan(d, man, top_min, "E1")
    w = man.p_sampling.to_numpy(float); w = w / w.sum()
    pa = np.searchsorted(q, A.d_image.to_numpy(float), side="right")
    out4["pita"] = [{"pita": b, "batas": (["min"] + [round(float(v), 4) for v in q] + [round(top_min, 4)])[b:b + 2] if b < 4 else [round(top_min, 4), "maks"],
                     "n_populasi": int((pita == b).sum()), "porsi_citra": float((pita == b).mean()), "porsi_tarikan_exp1": float(w[pita == b].sum()),
                     "n_audit": int((pa == b).sum()) if b < 4 else int(len(T)),
                     "k_audit": int(A.y_label.to_numpy()[pa == b].sum()) if b < 4 else int(T.y_label.sum())} for b in range(5)]
    for v in ("E1", "E2", "E3", "E1-S1"):
        dd = d[d.yakin != "Kurang yakin"] if v == "E1-S1" else d
        vv = "E1" if v == "E1-S1" else v
        r, U, W, _, _ = paparan(dd, man, top_min, vv)
        Tv, Av = dd[dd.strata == "tinggi"], dd[dd.strata == "acak"]
        bs = []
        for _ in range(N_BOOT):
            s = pd.concat([Tv.iloc[rng.integers(0, len(Tv), len(Tv))], Av.iloc[rng.integers(0, len(Av), len(Av))]])
            bs.append(paparan(s, man, top_min, vv)[0] if s[s.strata == "acak"].y_label.nunique() > 1 or vv != "E2" else np.nan)
        lo, hi = ci(bs)
        out4[v] = {"rasio": r, "ik": [lo, hi], "porsi_uniform": U, "porsi_exp1": W, "n_tinggi": int(len(Tv)), "n_acak": int(len(Av)),
                   "tafsiran": "lebih sering" if lo > 1 else ("lebih jarang" if hi < 1 else "belum dapat ditentukan")}
    res["P4"] = out4
    (OUT / "pascahoc_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    laporan(res)
    print("Selesai ->", OUT / "PASCAHOC_REPORT.md")


def laporan(r):
    f1 = lambda x: f"{x:.1f}".replace(".", ",")
    f2 = lambda x: f"{x:.2f}".replace(".", ",")
    f3 = lambda x: f"{x:.3f}".replace(".", ",")
    P = r["P1"]; G = ("tinggi", "acak", "populasi")
    L = ["# Analisis pasca-hoc audit label (EKSPLORATIF, bukan pra-registrasi)", "",
         f"Dibuat {r['waktu']}. Disetujui pengguna 4 November 2026 20.09; spesifikasi di docstring analyze_audit_pascahoc.py "
         "ditulis sebelum dijalankan, setelah hasil utama audit diketahui.", "",
         "## P1 Komposisi strata", "", "| Ciri | Tinggi | Acak | Populasi latih |", "|---|---|---|---|",
         f"| Citra (objek) | {P['tinggi']['n_citra']} ({P['tinggi']['n_objek']}) | {P['acak']['n_citra']} ({P['acak']['n_objek']}) | {P['populasi']['n_citra']} ({P['populasi']['n_objek']}) |"]
    for c in KELAS:
        L.append(f"| Kelas dominan {c} | " + " | ".join(f1(P[g]["kelas_dominan"][c]["pct"]) + "%" for g in G) + " |")
    L.append("| Jumlah kotak, median [Q1, Q3] | " + " | ".join(f"{P[g]['jumlah_kotak']['median']:.0f} [{P[g]['jumlah_kotak']['q1']:.0f}, {P[g]['jumlah_kotak']['q3']:.0f}]" for g in G) + " |")
    L.append("| Citra 1 kotak | " + " | ".join(f1(P[g]["jumlah_kotak"]["satu_kotak"]["pct"]) + "%" for g in G) + " |")
    L.append("| Citra >= 7 kotak | " + " | ".join(f1(P[g]["jumlah_kotak"]["tujuh_atau_lebih"]["pct"]) + "%" for g in G) + " |")
    for s in ("Small", "Medium", "Large"):
        L.append(f"| Objek {s} (% objek) | " + " | ".join(f1(P[g]["ukuran_objek"][s]["pct"]) + "%" for g in G) + " |")
    L.append("| Citra dengan >= 1 objek kecil | " + " | ".join(f1(P[g]["citra_ada_objek_kecil"]["pct"]) + "%" for g in G) + " |")
    for t in TAKS:
        L.append(f"| OOF {t} (% objek) | " + " | ".join(f1(P[g]["taksonomi_objek"][t]["pct"]) + "%" for g in G) + " |")
    L.append("| Citra dengan >= 1 objek OOF salah kelas | " + " | ".join(f1(P[g]["citra_ada_wrong_class"]["pct"]) + "%" for g in G) + " |")
    asal = sorted(set().union(*[P[g]["asal_roboflow"].keys() for g in G]))
    for a in asal:
        L.append(f"| Asal Roboflow {a} | " + " | ".join(f1(P[g]["asal_roboflow"].get(a, {"pct": 0})["pct"]) + "%" for g in G) + " |")
    for k, nm in (("baji_hitam", "Baji hitam"), ("bergaris_tanpa_baji", "Pinggiran bergaris (citra tanpa baji)"), ("artefak_apa_pun", "Artefak rotasi apa pun")):
        L.append(f"| {nm} | " + " | ".join(f1(P[g]["artefak"][k]["pct"]) + "%" for g in G) + " |")
    for k in ("missing>0", "swap>0", "badloc>0"):
        L.append(f"| Sinyal OOF {k} | " + " | ".join(f1(P[g]["sinyal_lq"][k]["pct"]) + "%" for g in G) + " |")
    L.append("| Median label_quality | " + " | ".join(f3(P[g]["sinyal_lq"]["median_label_quality"]) for g in G) + " |")
    L += ["", f"Artefak populasi dari sampel acak {P['populasi']['artefak']['n']} citra latih (Fase 5).", "",
          "## P2 Y_label per kelas dominan", "", "| Kelas | Tinggi | Acak |", "|---|---|---|"]
    for c, v in r["P2"]["per_kelas"].items():
        L.append(f"| {c} | {v['tinggi']['k']}/{v['tinggi']['n']} | {v['acak']['k']}/{v['acak']['n']} |")
    L += ["", f"Selisih tertimbang Mantel-Haenszel (tinggi - acak): {f1(100 * r['P2']['selisih_MH'])} poin "
          f"(IK bootstrap {f1(100 * r['P2']['ik'][0])} s.d. {f1(100 * r['P2']['ik'][1])}).", "",
          "## P3 D_image dan Y_label di strata acak", "",
          f"AUC {f3(r['P3']['auc_acak'])} (IK {f3(r['P3']['ik'][0])} s.d. {f3(r['P3']['ik'][1])}).", "",
          "## P4 Paparan EXP-1 tanpa asumsi monoton", "", "| Pita | Batas D_image | Citra latih | Porsi citra | Porsi tarikan EXP-1 | Y_label audit |",
          "|---|---|---|---|---|---|"]
    for p in r["P4"]["pita"]:
        L.append(f"| {p['pita']} | {p['batas'][0]} s.d. {p['batas'][1]} | {p['n_populasi']} | {f1(100 * p['porsi_citra'])}% | "
                 f"{f1(100 * p['porsi_tarikan_exp1'])}% | {p['k_audit']}/{p['n_audit']} |")
    L += ["", "| Estimator | Rasio paparan (IK 95%) | Porsi uniform | Porsi EXP-1 | Tafsiran |", "|---|---|---|---|---|"]
    for v in ("E1", "E2", "E3", "E1-S1"):
        e = r["P4"][v]
        L.append(f"| {v} | {f3(e['rasio'])} ({f3(e['ik'][0])} s.d. {f3(e['ik'][1])}) | {f1(100 * e['porsi_uniform'])}% | "
                 f"{f1(100 * e['porsi_exp1'])}% | {e['tafsiran']} |")
    L += ["", "E1 = 4 pita kuartil Sisa + Puncak (utama pasca-hoc); E2 = logistik pada citra acak untuk Sisa; "
          "E3 = Sisa tanpa gradien; E1-S1 = E1 tanpa \"Kurang yakin\".", "",
          "Semua hasil di laporan ini pasca-hoc dan eksploratif; tidak menggantikan analisis yang dipra-registrasi."]
    (OUT / "PASCAHOC_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"fitur": fitur, "analisis": analisis}[sys.argv[1]]()

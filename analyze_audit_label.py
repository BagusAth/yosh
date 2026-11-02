"""Analisis audit label manual (lembar v2) — PRA-REGISTRASI, dikunci 2 November 2026.

Skrip ini ditulis dan diuji dengan data tiruan SEBELUM kunci strata dibuka dan sebelum
isian audit dilihat. Hanya dijalankan pada data asli setelah semua 150 citra dinilai.

Data
  audit_label_150/LEMBAR_AUDIT_LABEL.xlsx  (sheet "Audit", rater 1, 150 citra)
  audit_label_150/LEMBAR_AUDIT_RATER2.xlsx (sheet "Rater2", opsional, 30 citra)
  audit_label_150/_kunci/audit_key.csv     (strata tinggi/acak, d_image, skor otomatis)
  stage_v2_difficulty_20261028/train_image_label_quality.csv  (skor kualitas label OOF)
  stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv (d_image, p_sampling EXP-1)

Definisi
  Y_label  = 1 bila Kategori utama ATAU tambahan salah satu dari empat masalah label:
             Daun tanpa kotak, Kotak bersarang/ganda, Objek bukan daun, Kotak tidak presisi.
  Y_hilang = 1 bila kolom "Daun tanpa kotak (perkiraan)" bernilai 1-2, 3-5, atau >5
             ("Ragu" dan kosong dikeluarkan dari analisis Y_hilang).
  Y_presisi = 1 bila "Kotak presisi?" bernilai Sebagian atau Tidak ("Ragu" dikeluarkan).

Analisis UTAMA (satu-satunya uji konfirmatori)
  Selisih proporsi Y_label: strata tinggi (100) − acak (50).
  IK 95% Newcombe (hybrid score, metode 10); uji eksak Fisher dua sisi sebagai pendamping.
  Verdict: "terdukung" bila selisih > 0 dan batas bawah IK > 0; "berlawanan" bila batas atas
  IK < 0; selain itu "tidak terbukti berbeda".

Sensitivitas (pra-registrasi)
  S1: analisis utama tanpa baris "Kurang yakin".
  S2: Y_label hanya dari Kategori utama.

Sekunder / eksploratif (tanpa klaim konfirmatori, tanpa koreksi multiplisitas)
  - Proporsi per kategori (utama atau tambahan) per strata + Fisher.
  - Y_hilang dan Y_presisi per strata (Newcombe).
  - Sebaran "Daun tanpa kotak" per strata.
  - Kesepakatan antar penilai (bila lembar rater 2 lengkap, 30 citra): persentase setuju dan
    kappa Cohen untuk Y_label dan Kategori utama, kappa berbobot linear untuk Daun tanpa kotak
    (0 < 1-2 < 3-5 < >5, "Ragu" dikeluarkan); IK 95% bootstrap 2.000 kali.
  - Validasi skor otomatis OOF terhadap penilaian manusia (AUC, IK bootstrap 2.000):
      1 − label_quality vs Y_label; missing_score vs Y_hilang; badloc_score vs Y_presisi.
    Skor dianggap valid bila AUC >= 0,70 dan batas bawah IK > 0,50. AUC pada strata acak
    saja dilaporkan sebagai sensitivitas (seleksi strata ikut berkorelasi dengan skor).
  - Paparan EXP-1: regresi logistik P(Y_label = 1 | d_image) pada 150 citra (seleksi hanya
    bergantung pada d_image, jadi model bersyarat tidak bias), diprediksi ke seluruh citra
    latih; rasio paparan = sum(p_sampling * p_hat) / mean(p_hat). IK 95% bootstrap bertingkat
    per strata, 2.000 kali. Pendamping: porsi tarikan EXP-1 vs uniform untuk citra dengan
    label_quality < 0,5 (ditafsirkan hanya bila skor otomatis valid).

Keluaran: audit_label_150/analysis/AUDIT_REPORT.md, audit_results.json, audit_per_citra.csv

Pakai
  python analyze_audit_label.py                 # data asli (menolak bila lembar belum lengkap)
  python analyze_audit_label.py --paksa         # jalankan walau ada baris kosong (tidak untuk laporan)
  python analyze_audit_label.py --uji-tiruan DIR [--efek 0.30] [--seed 1]
                                                # buat data tiruan di DIR lalu analisis (tidak menyentuh data asli)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook

LABEL_CATS = ["Daun tanpa kotak", "Kotak bersarang/ganda", "Objek bukan daun", "Kotak tidak presisi"]
OTHER_CATS = ["Artefak citra", "Citra padat/sulit", "Tidak ada masalah terlihat", "Lainnya"]
ALL_CATS = LABEL_CATS + OTHER_CATS
TANPA_ORD = {"0": 0, "1-2": 1, "3-5": 2, ">5": 3}
TANPA_VALS = list(TANPA_ORD) + ["Ragu"]
PRESISI_VALS = ["Ya", "Sebagian", "Tidak", "Ragu"]
YAKIN_VALS = ["Yakin", "Kurang yakin"]
H_KODE, H_UTAMA, H_TAMBAH = "Kode", "Kategori utama", "Kategori tambahan (opsional)"
H_TANPA, H_YAKIN, H_PRESISI = "Daun tanpa kotak (perkiraan)", "Keyakinan penilaian", "Kotak presisi?"
N_BOOT = 2000
SEED = 2026


# ----------------------------------------------------------------------------- pembacaan
def norm_tanpa(v):
    """Normalisasi isian 'Daun tanpa kotak'. Excel bisa mengubah 1-2 / 3-5 menjadi tanggal."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, (dt.datetime, dt.date)):
        md = {(v.month, v.day)}
        if md & {(1, 2), (2, 1)}:
            return "1-2"
        if md & {(3, 5), (5, 3)}:
            return "3-5"
        return f"?{v}"
    if isinstance(v, (int, float)):
        return "0" if float(v) == 0 else f"?{v}"
    s = str(v).strip()
    return s if s in TANPA_VALS else f"?{s}"


def norm_text(v, allowed):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    s = str(v).strip()
    return s if s in allowed else f"?{s}"


def baca_lembar(path: Path, sheet: str) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    idx = {h: i for i, h in enumerate(head)}
    for h in (H_KODE, H_UTAMA, H_TAMBAH, H_TANPA, H_YAKIN, H_PRESISI):
        if h not in idx:
            raise SystemExit(f"{path.name}: kolom '{h}' tidak ditemukan")
    out = []
    for r in rows[1:]:
        if not r or r[idx[H_KODE]] in (None, ""):
            continue
        out.append({
            "kode": str(r[idx[H_KODE]]).strip(),
            "utama": norm_text(r[idx[H_UTAMA]], ALL_CATS),
            "tambahan": norm_text(r[idx[H_TAMBAH]], ALL_CATS),
            "tanpa": norm_tanpa(r[idx[H_TANPA]]),
            "yakin": norm_text(r[idx[H_YAKIN]], YAKIN_VALS),
            "presisi": norm_text(r[idx[H_PRESISI]], PRESISI_VALS),
        })
    return pd.DataFrame(out)


def turunan(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["y_label"] = (d.utama.isin(LABEL_CATS) | d.tambahan.isin(LABEL_CATS)).astype(int)
    d["y_label_utama"] = d.utama.isin(LABEL_CATS).astype(int)
    d["y_hilang"] = d.tanpa.map(lambda v: (1 if TANPA_ORD[v] > 0 else 0) if v in TANPA_ORD else np.nan)
    d["y_presisi"] = d.presisi.map(lambda v: {"Ya": 0, "Sebagian": 1, "Tidak": 1}.get(v, np.nan))
    return d


# ----------------------------------------------------------------------------- statistik
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    K, N = k1 + k2, n1 + n2

    def pmf(x):
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)

    lo, hi = max(0, K - n2), min(K, n1)
    p_obs = pmf(k1)
    return float(min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))))


def kappa(a, b, cats, weighted: bool = False) -> float:
    a, b = list(a), list(b)
    m = len(cats)
    ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((m, m))
    for x, y in zip(a, b):
        O[ix[x], ix[y]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.array([[abs(i - j) / (m - 1) if weighted else float(i != j) for j in range(m)] for i in range(m)])
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def auc(score, y) -> float:
    s, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logit_fit(x, y, iters: int = 50):
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    y = np.asarray(y, float)
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def logit_pred(b, x):
    return 1 / (1 + np.exp(-np.clip(b[0] + b[1] * np.asarray(x, float), -30, 30)))


def boot_ci(vals) -> tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"), float("nan"))


# ----------------------------------------------------------------------------- analisis
def banding(d: pd.DataFrame, col: str) -> dict:
    t = d[d.strata == "tinggi"][col].dropna()
    a = d[d.strata == "acak"][col].dropna()
    k1, n1, k2, n2 = int(t.sum()), len(t), int(a.sum()), len(a)
    if n1 == 0 or n2 == 0:
        return {"n_tinggi": n1, "n_acak": n2}
    diff, lo, hi = newcombe(k1, n1, k2, n2)
    return {"k_tinggi": k1, "n_tinggi": n1, "p_tinggi": k1 / n1, "ik_tinggi": wilson(k1, n1),
            "k_acak": k2, "n_acak": n2, "p_acak": k2 / n2, "ik_acak": wilson(k2, n2),
            "selisih": diff, "ik_selisih": (lo, hi), "p_fisher": fisher_two_sided(k1, n1, k2, n2),
            "verdict": "terdukung" if lo > 0 else ("berlawanan" if hi < 0 else "tidak terbukti berbeda")}


def kesepakatan(d1: pd.DataFrame, d2: pd.DataFrame) -> dict:
    m = d1.merge(d2, on="kode", suffixes=("_1", "_2"))
    m = m[m.utama_1.isin(ALL_CATS) & m.utama_2.isin(ALL_CATS)]
    if len(m) < 30:
        return {"tersedia": False, "n": int(len(m))}
    rng = np.random.default_rng(SEED)
    res = {"tersedia": True, "n": int(len(m))}
    specs = [("y_label", [0, 1], False, m), ("utama", ALL_CATS, False, m)]
    mt = m[m.tanpa_1.isin(TANPA_ORD) & m.tanpa_2.isin(TANPA_ORD)]
    specs.append(("tanpa", list(TANPA_ORD), True, mt))
    for col, cats, w, mm in specs:
        a, b = mm[f"{col}_1"].tolist(), mm[f"{col}_2"].tolist()
        k = kappa(a, b, cats, w)
        bs = []
        for _ in range(N_BOOT):
            ii = rng.integers(0, len(a), len(a))
            bs.append(kappa([a[i] for i in ii], [b[i] for i in ii], cats, w))
        res[col] = {"n": len(a), "setuju": float(np.mean([x == y for x, y in zip(a, b)])),
                    "kappa": k, "ik": boot_ci(bs), "berbobot": w}
    return res


def validasi_skor(d: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for nama, skor, ycol in (("1 - label_quality vs Y_label", 1 - d.label_quality, "y_label"),
                             ("missing_score vs Y_hilang", d.missing_score, "y_hilang"),
                             ("badloc_score vs Y_presisi", d.badloc_score, "y_presisi")):
        for sub, mask in (("semua", np.ones(len(d), bool)), ("acak", (d.strata == "acak").to_numpy())):
            ok = mask & d[ycol].notna().to_numpy() & np.isfinite(skor.to_numpy())
            s, y = skor.to_numpy()[ok], d[ycol].to_numpy()[ok].astype(int)
            a = auc(s, y)
            bs = []
            for _ in range(N_BOOT):
                ii = rng.integers(0, len(s), len(s))
                bs.append(auc(s[ii], y[ii]))
            lo, hi = boot_ci(bs)
            out[f"{nama} [{sub}]"] = {"n": int(ok.sum()), "n_pos": int(y.sum()), "auc": a, "ik": (lo, hi),
                                      "valid": bool(np.isfinite(a) and a >= 0.70 and lo > 0.50)}
    return out


def paparan(d: pd.DataFrame, man: pd.DataFrame, lq_all: pd.DataFrame, skor_valid: bool) -> dict:
    rng = np.random.default_rng(SEED + 2)
    N = len(man)
    xs, ws = man.d_image.to_numpy(float), man.p_sampling.to_numpy(float)
    ws = ws / ws.sum()

    def rasio(sub):
        b = logit_fit(sub.d_image, sub.y_label)
        p = logit_pred(b, xs)
        return float((ws * p).sum() / p.mean()), b, float(p.mean()), float((ws * p).sum())

    r, b, pu, pe = rasio(d)
    t, a = d[d.strata == "tinggi"], d[d.strata == "acak"]
    bs = []
    for _ in range(N_BOOT):
        s = pd.concat([t.iloc[rng.integers(0, len(t), len(t))], a.iloc[rng.integers(0, len(a), len(a))]])
        bs.append(rasio(s)[0] if s.y_label.nunique() > 1 else np.nan)
    m = man.merge(lq_all[["image_id", "label_quality"]], on="image_id", how="left")
    flag = (m.label_quality < 0.5).to_numpy()
    w2 = m.p_sampling.to_numpy(float) / m.p_sampling.sum()
    return {"n_citra_latih": int(N), "logit_b0": float(b[0]), "logit_b1": float(b[1]),
            "prev_uniform": pu, "prev_exp1": pe, "rasio_paparan": r, "ik_rasio": boot_ci(bs),
            "flag_lq_lt_0_5": {"n": int(flag.sum()), "porsi_uniform": float(flag.mean()),
                               "porsi_exp1": float(w2[flag].sum()),
                               "rasio": float(w2[flag].sum() / flag.mean()) if flag.any() else float("nan"),
                               "ditafsirkan": skor_valid}}


# ----------------------------------------------------------------------------- laporan
def fp(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}%"


def fik(t, pct=True):
    lo, hi = t
    return f"{fp(lo)} s.d. {fp(hi)}" if pct else f"{lo:.2f} s.d. {hi:.2f}"


def baris_banding(nama, r):
    if "selisih" not in r:
        return f"| {nama} | {r.get('n_tinggi', 0)} | – | {r.get('n_acak', 0)} | – | – | – | – |"
    return (f"| {nama} | {r['k_tinggi']}/{r['n_tinggi']} ({fp(r['p_tinggi'])}) | {fik(r['ik_tinggi'])} | "
            f"{r['k_acak']}/{r['n_acak']} ({fp(r['p_acak'])}) | {fik(r['ik_acak'])} | "
            f"{100 * r['selisih']:+.1f} poin ({100 * r['ik_selisih'][0]:+.1f} s.d. {100 * r['ik_selisih'][1]:+.1f}) | "
            f"{r['p_fisher']:.3f} | {r['verdict']} |")


def tulis_laporan(res: dict, out: Path, tiruan: bool) -> None:
    L = ["# Laporan analisis audit label (lembar v2)", ""]
    if tiruan:
        L += ["**DATA TIRUAN — hanya untuk menguji skrip, bukan hasil penelitian.**", ""]
    L += [f"Dibuat {res['waktu']}. Rencana analisis dikunci 2 November 2026 (docstring skrip).", "",
          "## Kelengkapan", "",
          f"- Rater 1: {res['kelengkapan']['n_baris']} baris, {res['kelengkapan']['n_lengkap']} dengan Kategori utama, "
          f"{res['kelengkapan']['n_tidak_valid']} isian tidak dikenal, {res['kelengkapan']['n_kurang_yakin']} \"Kurang yakin\".",
          f"- Strata: tinggi {res['kelengkapan']['n_tinggi']}, acak {res['kelengkapan']['n_acak']}.", "",
          "## Analisis utama", "",
          "| Luaran | Tinggi | IK 95% | Acak | IK 95% | Selisih (IK 95% Newcombe) | p Fisher | Verdict |",
          "|---|---|---|---|---|---|---|---|",
          baris_banding("Y_label (utama)", res["utama"]),
          baris_banding("S1: tanpa \"Kurang yakin\"", res["s1"]),
          baris_banding("S2: Kategori utama saja", res["s2"]),
          baris_banding("Y_hilang (sekunder)", res["y_hilang"]),
          baris_banding("Y_presisi (sekunder)", res["y_presisi"]), "",
          "## Kategori (utama atau tambahan), eksploratif", "",
          "| Kategori | Tinggi | Acak | Selisih (IK 95%) | p Fisher |", "|---|---|---|---|---|"]
    for c, r in res["kategori"].items():
        if "selisih" in r:
            L.append(f"| {c} | {r['k_tinggi']} ({fp(r['p_tinggi'])}) | {r['k_acak']} ({fp(r['p_acak'])}) | "
                     f"{100 * r['selisih']:+.1f} ({100 * r['ik_selisih'][0]:+.1f} s.d. {100 * r['ik_selisih'][1]:+.1f}) | "
                     f"{r['p_fisher']:.3f} |")
    L += ["", "## Sebaran \"Daun tanpa kotak\"", "", "| Nilai | Tinggi | Acak |", "|---|---|---|"]
    for v, (a, b) in res["sebaran_tanpa"].items():
        L.append(f"| {v} | {a} | {b} |")
    k = res["kesepakatan"]
    L += ["", "## Kesepakatan antar penilai", ""]
    if not k.get("tersedia"):
        L.append(f"Tidak tersedia (baris rater 2 yang lengkap: {k.get('n', 0)} dari 30).")
    else:
        L += ["| Ukuran | n | Setuju | Kappa (IK 95%) |", "|---|---|---|---|"]
        for col, nama in (("y_label", "Y_label"), ("utama", "Kategori utama"), ("tanpa", "Daun tanpa kotak (berbobot)")):
            r = k[col]
            L.append(f"| {nama} | {r['n']} | {fp(r['setuju'])} | {r['kappa']:.2f} ({fik(r['ik'], False)}) |")
    L += ["", "## Validasi skor otomatis OOF", "", "| Skor vs luaran [sampel] | n | positif | AUC (IK 95%) | Valid |",
          "|---|---|---|---|---|"]
    for nama, r in res["validasi_skor"].items():
        L.append(f"| {nama} | {r['n']} | {r['n_pos']} | {r['auc']:.2f} ({fik(r['ik'], False)}) | {'ya' if r['valid'] else 'tidak'} |")
    p = res["paparan"]
    f = p["flag_lq_lt_0_5"]
    L += ["", "## Paparan EXP-1 terhadap citra bermasalah label (eksploratif)", "",
          f"- Logistik P(Y_label | d_image): b0 = {p['logit_b0']:.2f}, b1 = {p['logit_b1']:.2f}.",
          f"- Prevalensi terprediksi di {p['n_citra_latih']} citra latih: uniform {fp(p['prev_uniform'])}, "
          f"tertimbang EXP-1 {fp(p['prev_exp1'])}; rasio paparan {p['rasio_paparan']:.3f} "
          f"(IK 95% {p['ik_rasio'][0]:.3f} s.d. {p['ik_rasio'][1]:.3f}).",
          f"- Citra dengan label_quality < 0,5: {f['n']} citra, porsi uniform {fp(f['porsi_uniform'])}, porsi tarikan EXP-1 "
          f"{fp(f['porsi_exp1'])} (rasio {f['rasio']:.2f}); "
          + ("ditafsirkan karena skor otomatis valid." if f["ditafsirkan"] else "TIDAK ditafsirkan karena skor otomatis tidak valid."),
          "", "## Catatan", "",
          "- Benar-salahnya kelas tidak dinilai (penilai bukan ahli penyakit tanaman); keterbatasan penelitian.",
          "- Hanya analisis utama yang konfirmatori; bagian lain eksploratif tanpa koreksi multiplisitas."]
    (out / "AUDIT_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- utama
def jalankan(p: dict, paksa: bool, tiruan: bool) -> dict:
    r1 = turunan(baca_lembar(p["lembar1"], "Audit"))
    key = pd.read_csv(p["kunci"])
    if len(r1) != len(key) or set(r1.kode) != set(key.kode):
        raise SystemExit("Kode di lembar rater 1 tidak cocok dengan kunci.")
    n_lengkap = int(r1.utama.isin(ALL_CATS).sum())
    bad = int(sum(r1[c].astype(str).str.startswith("?").sum() for c in ("utama", "tambahan", "tanpa", "yakin", "presisi")))
    if (n_lengkap < len(r1) or bad) and not paksa:
        raise SystemExit(f"Lembar belum lengkap/valid: {n_lengkap}/{len(r1)} Kategori utama, {bad} isian tidak dikenal. "
                         "Selesaikan dulu (atau --paksa untuk uji, bukan laporan).")
    lq = pd.read_csv(p["lq"])
    man = pd.read_csv(p["manifest"])
    d = r1.merge(key[["kode", "image_id", "strata", "d_image"]], on="kode").merge(
        lq[["image_id", "label_quality", "missing_score", "badloc_score", "swap_score"]], on="image_id", how="left")
    d = d[d.utama.isin(ALL_CATS)].reset_index(drop=True)

    res = {"waktu": dt.datetime.now().isoformat(timespec="seconds"), "tiruan": tiruan,
           "kelengkapan": {"n_baris": int(len(r1)), "n_lengkap": n_lengkap, "n_tidak_valid": bad,
                           "n_kurang_yakin": int((r1.yakin == "Kurang yakin").sum()),
                           "n_tinggi": int((d.strata == "tinggi").sum()), "n_acak": int((d.strata == "acak").sum())}}
    res["utama"] = banding(d, "y_label")
    res["s1"] = banding(d[d.yakin != "Kurang yakin"], "y_label")
    res["s2"] = banding(d, "y_label_utama")
    res["y_hilang"] = banding(d, "y_hilang")
    res["y_presisi"] = banding(d, "y_presisi")
    res["kategori"] = {}
    for c in ALL_CATS:
        d["_c"] = ((d.utama == c) | (d.tambahan == c)).astype(int)
        res["kategori"][c] = banding(d, "_c")
    d = d.drop(columns="_c")
    res["sebaran_tanpa"] = {v: (int(((d.strata == "tinggi") & (d.tanpa == v)).sum()),
                                int(((d.strata == "acak") & (d.tanpa == v)).sum())) for v in TANPA_VALS}
    if p["lembar2"].exists():
        r2 = turunan(baca_lembar(p["lembar2"], "Rater2"))
        res["kesepakatan"] = kesepakatan(r1, r2)
    else:
        res["kesepakatan"] = {"tersedia": False, "n": 0}
    res["validasi_skor"] = validasi_skor(d)
    valid = res["validasi_skor"]["1 - label_quality vs Y_label [semua]"]["valid"]
    res["paparan"] = paparan(d, man, lq, valid)

    out = p["out"]
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "audit_per_citra.csv", index=False)
    (out / "audit_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    tulis_laporan(res, out, tiruan)
    print(f"Selesai -> {out / 'AUDIT_REPORT.md'}")
    print(f"Utama: {res['utama'].get('verdict')} (selisih {res['utama'].get('selisih', float('nan')):+.3f})")
    return res


# ----------------------------------------------------------------------------- data tiruan
def buat_tiruan(dirp: Path, efek: float, seed: int) -> dict:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    dirp.mkdir(parents=True, exist_ok=True)
    (dirp / "_kunci").mkdir(exist_ok=True)
    N = 3000
    ids = [f"img_{i:05d}.jpg" for i in range(N)]
    dimg = np.clip(nrng.beta(2, 3, N), 0, 1)
    dimg[nrng.choice(N, 120, replace=False)] = 1.0
    ps = 0.3 / N + 0.7 * dimg / dimg.sum()
    pd.DataFrame({"image_id": ids, "d_image": dimg, "p_sampling": ps}).to_csv(dirp / "manifest.csv", index=False)
    base = 0.30
    p_true = np.clip(base + efek * (dimg - dimg.mean()) / (1 - dimg.mean()), 0.02, 0.98)
    y_true = (nrng.random(N) < p_true).astype(int)
    lq = np.clip(np.where(y_true == 1, nrng.beta(2, 3, N), nrng.beta(5, 2, N)), 0, 1)
    pd.DataFrame({"image_id": ids, "label_quality": lq, "missing_score": 1 - lq + nrng.normal(0, .1, N),
                  "badloc_score": nrng.random(N), "swap_score": nrng.random(N)}).to_csv(dirp / "lq.csv", index=False)
    order = np.argsort(-dimg, kind="stable")
    top = list(order[:100])
    rest = [i for i in range(N) if i not in set(top)]
    rng.shuffle(rest)
    picks = [(i, "tinggi") for i in top] + [(i, "acak") for i in rest[:50]]
    rng.shuffle(picks)
    key = pd.DataFrame([{"kode": f"A{k:03d}", "image_id": ids[i], "strata": s, "d_image": dimg[i],
                         "label_quality": lq[i]} for k, (i, s) in enumerate(picks, 1)])
    key.to_csv(dirp / "_kunci" / "audit_key.csv", index=False)

    def jawab(i, noise):
        y = y_true[i] if rng.random() > noise else 1 - y_true[i]
        ut = rng.choice(LABEL_CATS) if y else rng.choice(OTHER_CATS)
        tb = rng.choice([None, None, rng.choice(OTHER_CATS)])
        tanpa = rng.choice(["1-2", "3-5", ">5"]) if (y and ut == "Daun tanpa kotak") else rng.choice(["0", "0", "1-2", "Ragu"])
        return ut, tb, tanpa, rng.choice(YAKIN_VALS + ["Yakin"] * 3), rng.choice(PRESISI_VALS)

    def tulis(path, sheet, rows):
        wb = Workbook()
        ws = wb.active
        ws.title = sheet
        ws.append(["No", H_KODE, "Gambar", "Jumlah kotak", "Kelas pada label", H_UTAMA, H_TAMBAH, H_TANPA, H_YAKIN,
                   H_PRESISI, "Catatan"])
        for n, (kode, (ut, tb, tanpa, yk, pr)) in enumerate(rows, 1):
            tv = {"1-2": dt.datetime(2026, 2, 1), "3-5": dt.datetime(2026, 5, 3), "0": 0}.get(tanpa, tanpa) \
                if n % 3 == 0 else tanpa  # sebagian disimpan seperti konversi Excel
            ws.append([n, kode, "Buka", 1, "CMD x1", ut, tb, tv, yk, pr, None])
        wb.save(path)

    ans = {f"A{k:03d}": jawab(i, 0.10) for k, (i, s) in enumerate(picks, 1)}
    tulis(dirp / "lembar1.xlsx", "Audit", list(ans.items()))
    r2k = rng.sample([k for k, (i, s) in zip(ans, picks) if s == "tinggi"], 20) + \
        rng.sample([k for k, (i, s) in zip(ans, picks) if s == "acak"], 10)
    i_of = {f"A{k:03d}": i for k, (i, s) in enumerate(picks, 1)}
    r2 = [(k, ans[k] if rng.random() < 0.8 else jawab(i_of[k], 0.10)) for k in r2k]
    tulis(dirp / "lembar2.xlsx", "Rater2", r2)
    return {"lembar1": dirp / "lembar1.xlsx", "lembar2": dirp / "lembar2.xlsx", "kunci": dirp / "_kunci" / "audit_key.csv",
            "lq": dirp / "lq.csv", "manifest": dirp / "manifest.csv", "out": dirp / "analysis"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paksa", action="store_true")
    ap.add_argument("--uji-tiruan", type=Path, default=None)
    ap.add_argument("--efek", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.uji_tiruan:
        jalankan(buat_tiruan(a.uji_tiruan, a.efek, a.seed), paksa=False, tiruan=True)
        return
    root = Path(__file__).resolve().parent
    aud = root / "audit_label_150"
    dd = root / "stage_v2_difficulty_20261028"
    jalankan({"lembar1": aud / "LEMBAR_AUDIT_LABEL.xlsx", "lembar2": aud / "LEMBAR_AUDIT_RATER2.xlsx",
              "kunci": aud / "_kunci" / "audit_key.csv", "lq": dd / "train_image_label_quality.csv",
              "manifest": dd / "train_image_difficulty_manifest.csv", "out": aud / "analysis"}, a.paksa, tiruan=False)


if __name__ == "__main__":
    main()

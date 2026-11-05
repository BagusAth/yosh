#!/usr/bin/env python3
"""
Analisis pasca-hoc butir 6–7: kontribusi jenis error dalam satuan AP (TIDE) dan AP per ambang IoU
===================================================================================================
Spesifikasi lengkap: stage_v2_ablation/analysis/tide_iou/SPESIFIKASI.md (ditulis 5 Nov 2026 17.00 WIB, sebelum
dijalankan; SHA-256 c69090cb1ff12618a8175b9bdd4eb8a4fd0d05ba1ceb2a6f8902a19d0832441f). Disetujui pengguna 5 Nov 2026 16.37.

Status: PASCA-HOC dan EKSPLORATIF. Tidak mengubah verdict RQ1–RQ3. Hanya validasi (2.567 citra); test set tidak
disentuh (pra-registrasi Fase 6 bagian 7). Tanpa GPU; memakai prediksi per citra 15 model yang sudah ada.

Butir 6 (TIDE, Bolya dkk. 2020): aturan tidecv 1.0.1 (commit 49a5d2a) — pos 0,5, latar 0,1, max_dets 100, mode
kotak, semua citra validasi didaftarkan. Enam jenis error (Cls, Loc, Both, Dupe, Bkg, Miss) dan dua error khusus
(FalsePos, FalseNeg). dAP = AP50 setelah satu jenis diperbaiki − AP50 asli (jenis utama dipotong pada 0 per model,
seperti tidecv). Implementasi numpy di bawah meniru tidecv baris demi baris agar dapat di-bootstrap; mode
`cek-tidecv` membandingkannya dengan kode tidecv asli (syarat pakai: selisih < 1e-6 poin pada ke-15 model).

Butir 7 (diilhami PISA, Cao dkk. 2020): AP per ambang IoU 0,50…0,95 dengan pencocokan dan AP hitung ulang yang sama
dengan analisis utama (prepare_run + fast_map). Syarat pakai: rata-rata 10 ambang = mAP@50–95 di point.json dan =
kolom |m di boot_seed*.csv (resampel identik, generator seed 9000 + k).

Luaran utama: (6) selisih dAP_Bkg EXP-1 − EXP-0R; (7) selisih AP per ambang EXP-1 − EXP-0R dan
G = rata-rata selisih IoU 0,75–0,95 − rata-rata selisih IoU 0,50–0,70. Pendamping: jenis error lain, FalsePos,
FalseNeg, AP50 TIDE, selisih relatif R_t dan G_rel, semua kontras, jumlah error per citra (deskriptif).
IK 95%: bootstrap klaster per grup foto, 1.000 iterasi (4 × 250), resampel sama untuk semua model, dirata-rata antar seed.

Pemakaian (dari folder skripsi):
  python analyze_tide_iou.py prep
  python analyze_tide_iou.py point
  python analyze_tide_iou.py cek-tidecv --tidecv-src <folder repo tide>   (butuh tidecv; pycocotools boleh diganti shim)
  python analyze_tide_iou.py boot --seed 0 --iters 250   (ulangi untuk seed 1, 2, 3)
  python analyze_tide_iou.py report
Keluaran: stage_v2_ablation/analysis/tide_iou/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_ablation import CONTRASTS, EXPS, SEEDS, split_by_class  # noqa: E402
from analyze_eleak_per_image import CLASS_NAMES, load_groups, prepare_run  # noqa: E402

NC = len(CLASS_NAMES)
POS, BG, MAXDET = 0.5, 0.1, 100
IOUS = np.linspace(0.5, 0.95, 10)
X101_TIDE = np.array([x / 100 for x in range(101)])  # sama persis dengan tidecv (bukan linspace)
X101_UL = np.linspace(0, 1, 101)                       # sama dengan analyze_ablation.fast_map
_trap = getattr(np, "trapezoid", None) or np.trapz
EPS = 1e-16
MAIN = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
SPECIAL = ["FalsePos", "FalseNeg"]
VARIANTS = ["orig"] + MAIN + SPECIAL
# kode jenis prediksi
TP, E_CLS, E_LOC, E_BOTH, E_DUPE, E_BKG = 0, 1, 2, 3, 4, 5
TYPE_OF = {"Cls": E_CLS, "Loc": E_LOC, "Both": E_BOTH, "Dupe": E_DUPE, "Bkg": E_BKG}


# ----------------------------------------------------------------------------- TIDE per citra
def coco_bbox_iou(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """IoU kotak [x, y, w, h] seperti pycocotools maskApi bbIou (iscrowd = False). Bentuk (n_det, n_gt)."""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    w = np.minimum(d[:, None, 0] + d[:, None, 2], g[None, :, 0] + g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0])
    h = np.minimum(d[:, None, 1] + d[:, None, 3], g[None, :, 1] + g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1])
    inter = np.where((w > 0) & (h > 0), w * h, 0.0)
    da = d[:, 2] * d[:, 3]
    ga = g[:, 2] * g[:, 3]
    u = da[:, None] + ga[None, :] - inter
    return np.where(inter > 0, inter / np.where(u > 0, u, 1.0), 0.0)


def tide_image(pb: np.ndarray, pc: np.ndarray, ps: np.ndarray, gb: np.ndarray, gc: np.ndarray) -> dict:
    """Meniru tidecv.quantify.TIDEExample + TIDERun._eval_image + BestGTMatch untuk satu citra.
    pb/gb: kotak [x, y, w, h]; prediksi dalam urutan baris asli (diurutkan stabil di sini)."""
    ng = len(gc)
    order = np.argsort(-ps, kind="stable")[:MAXDET]
    pb, pc, ps = pb[order], pc[order], ps[order]
    nd = len(pc)
    ptype = np.full(nd, -1, dtype=np.int8)
    fix_ok = np.zeros(nd, dtype=bool)        # Loc/Cls: menjadi TP bila jenisnya diperbaiki
    fix_cls = pc.copy()                       # Cls: kelas setelah diperbaiki
    used_g = np.zeros(ng, dtype=bool)
    usable = np.zeros(ng, dtype=bool)
    if nd == 0:
        return {"pc": pc, "ps": ps, "ptype": ptype, "fix_ok": fix_ok, "fix_cls": fix_cls, "gc": gc,
                "missed": np.ones(ng, dtype=bool), "unused": np.ones(ng, dtype=bool)}
    if ng == 0:
        ptype[:] = E_BKG
        return {"pc": pc, "ps": ps, "ptype": ptype, "fix_ok": fix_ok, "fix_cls": fix_cls, "gc": gc,
                "missed": np.zeros(0, dtype=bool), "unused": np.zeros(0, dtype=bool)}
    iou = coco_bbox_iou(pb, gb)
    same = pc[:, None] == gc[None, :]
    cls_iou = iou * same
    buf = cls_iou.copy()
    for i in range(nd):
        j = int(np.argmax(buf[i]))
        if buf[i, j] >= POS:
            ptype[i] = TP
            used_g[j] = True
            buf[:, j] = 0
    noncls_iou = iou * ~same
    used_cls = (used_g[None, :] * iou) * same
    best_taken = np.zeros(ng, dtype=bool)
    for i in range(nd):
        if ptype[i] == TP:
            continue
        j = int(np.argmax(cls_iou[i]))
        if BG <= cls_iou[i, j] <= POS:
            ptype[i] = E_LOC
        else:
            j = int(np.argmax(noncls_iou[i]))
            if noncls_iou[i, j] >= POS:
                ptype[i] = E_CLS
                fix_cls[i] = gc[j]
            else:
                j = int(np.argmax(used_cls[i]))
                if used_cls[i, j] >= POS:
                    ptype[i] = E_DUPE
                elif iou[i].max() <= BG:
                    ptype[i] = E_BKG
                else:
                    ptype[i] = E_BOTH
                continue
        # BestGTMatch (Loc/Cls): hanya bila GT sasaran belum terpakai; skor tertinggi (yang pertama) yang diperbaiki
        if not used_g[j]:
            usable[j] = True
            if not best_taken[j]:
                best_taken[j] = True
                fix_ok[i] = True
    unused = ~used_g
    return {"pc": pc, "ps": ps, "ptype": ptype, "fix_ok": fix_ok, "fix_cls": fix_cls, "gc": gc,
            "missed": unused & ~usable, "unused": unused}


def tide_prepare(preds_csv: Path, gt_csv: Path, images: np.ndarray) -> dict:
    """Jalankan tide_image untuk semua citra (urutan `images`) dan susun titik data per variasi perbaikan,
    dalam urutan sisip yang sama dengan tidecv (ap_data asli; fix_errors: titik error dulu, lalu TP asli)."""
    P = pd.read_csv(preds_csv)
    G = pd.read_csv(gt_csv)
    pidx = P.groupby("image_id", sort=False).indices
    gidx = G.groupby("image_id", sort=False).indices
    Pb = P[["x1", "y1", "x2", "y2"]].to_numpy(float); Pc = P.class_id.to_numpy(int); Ps = P.confidence.to_numpy(float)
    Gb = G[["x1", "y1", "x2", "y2"]].to_numpy(float); Gc = G.class_id.to_numpy(int)

    def xywh(b):
        return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) if len(b) else np.zeros((0, 4))

    n = len(images)
    gt_cnt = np.zeros((n, NC), dtype=np.int64)
    miss_cnt = np.zeros((n, NC), dtype=np.int64)
    unused_cnt = np.zeros((n, NC), dtype=np.int64)
    rows = []  # per prediksi: img, cls, score, ptype, fix_ok, fix_cls
    err_counts = np.zeros((n, 6), dtype=np.int64)  # Cls, Loc, Both, Dupe, Bkg, Miss
    empty = np.array([], dtype=int)
    for k, im in enumerate(images):
        pi = np.sort(pidx.get(im, empty)); gi = np.sort(gidx.get(im, empty))
        r = tide_image(xywh(Pb[pi]), Pc[pi], Ps[pi], xywh(Gb[gi]), Gc[gi])
        np.add.at(gt_cnt[k], r["gc"], 1)
        np.add.at(miss_cnt[k], r["gc"][r["missed"]], 1)
        np.add.at(unused_cnt[k], r["gc"][r["unused"]], 1)
        m = len(r["pc"])
        if m:
            rows.append(np.column_stack([np.full(m, k), r["pc"], r["ps"], r["ptype"], r["fix_ok"], r["fix_cls"]]))
        for t, code in TYPE_OF.items():
            err_counts[k, MAIN.index(t)] = int((r["ptype"] == code).sum())
        err_counts[k, 5] = int(r["missed"].sum())
    R = np.concatenate(rows) if rows else np.zeros((0, 6))
    img = R[:, 0].astype(np.int64); cls = R[:, 1].astype(np.int64); sc = R[:, 2]; pt = R[:, 3].astype(np.int64)
    fok = R[:, 4].astype(bool); fcls = R[:, 5].astype(np.int64)
    is_tp = pt == TP
    out = {"images": np.asarray(images), "gt_cnt": gt_cnt, "miss_cnt": miss_cnt, "unused_cnt": unused_cnt,
           "err_counts": err_counts}
    # orig: semua prediksi dalam urutan citra/prediksi
    out["orig"] = (img, cls, sc, is_tp.astype(float))
    e = ~is_tp
    e_img, e_cls, e_sc, e_pt, e_fok, e_fcls = img[e], cls[e], sc[e], pt[e], fok[e], fcls[e]
    t_img, t_cls, t_sc = img[is_tp], cls[is_tp], sc[is_tp]
    for name in MAIN:
        if name == "Miss":
            keep = np.ones(len(e_img), dtype=bool)  # semua FP tetap; jumlah GT dikurangi di AP
            a = (e_img, e_cls, e_sc, np.zeros(len(e_img)))
        else:
            code = TYPE_OF[name]
            mine = e_pt == code
            if name in ("Cls", "Loc"):
                keep = ~mine | e_fok
                tpf = (mine & e_fok).astype(float)
                ccls = np.where(mine & e_fok & (name == "Cls"), e_fcls, e_cls)
            else:
                keep = ~mine
                tpf = np.zeros(len(e_img))
                ccls = e_cls
            a = (e_img[keep], ccls[keep], e_sc[keep], tpf[keep])
        out[name] = tuple(np.concatenate([x, y]) for x, y in zip(a, (t_img, t_cls, t_sc, np.ones(len(t_img)))))
    # FalsePos: skor FP = 0, TP = 1 (titik error dulu, lalu TP)
    out["FalsePos"] = (np.concatenate([e_img, t_img]), np.concatenate([e_cls, t_cls]),
                       np.concatenate([np.zeros(len(e_img)), np.ones(len(t_img))]),
                       np.concatenate([np.zeros(len(e_img)), np.ones(len(t_img))]))
    out["FalseNeg"] = (np.concatenate([e_img, t_img]), np.concatenate([e_cls, t_cls]),
                       np.concatenate([e_sc, t_sc]), np.concatenate([np.zeros(len(e_img)), np.ones(len(t_img))]))
    return out


def tide_sorted(prep: dict) -> dict:
    """Per variasi dan kelas: indeks citra dan flag TP, terurut menurut skor (stabil terhadap urutan sisip)."""
    S = {}
    for v in VARIANTS:
        img, cls, sc, tp = prep[v]
        per = []
        for c in range(NC):
            m = cls == c
            o = np.argsort(-sc[m], kind="stable")
            per.append((img[m][o], tp[m][o]))
        S[v] = per
    return S


def tide_ap_class(img: np.ndarray, tp: np.ndarray, w_img: np.ndarray, ngt: float) -> float:
    """APDataObject.get_ap tidecv dengan bobot citra (bobot bilangan bulat = citra diduplikasi)."""
    if ngt <= 0:
        return 0.0
    w = w_img[img].astype(np.float64)
    k = w > 0
    if not k.any():
        return 0.0
    w = w[k]; t = tp[k]
    tpc = np.cumsum(w * t)
    prec = tpc / np.cumsum(w)
    rec = tpc / ngt
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, X101_TIDE, side="left")
    y = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(y.sum() / len(y) * 100)


def tide_scores(S: dict, prep: dict, w_img: np.ndarray) -> dict:
    gt = (prep["gt_cnt"] * w_img[:, None]).sum(0).astype(float)
    miss = (prep["miss_cnt"] * w_img[:, None]).sum(0).astype(float)
    unused = (prep["unused_cnt"] * w_img[:, None]).sum(0).astype(float)
    res = {}
    for v in VARIANTS:
        ngt = gt - miss if v == "Miss" else (gt - unused if v == "FalseNeg" else gt)
        aps = []
        for c in range(NC):
            img, tp = S[v][c]
            has_pts = bool((w_img[img] > 0).any()) if len(img) else False
            if not has_pts and ngt[c] == 0:
                continue  # kelas kosong tidak dihitung (is_empty di tidecv)
            aps.append(tide_ap_class(img, tp, w_img, ngt[c]))
        res[v] = sum(aps) / len(aps)
    out = {"AP50_tide": res["orig"]}
    for v in MAIN:
        out[f"dAP_{v}"] = max(res[v] - res["orig"], 0.0)
    for v in SPECIAL:
        out[f"dAP_{v}"] = res[v] - res["orig"]
    return out


# ----------------------------------------------------------------------------- AP per ambang IoU (butir 7)
def ap_per_iou(sp: dict, img_w: np.ndarray) -> np.ndarray:
    """Salinan analyze_ablation.fast_map yang mengembalikan AP rata-rata kelas per ambang IoU (10 angka)."""
    nl = (sp["gt_counts"] * img_w[:, None]).sum(0)
    classes = np.nonzero(nl > 0)[0]
    ap = np.zeros((len(classes), 10))
    for ci, c in enumerate(classes):
        d = sp["pc"][c]
        if len(d["img"]) == 0:
            continue
        w = img_w[d["img"]].astype(np.float64)
        k = w > 0
        if not k.any():
            continue
        w = w[k]
        tpc = np.cumsum(d["tp"][k] * w[:, None], axis=0)
        cw = np.cumsum(w)
        n = len(w)
        mpre = np.empty((n + 2, 10)); mpre[0] = 1; mpre[1:-1] = tpc / cw[:, None]; mpre[-1] = 0
        mpre = np.maximum.accumulate(mpre[::-1], axis=0)[::-1]
        mrec = np.empty((n + 2, 10)); mrec[0] = 0; mrec[1:-1] = tpc / (nl[c] + EPS); mrec[-1] = 1
        for j in range(10):
            ap[ci, j] = _trap(np.interp(X101_UL, mrec[:, j], mpre[:, j]), X101_UL)
    return ap.mean(axis=0)


# ----------------------------------------------------------------------------- utilitas
def keys_all():
    return [(s, e, lab) for s in SEEDS for e, lab in EXPS]


def load_cache(cache: Path, s: int, e: str):
    z = dict(np.load(cache / f"seed{s}_{e}.npz", allow_pickle=False))
    sp = split_by_class({"det_img": z["ul_det_img"], "det_cls": z["ul_det_cls"], "det_tp": z["ul_det_tp"],
                         "gt_counts": z["ul_gt_counts"]})
    prep = {k: z[k] for k in ("images", "gt_cnt", "miss_cnt", "unused_cnt", "err_counts")}
    for v in VARIANTS:
        prep[v] = (z[f"{v}_img"], z[f"{v}_cls"], z[f"{v}_sc"], z[f"{v}_tp"])
    return sp, prep


def all_scores(sp, S, prep, w) -> dict:
    r = tide_scores(S, prep, w)
    a = ap_per_iou(sp, w)
    for j, t in enumerate(IOUS):
        r[f"AP{int(round(t * 100))}"] = float(a[j] * 100)
    r["m"] = float(a.mean())
    return r


def stat_names() -> list[str]:
    return ["AP50_tide"] + [f"dAP_{v}" for v in MAIN + SPECIAL] + [f"AP{int(round(t * 100))}" for t in IOUS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "point", "cek-tidecv", "boot", "report"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--per-image-dir", default="stage_v2_ablation/per_image")
    ap.add_argument("--gt-name", default="gt_cleanval.csv")
    ap.add_argument("--main-analysis", default="stage_v2_ablation/analysis")
    ap.add_argument("--out", default="stage_v2_ablation/analysis/tide_iou")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--only", default=None, help="Batasi ke model tertentu, mis. 42:exp_1_adaptive_sampling")
    ap.add_argument("--tidecv-src", default=None, help="Folder repo tidecv (berisi paket tidecv/)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    pdir = root / args.per_image_dir
    out = root / args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    gt_csv = pdir / args.gt_name
    keys = keys_all()
    if args.only:
        s0, e0 = args.only.split(":")
        keys = [k for k in keys if k[0] == int(s0) and k[1] == e0]

    if args.mode == "prep":
        for s, e, lab in keys:
            dst = cache / f"seed{s}_{e}.npz"
            f = pdir / f"preds_seed{s}_{e}.csv"
            if dst.exists():
                continue
            t0 = time.time()
            ul = prepare_run(f, gt_csv)
            tp = tide_prepare(f, gt_csv, ul["images"])
            z = {"ul_det_img": ul["det_img"], "ul_det_cls": ul["det_cls"], "ul_det_tp": ul["det_tp"],
                 "ul_gt_counts": ul["gt_counts"]}
            for k in ("images", "gt_cnt", "miss_cnt", "unused_cnt", "err_counts"):
                z[k] = tp[k]
            for v in VARIANTS:
                for nm, arr in zip(("img", "cls", "sc", "tp"), tp[v]):
                    z[f"{v}_{nm}"] = arr
            np.savez_compressed(dst, **z)
            print(f"prep {dst.name}: {time.time() - t0:.1f} detik", flush=True)
        return

    if args.mode == "point":
        P = json.loads((root / args.main_analysis / "point.json").read_text(encoding="utf-8"))["recomputed"]
        res, cek = {}, {}
        for s, e, lab in keys:
            sp, prep = load_cache(cache, s, e)
            S = tide_sorted(prep)
            w = np.ones(len(prep["images"]), dtype=np.int64)
            r = all_scores(sp, S, prep, w)
            ec = prep["err_counts"].sum(0) / len(prep["images"])
            r.update({f"n_{v}_per_citra": float(x) for v, x in zip(MAIN, ec)})
            res[f"{s}|{lab}"] = r
            ref = P[f"{s}|{lab}"]["mAP50_95"]
            cek[f"{s}|{lab}"] = {"mAP50_95_point_json": ref, "rata_rata_10_ambang": r["m"], "selisih": r["m"] - ref,
                                  "mAP50_point_json": P[f"{s}|{lab}"]["mAP50"] * 100, "AP50_tide": r["AP50_tide"]}
            print(f"{s}|{lab}: AP50_tide {r['AP50_tide']:.3f}, dAP_Bkg {r['dAP_Bkg']:.3f}, selisih mAP {r['m'] - ref:+.2e}", flush=True)
        (out / "point_tide_iou.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        mx = max(abs(v["selisih"]) for v in cek.values())
        f = out / "cek_utama.json"
        C = json.loads(f.read_text()) if f.exists() else {}
        C.update({"titik": cek, "maks_selisih_titik": mx, "n_images": int(len(prep["images"]))})
        f.write_text(json.dumps(C, indent=2), encoding="utf-8")
        print(f"maks |rata-rata 10 ambang − mAP50_95 point.json| = {mx:.2e}")
        return

    if args.mode == "cek-tidecv":
        if args.tidecv_src:
            sys.path.insert(0, str(Path(args.tidecv_src).resolve()))
        try:
            import pycocotools.mask  # noqa: F401
            shim = False
        except ImportError:
            # Shim: tidecv hanya memakai mask_utils.iou untuk kotak (iscrowd False) — diganti rumus bbIou COCO yang sama.
            pkg = types.ModuleType("pycocotools"); mod = types.ModuleType("pycocotools.mask")
            mod.iou = lambda d, g, crowd: coco_bbox_iou(np.asarray(d, float).reshape(-1, 4), np.asarray(g, float).reshape(-1, 4)) if len(d) and len(g) else []
            pkg.mask = mod
            sys.modules["pycocotools"] = pkg; sys.modules["pycocotools.mask"] = mod
            shim = True
        try:
            import appdirs  # noqa: F401
        except ImportError:
            ad = types.ModuleType("appdirs"); ad.user_data_dir = lambda *a, **k: str(out / "_appdirs"); sys.modules["appdirs"] = ad
        try:
            import seaborn  # noqa: F401
        except ImportError:
            sys.modules["seaborn"] = types.ModuleType("seaborn")  # hanya dipakai modul plotting tidecv, tidak dipanggil di sini
        import tidecv  # noqa: E402
        from tidecv import TIDE, Data  # noqa: E402
        G = pd.read_csv(gt_csv)
        hasil = {}
        prev = json.loads((out / "cek_tidecv.json").read_text()) if (out / "cek_tidecv.json").exists() else {}
        for s, e, lab in keys:
            sp, prep = load_cache(cache, s, e)
            images = list(prep["images"])
            S = tide_sorted(prep)
            mine = tide_scores(S, prep, np.ones(len(images), dtype=np.int64))
            P = pd.read_csv(pdir / f"preds_seed{s}_{e}.csv")
            gd, pdd = Data("gt"), Data(f"{s}|{lab}")
            idx = {n: i for i, n in enumerate(images)}
            for i, n in enumerate(images):
                gd.add_image(i, n); pdd.add_image(i, n)
            for r in G.itertuples(index=False):
                gd.add_ground_truth(idx[r.image_id], int(r.class_id), [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1])
            # prediksi dalam urutan baris CSV per citra (sama dengan tide_prepare)
            for r in P.itertuples(index=False):
                pdd.add_detection(idx[r.image_id], int(r.class_id), float(r.confidence), [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1])
            t0 = time.time()
            td = TIDE()
            td.evaluate(gd, pdd, mode=TIDE.BOX)
            me = td.get_main_errors()[f"{s}|{lab}"]; se = td.get_special_errors()[f"{s}|{lab}"]
            ref = {"AP50_tide": td.runs[f"{s}|{lab}"].ap}
            ref.update({f"dAP_{k}": float(v) for k, v in me.items()})
            ref.update({f"dAP_{k}": float(v) for k, v in se.items()})
            d = {k: mine[k] - ref[k] for k in ref}
            hasil[f"{s}|{lab}"] = {"tidecv": ref, "numpy": mine, "selisih": d, "maks": max(abs(x) for x in d.values()),
                                   "detik": time.time() - t0}
            print(f"{s}|{lab}: maks selisih {hasil[f'{s}|{lab}']['maks']:.2e} ({time.time() - t0:.0f} detik)", flush=True)
        prev.update(hasil)
        prev["_catatan"] = {"tidecv_versi": getattr(tidecv, "__version__", "?"), "shim_pycocotools": shim,
                            "maks_semua": max(v["maks"] for k, v in prev.items() if not k.startswith("_"))}
        (out / "cek_tidecv.json").write_text(json.dumps(prev, indent=2), encoding="utf-8")
        return

    if args.mode == "boot":
        groups = load_groups(root)
        models = {}
        images = None
        for s, e, lab in keys:
            sp, prep = load_cache(cache, s, e)
            if images is None:
                images = prep["images"]
            assert np.array_equal(images, prep["images"]), "Himpunan citra antar model berbeda"
            models[(s, lab)] = (sp, tide_sorted(prep), prep)
        gids = np.array([groups.get(n, n) for n in images])
        uniq, inv = np.unique(gids, return_inverse=True)
        rng = np.random.default_rng(9000 + args.seed)  # sama dengan analyze_ablation.py boot
        rows = []
        t0 = time.time()
        for it in range(args.iters):
            cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            w = cnt[inv]
            r = {}
            for (s, lab), (sp, S, prep) in models.items():
                for k, v in all_scores(sp, S, prep, w).items():
                    r[f"{s}|{lab}|{k}"] = v
            rows.append(r)
            if (it + 1) % 25 == 0:
                print(f"seed {args.seed}: {it + 1}/{args.iters} ({time.time() - t0:.0f} detik)", flush=True)
        pd.DataFrame(rows).to_csv(out / f"boot_tide_iou_seed{args.seed}.csv", index=False)
        # verifikasi resampel identik: rata-rata 10 ambang = kolom |m analisis utama
        ref = pd.read_csv(root / args.main_analysis / f"boot_seed{args.seed}.csv").iloc[:args.iters]
        B = pd.DataFrame(rows)
        mx = max(float(np.max(np.abs(B[f"{s}|{lab}|m"].to_numpy() - ref[f"{s}|{lab}|m"].to_numpy()))) for (s, lab) in models)
        f = out / "cek_utama.json"
        C = json.loads(f.read_text()) if f.exists() else {}
        C[f"boot_seed{args.seed}_maks_selisih_m"] = mx
        f.write_text(json.dumps(C, indent=2), encoding="utf-8")
        print(f"boot seed {args.seed}: {len(rows)} iterasi; maks |m − boot_seed{args.seed}.csv| = {mx:.2e}")
        return

    # ------------------------------------------------------------------------- report
    P = json.loads((out / "point_tide_iou.json").read_text(encoding="utf-8"))
    C = json.loads((out / "cek_utama.json").read_text(encoding="utf-8"))
    T = json.loads((out / "cek_tidecv.json").read_text(encoding="utf-8")) if (out / "cek_tidecv.json").exists() else None
    ftr = root / "_panduan_kerja" / "verif_tide_iou" / "hasil_transkripsi.json"
    TR = json.loads(ftr.read_text(encoding="utf-8")) if ftr.exists() else None
    files = sorted(glob.glob(str(out / "boot_tide_iou_seed*.csv")))
    B = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    labs = [lab for _, lab in EXPS]
    names = stat_names()
    hi = [f"AP{int(round(t * 100))}" for t in IOUS[5:]]
    lo = [f"AP{int(round(t * 100))}" for t in IOUS[:5]]

    def contrast(coef: dict, stat_fn):
        per_seed = [stat_fn(lambda lab, k: P[f"{s}|{lab}"][k], coef) for s in SEEDS]
        boot = np.mean([stat_fn(lambda lab, k, s=s: B[f"{s}|{lab}|{k}"].to_numpy(), coef) for s in SEEDS], axis=0)
        lo_, hi_ = np.percentile(boot, [2.5, 97.5])
        mean = float(np.mean(per_seed))
        if all(d > 0 for d in per_seed) and lo_ > 0:
            v = "meningkat konsisten"
        elif all(d < 0 for d in per_seed) and hi_ < 0:
            v = "menurun konsisten"
        else:
            v = "tidak terbukti berbeda"
        return {"per_seed": [float(x) for x in per_seed], "mean": mean, "ci95": [float(lo_), float(hi_)], "verdict": v}

    def lin(name):
        return lambda get, coef: sum(c * get(lab, name) for lab, c in coef.items())

    def gap(get, coef):
        return sum(c * (np.mean([get(lab, k) for k in hi], axis=0) - np.mean([get(lab, k) for k in lo], axis=0)) for lab, c in coef.items())

    res = {}
    for cname, coef, arti in CONTRASTS:
        rc = {k: contrast(coef, lin(k)) for k in names}
        rc["G"] = contrast(coef, gap)
        res[cname] = rc

    # selisih relatif EXP-1 vs EXP-0R (AP dijumlah antar seed)
    def rel(get, num="EXP-1", den="EXP-0R"):
        R = {}
        for k in hi + lo:
            R[k] = sum(get(s, num, k) for s in SEEDS) / sum(get(s, den, k) for s in SEEDS) - 1
        return R

    Rp = rel(lambda s, lab, k: P[f"{s}|{lab}"][k])
    Rb = rel(lambda s, lab, k: B[f"{s}|{lab}|{k}"].to_numpy())
    rel_res = {}
    for k in lo + hi:
        a, b = np.percentile(Rb[k], [2.5, 97.5])
        rel_res[k] = {"R": float(Rp[k]), "ci95": [float(a), float(b)]}
    Gp = np.mean([Rp[k] for k in hi]) - np.mean([Rp[k] for k in lo])
    Gb = np.mean([Rb[k] for k in hi], axis=0) - np.mean([Rb[k] for k in lo], axis=0)
    rel_res["G_rel"] = {"G_rel": float(Gp), "ci95": [float(x) for x in np.percentile(Gb, [2.5, 97.5])]}

    def mean_lab(lab, k):
        return float(np.mean([P[f"{s}|{lab}"][k] for s in SEEDS]))

    def fmt(r, nd=2):
        return f"{r['mean']:+.{nd}f} ({r['ci95'][0]:+.{nd}f} s.d. {r['ci95'][1]:+.{nd}f})"

    L = ["# Analisis pasca-hoc butir 6–7: TIDE dan AP per ambang IoU (validasi)", "",
         f"Tanggal: {datetime.now():%Y-%m-%d %H:%M} · {C['n_images']:,} citra validasi · "
         f"bootstrap klaster {len(B)} iterasi · spesifikasi: SPESIFIKASI.md (ditulis sebelum dijalankan)", "",
         "**Status: pasca-hoc dan eksploratif.** Tidak mengubah verdict RQ1–RQ3. Test set tidak disentuh.", "",
         "## Verifikasi", "",
         f"- Rata-rata AP 10 ambang vs mAP@50–95 `point.json` (15 model): selisih maks {C['maks_selisih_titik']:.1e}.",
         "- Resampel bootstrap identik dengan analisis utama (rata-rata 10 ambang per resampel vs kolom `|m`): "
         + ", ".join(f"seed {k[9]} {v:.1e}" for k, v in C.items() if k.startswith("boot_seed")) + ".",
         (f"- Implementasi TIDE numpy vs kode tidecv asli ({T['_catatan']['tidecv_versi']}, shim pycocotools: "
          f"{'ya' if T['_catatan']['shim_pycocotools'] else 'tidak'}): {sum(1 for k in T if not k.startswith('_'))} model, "
          f"selisih maks {T['_catatan']['maks_semua']:.1e} poin.") if T else
         "- **Cek terhadap kode tidecv asli belum dijalankan** (syarat pakai butir 6 di SPESIFIKASI.md); jalankan "
         "`jalankan_cek_tidecv.bat`. Sampai cek itu lolos, angka butir 6 bersifat sementara.",
         (f"- Verifikasi antara: transkripsi Python murni alur tidecv (`_panduan_kerja/verif_tide_iou/transkripsi_tidecv.py`) "
          f"vs implementasi numpy: {len(TR)} model, selisih maks {max(v['maks'] for v in TR.values()):.1e} poin. "
          "Ini menguji vektorisasi, bukan tafsir aturan tidecv.") if TR else "", "",
         "## Butir 6 — kontribusi jenis error dalam satuan AP50 (TIDE), rata-rata 3 seed", "",
         "| Model | AP50 TIDE | Cls | Loc | Both | Dupe | Bkg | Miss | FalsePos | FalseNeg |", "| --- |" + " --- |" * 9]
    for lab in labs:
        L.append(f"| {lab} | {mean_lab(lab, 'AP50_tide'):.2f} | " + " | ".join(f"{mean_lab(lab, f'dAP_{v}'):.2f}" for v in MAIN + SPECIAL) + " |")
    L += ["", "Jumlah error per citra menurut TIDE (deskriptif, rata-rata 3 seed; max_dets 100, tanpa ambang keyakinan):", "",
          "| Model | Cls | Loc | Both | Dupe | Bkg | Miss |", "| --- |" + " --- |" * 6]
    for lab in labs:
        L.append(f"| {lab} | " + " | ".join(f"{mean_lab(lab, f'n_{v}_per_citra'):.2f}" for v in MAIN) + " |")
    for cname in ["EXP-1 − EXP-0R", "EXP-2 − EXP-0"]:
        L += ["", f"### {cname} (poin dAP; per seed · rata-rata (IK 95%) · verdict)", "",
              "| Besaran | Per seed | Rata-rata (IK 95%) | Verdict |", "| --- | --- | --- | --- |"]
        for k in ["AP50_tide"] + [f"dAP_{v}" for v in MAIN + SPECIAL]:
            r = res[cname][k]
            L.append(f"| {k} | " + " / ".join(f"{x:+.2f}" for x in r["per_seed"]) + f" | {fmt(r)} | {r['verdict']} |")
    L += ["", "Semua kontras (rata-rata (IK 95%)) untuk AP50 TIDE, dAP_Bkg, dAP_Miss, dAP_FalsePos, dAP_FalseNeg:", "",
          "| Kontras | AP50 TIDE | dAP_Bkg | dAP_Miss | dAP_FalsePos | dAP_FalseNeg |", "| --- | --- | --- | --- | --- | --- |"]
    for cname, _, _ in CONTRASTS:
        L.append(f"| {cname} | " + " | ".join(fmt(res[cname][k]) for k in ["AP50_tide", "dAP_Bkg", "dAP_Miss", "dAP_FalsePos", "dAP_FalseNeg"]) + " |")
    L += ["", "## Butir 7 — AP per ambang IoU (hitung ulang, poin), rata-rata 3 seed", "",
          "| Model | " + " | ".join(f"{t:.2f}" for t in IOUS) + " |", "| --- |" + " --- |" * 10]
    for lab in labs:
        L.append(f"| {lab} | " + " | ".join(f"{mean_lab(lab, f'AP{int(round(t * 100))}'):.2f}" for t in IOUS) + " |")
    for cname in ["EXP-1 − EXP-0R", "EXP-1 − EXP-0", "EXP-2 − EXP-0"]:
        L += ["", f"### {cname} per ambang IoU", "", "| Ambang | Per seed | Rata-rata (IK 95%) | Verdict |", "| --- | --- | --- | --- |"]
        for t in IOUS:
            k = f"AP{int(round(t * 100))}"
            r = res[cname][k]
            L.append(f"| {t:.2f} | " + " / ".join(f"{x:+.2f}" for x in r["per_seed"]) + f" | {fmt(r)} | {r['verdict']} |")
        r = res[cname]["G"]
        L.append(f"| G (0,75–0,95 − 0,50–0,70) | " + " / ".join(f"{x:+.2f}" for x in r["per_seed"]) + f" | {fmt(r)} | {r['verdict']} |")
    L += ["", "Selisih relatif EXP-1 terhadap EXP-0R (R_t = AP EXP-1 / AP EXP-0R − 1, AP dijumlah antar seed):", "",
          "| Ambang | R_t | IK 95% |", "| --- | --- | --- |"]
    for t in IOUS:
        k = f"AP{int(round(t * 100))}"
        L.append(f"| {t:.2f} | {rel_res[k]['R'] * 100:+.1f}% | {rel_res[k]['ci95'][0] * 100:+.1f}% s.d. {rel_res[k]['ci95'][1] * 100:+.1f}% |")
    g = rel_res["G_rel"]
    L.append(f"| G_rel | {g['G_rel'] * 100:+.1f} poin persen | {g['ci95'][0] * 100:+.1f} s.d. {g['ci95'][1] * 100:+.1f} |")
    L += ["", "Catatan:",
          "- dAP jenis utama dipotong pada 0 per model (aturan tidecv); dAP FalsePos/FalseNeg tidak dipotong. Jenis error "
          "TIDE tidak saling lepas dalam satuan AP: jumlah dAP tidak sama dengan selisih ke AP sempurna.",
          "- AP50 TIDE memakai pencocokan TIDE dan max_dets 100, sehingga berbeda dari mAP@50 hitung ulang (max_det 300).",
          "- Verdict memakai bahasa aturan keputusan penelitian, tetapi analisis ini pasca-hoc dan tanpa koreksi multiplisitas."]
    (out / "TIDE_IOU_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "tide_iou_results.json").write_text(json.dumps({"kontras": res, "relatif_exp1_exp0r": rel_res}, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

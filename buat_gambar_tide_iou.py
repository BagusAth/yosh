"""Gambar draf Bab 4 (analisis pasca-hoc butir 7): selisih AP per ambang IoU 0,50-0,95 pada validasi.

Sumber: stage_v2_ablation/analysis/tide_iou/tide_iou_results.json (analyze_tide_iou.py report). Titik = rata-rata
tiga seed, garis = IK 95% bootstrap klaster per grup foto (dirata-rata antar seed). Hanya membaca JSON; tidak
menghitung ulang apa pun. Keluaran: _gambar_bab4/gbr4_9_ap_per_iou.png (300 dpi).
Warna dan gaya sama dengan buat_gambar_test.py: slot kategorikal 1-2 palet acuan (biru, oranye; lolos validator CVD),
bentuk penanda berbeda agar tetap terbaca saat dicetak hitam-putih.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / "stage_v2_ablation/analysis/tide_iou/tide_iou_results.json").read_text(encoding="utf-8"))["kontras"]
SERI = [("EXP-1 − EXP-0R", "EXP-1 − EXP-0R (pembobotan skor kesulitan)", "#2a78d6", "o", -0.13),
        ("EXP-2 − EXP-0", "EXP-2 − EXP-0 (context crop)", "#eb6834", "s", 0.13)]
AMBANG = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def koma(x, _=None):
    s = f"{x:+.1f}" if abs(x) > 1e-9 else "0"
    return s.replace(".", ",").replace("-", "−")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=300)
    fig.patch.set_facecolor("white")
    for k, nama, c, m, off in SERI:
        for i, t in enumerate(AMBANG):
            r = R[k][f"AP{t}"]
            lo, hi = r["ci95"]
            ax.plot([i + off, i + off], [lo, hi], color=c, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(i + off, r["mean"], m, ms=6, color=c, mec="white", mew=1.1, zorder=3, label=nama if i == 0 else None)
    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_xticks(range(len(AMBANG)))
    ax.set_xticklabels([f"0,{t}" for t in AMBANG], color=INK2)
    ax.set_xlim(-0.6, len(AMBANG) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(koma))
    ax.set_xlabel("Ambang IoU", color=INK2)
    ax.set_ylabel("Selisih AP (poin)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    out = ROOT / "_gambar_bab4" / "gbr4_9_ap_per_iou.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("tersimpan:", out)


if __name__ == "__main__":
    main()

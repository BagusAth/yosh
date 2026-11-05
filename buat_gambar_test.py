"""Gambar 4.6 draf Bab 4: delapan kontras mAP@50-95, validasi vs test set (Fase 6).

Sumber: stage_v2_ablation/analysis/ablation_results.json (validasi, 2.567 citra) dan
stage_v2_fase6/analysis/ablation_results.json (test, 1.282 citra). Titik = rata-rata tiga seed,
garis = IK 95% bootstrap klaster per grup foto (dirata-rata antar seed). Hanya membaca JSON; tidak
menghitung ulang apa pun. Keluaran: _gambar_bab4/gbr4_8_validasi_vs_test.png (300 dpi).
Warna: slot kategorikal 1-2 palet acuan (biru, oranye; lolos validator CVD), bentuk penanda berbeda
agar tetap terbaca saat dicetak hitam-putih.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
VAL = json.loads((ROOT / "stage_v2_ablation/analysis/ablation_results.json").read_text(encoding="utf-8"))
TES = json.loads((ROOT / "stage_v2_fase6/analysis/ablation_results.json").read_text(encoding="utf-8"))
BARIS = [("EXP-0R − EXP-0", "EXP-0R − EXP-0"), ("EXP-1 − EXP-0R", "EXP-1 − EXP-0R (RQ1)"), ("EXP-1 − EXP-0", "EXP-1 − EXP-0"),
         ("EXP-2 − EXP-0", "EXP-2 − EXP-0 (RQ2)"), ("EXP-3 − EXP-0", "EXP-3 − EXP-0"), ("EXP-3 − EXP-2", "EXP-3 − EXP-2"),
         ("EXP-3 − EXP-1", "EXP-3 − EXP-1"), ("Interaksi", "Interaksi")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_VAL, C_TES = "#2a78d6", "#eb6834"


def koma(x, _=None):
    s = f"{x:+.1f}" if abs(x) > 1e-9 else "0"
    return s.replace(".", ",").replace("-", "−")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=300)
    fig.patch.set_facecolor("white")
    n = len(BARIS)
    for i, (k, lab) in enumerate(BARIS):
        y = n - 1 - i
        for d, off, c, m, nm in ((VAL, 0.17, C_VAL, "o", "Validasi (2.567 citra)"), (TES, -0.17, C_TES, "s", "Test (1.282 citra)")):
            mu, (lo, hi) = d[k]["mean"], d[k]["ci95"]
            ax.plot([lo, hi], [y + off, y + off], color=c, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(mu, y + off, m, ms=6.5, color=c, mec="white", mew=1.2, zorder=3, label=nm if i == 0 else None)
        sama = VAL[k]["verdict"] == TES[k]["verdict"]
        ax.text(1.02, y, "terkonfirmasi" if sama else "tidak terkonfirmasi", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color=INK2 if sama else INK, fontweight="normal" if sama else "bold")
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in reversed(BARIS)], color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_formatter(FuncFormatter(koma))
    ax.set_xlabel("Selisih mAP@50–95 (poin), rata-rata tiga seed dengan IK 95%", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    out = ROOT / "_gambar_bab4" / "gbr4_8_validasi_vs_test.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("tersimpan:", out)


if __name__ == "__main__":
    main()

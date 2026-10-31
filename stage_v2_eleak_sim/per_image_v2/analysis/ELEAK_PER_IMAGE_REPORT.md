# E-LEAK per citra — bootstrap klaster per grup foto
Tanggal: 2026-10-30 10:03
Iterasi bootstrap: 1000

## Estimasi titik (dihitung ulang dari prediksi per citra)

| Model/subset | Citra | mAP@50 | mAP@50–95 | F1 per citra | Recall per citra |
| --- | --- | --- | --- | --- | --- |
| sim/val_all | 2,567 | 42.84% | 32.15% | 47.06% | 55.19% |
| sim/val_leaked | 1,554 | 46.79% | 36.66% | 48.87% | 56.58% |
| sim/val_twinval | 432 | 46.39% | 36.50% | 45.46% | 55.14% |
| sim/val_single | 581 | 43.17% | 31.46% | 43.41% | 51.47% |
| sim/diag_A | 227 | 39.65% | 28.67% | 43.99% | 50.97% |
| sim/diag_B | 786 | 39.88% | 29.71% | 44.37% | 53.65% |
| group/val_all | 2,567 | 35.42% | 25.67% | 42.36% | 49.40% |
| group/val_twin | 1,972 | 35.34% | 26.14% | 41.51% | 47.97% |
| group/diag_A | 227 | 38.75% | 28.41% | 46.15% | 54.02% |

## Selisih mAP@50–95 (poin, rata-rata bootstrap dan IK 95%)

| Perbandingan | Rata-rata | IK 95% |
| --- | --- | --- |
| d_same (sim A - group A) | +0.14 | [-2.40, +2.59] |
| d_leak (sim bocor - sim kembar) | +0.41 | [-6.52, +8.63] |
| d_report (sim val_all - group val_all) | +6.39 | [+4.14, +8.56] |
| d_infl (d_report - d_same) | +6.25 | [+2.86, +9.59] |

## Selisih mAP@50 (poin, rata-rata bootstrap dan IK 95%)

| Perbandingan | Rata-rata | IK 95% |
| --- | --- | --- |
| d_same (sim A - group A) | +0.77 | [-2.56, +3.88] |
| d_leak (sim bocor - sim kembar) | +0.63 | [-6.62, +9.43] |
| d_report (sim val_all - group val_all) | +7.31 | [+4.75, +9.91] |
| d_infl (d_report - d_same) | +6.54 | [+2.39, +10.57] |

## Selisih F1 per citra (conf ≥ 0,25) (poin, rata-rata bootstrap dan IK 95%)

| Perbandingan | Rata-rata | IK 95% |
| --- | --- | --- |
| d_same (sim A - group A) | -2.17 | [-6.32, +1.67] |
| d_leak (sim bocor - sim kembar) | +3.36 | [-1.20, +7.78] |
| d_report (sim val_all - group val_all) | +4.67 | [+2.56, +6.60] |
| d_infl (d_report - d_same) | +6.84 | [+2.82, +11.44] |

Catatan: resampel per grup foto (pasangan kembaran ikut bersama); d_same berpasangan pada citra yang sama.

# E-LEAK per citra — bootstrap klaster per grup foto
Tanggal: 2026-10-29 10:05
Iterasi bootstrap: 1000

## Estimasi titik (dihitung ulang dari prediksi per citra)

| Model/subset | Citra | mAP@50 | mAP@50–95 | F1 per citra | Recall per citra |
| --- | --- | --- | --- | --- | --- |
| sim/val_all | 2,567 | 41.69% | 31.33% | 46.34% | 54.10% |
| sim/val_leaked | 1,554 | 45.24% | 35.46% | 47.90% | 55.25% |
| sim/val_twinval | 432 | 45.71% | 36.00% | 45.13% | 54.52% |
| sim/val_single | 581 | 41.76% | 30.53% | 43.06% | 50.69% |
| sim/diag_A | 227 | 38.53% | 28.04% | 44.17% | 50.75% |
| sim/diag_B | 786 | 39.04% | 29.10% | 43.88% | 52.79% |
| group/val_all | 2,567 | 34.06% | 24.71% | 41.51% | 48.20% |
| group/val_twin | 1,972 | 34.41% | 25.58% | 40.76% | 46.96% |
| group/diag_A | 227 | 36.51% | 26.64% | 45.02% | 51.99% |

## Selisih mAP@50–95 (poin, rata-rata bootstrap dan IK 95%)

| Perbandingan | Rata-rata | IK 95% |
| --- | --- | --- |
| d_same (sim A - group A) | +1.29 | [-1.77, +4.29] |
| d_leak (sim bocor - sim kembar) | -0.23 | [-7.03, +8.24] |
| d_report (sim val_all - group val_all) | +6.54 | [+4.23, +8.69] |
| d_infl (d_report - d_same) | +5.25 | [+1.55, +8.98] |

## Selisih mAP@50 (poin, rata-rata bootstrap dan IK 95%)

| Perbandingan | Rata-rata | IK 95% |
| --- | --- | --- |
| d_same (sim A - group A) | +1.86 | [-2.36, +5.71] |
| d_leak (sim bocor - sim kembar) | -0.20 | [-7.75, +8.80] |
| d_report (sim val_all - group val_all) | +7.53 | [+4.89, +10.16] |
| d_infl (d_report - d_same) | +5.66 | [+0.72, +10.50] |

## Selisih F1 per citra (conf ≥ 0,25) (poin, rata-rata bootstrap dan IK 95%)

| Perbandingan | Rata-rata | IK 95% |
| --- | --- | --- |
| d_same (sim A - group A) | -0.84 | [-5.12, +3.09] |
| d_leak (sim bocor - sim kembar) | +2.71 | [-1.90, +7.18] |
| d_report (sim val_all - group val_all) | +4.80 | [+2.64, +6.80] |
| d_infl (d_report - d_same) | +5.64 | [+1.14, +10.42] |

Catatan: resampel per grup foto (pasangan kembaran ikut bersama); d_same berpasangan pada citra yang sama.

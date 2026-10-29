# E-LEAK — uji efek kembaran pada citra yang sama (selisih-dalam-selisih)
Tanggal: 2026-10-29 03:10
Iterasi bootstrap: 1000; L_cv = 333 citra, A = 227 citra

| Subset | Model split acak | Model split per grup |
| --- | --- | --- |
| L_cv (kembaran dilihat model split acak): mAP@50–95 | 34.63% | 24.98% |
| L_cv (kembaran dilihat model split acak): F1 per citra | 47.31% | 42.69% |
| A (tidak terlihat kedua model): mAP@50–95 | 28.04% | 26.64% |
| A (tidak terlihat kedua model): F1 per citra | 44.17% | 45.02% |

| Selisih (poin) | Rata-rata | IK 95% |
| --- | --- | --- |
| mAP@50–95: sim − group pada L_cv | +9.54 | [+6.39, +13.30] |
| mAP@50–95: sim − group pada A | +1.30 | [-1.79, +4.24] |
| mAP@50–95: efek kembaran (L_cv − A) | +8.25 | [+3.87, +12.83] |
| mAP@50: sim − group pada L_cv | +9.86 | [+6.13, +14.16] |
| mAP@50: sim − group pada A | +1.85 | [-2.15, +5.85] |
| mAP@50: efek kembaran (L_cv − A) | +8.01 | [+2.59, +13.68] |
| F1 per citra: sim − group pada L_cv | +4.58 | [+2.19, +6.97] |
| F1 per citra: sim − group pada A | -0.81 | [-5.13, +3.33] |
| F1 per citra: efek kembaran (L_cv − A) | +5.39 | [+0.44, +10.22] |
| Recall per citra: sim − group pada L_cv | +5.89 | [+3.16, +8.66] |
| Recall per citra: sim − group pada A | -1.20 | [-6.47, +3.38] |
| Recall per citra: efek kembaran (L_cv − A) | +7.09 | [+1.75, +12.83] |

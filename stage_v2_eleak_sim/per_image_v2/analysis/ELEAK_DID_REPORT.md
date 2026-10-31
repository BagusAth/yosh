# E-LEAK — uji efek kembaran pada citra yang sama (selisih-dalam-selisih)
Tanggal: 2026-10-30 10:03
Iterasi bootstrap: 1000; L_cv = 333 citra, kontrol A = 227 citra

| Subset | Model split acak | Model split per grup |
| --- | --- | --- |
| L_cv (kembaran dilihat model split acak): mAP@50–95 | 34.99% | 25.62% |
| L_cv (kembaran dilihat model split acak): F1 per citra | 48.15% | 43.25% |
| A (tidak terlihat kedua model): mAP@50–95 | 28.67% | 28.41% |
| A (tidak terlihat kedua model): F1 per citra | 43.99% | 46.15% |

| Selisih (poin) | Rata-rata | IK 95% |
| --- | --- | --- |
| mAP@50–95: sim − group pada L_cv | +9.27 | [+6.52, +12.10] |
| mAP@50–95: sim − group pada A | +0.21 | [-2.24, +2.69] |
| mAP@50–95: efek kembaran (L_cv − A) | +9.06 | [+5.36, +12.53] |
| mAP@50: sim − group pada L_cv | +9.73 | [+6.66, +13.10] |
| mAP@50: sim − group pada A | +0.86 | [-2.52, +3.95] |
| mAP@50: efek kembaran (L_cv − A) | +8.87 | [+4.36, +13.16] |
| F1 per citra: sim − group pada L_cv | +4.86 | [+2.77, +6.92] |
| F1 per citra: sim − group pada A | -2.10 | [-6.19, +1.91] |
| F1 per citra: efek kembaran (L_cv − A) | +6.96 | [+2.38, +11.27] |
| Recall per citra: sim − group pada L_cv | +6.75 | [+4.38, +9.03] |
| Recall per citra: sim − group pada A | -3.00 | [-7.74, +1.27] |
| Recall per citra: efek kembaran (L_cv − A) | +9.76 | [+5.05, +15.14] |

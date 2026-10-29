# E-LEAK — uji efek kembaran pada citra yang sama (selisih-dalam-selisih)
Tanggal: 2026-10-29 03:12
Iterasi bootstrap: 1000; L_cv = 333 citra, kontrol TW = 108 citra

| Subset | Model split acak | Model split per grup |
| --- | --- | --- |
| L_cv (kembaran dilihat model split acak): mAP@50–95 | 34.63% | 24.98% |
| L_cv (kembaran dilihat model split acak): F1 per citra | 47.31% | 42.69% |
| TW (grup 2 salinan, keduanya di validasi split acak): mAP@50–95 | 39.32% | 37.54% |
| TW (grup 2 salinan, keduanya di validasi split acak): F1 per citra | 44.70% | 42.46% |

| Selisih (poin) | Rata-rata | IK 95% |
| --- | --- | --- |
| mAP@50–95: sim − group pada L_cv | +9.50 | [+6.43, +13.39] |
| mAP@50–95: sim − group pada TW | +1.24 | [-2.70, +5.33] |
| mAP@50–95: efek kembaran (L_cv − TW) | +8.27 | [+3.17, +13.82] |
| mAP@50: sim − group pada L_cv | +9.81 | [+6.21, +14.43] |
| mAP@50: sim − group pada TW | +3.06 | [-1.44, +7.93] |
| mAP@50: efek kembaran (L_cv − TW) | +6.74 | [+0.83, +12.99] |
| F1 per citra: sim − group pada L_cv | +4.64 | [+2.32, +6.89] |
| F1 per citra: sim − group pada TW | +2.32 | [-4.51, +8.38] |
| F1 per citra: efek kembaran (L_cv − TW) | +2.32 | [-4.44, +9.35] |
| Recall per citra: sim − group pada L_cv | +5.95 | [+3.20, +8.71] |
| Recall per citra: sim − group pada TW | +2.16 | [-5.93, +10.32] |
| Recall per citra: efek kembaran (L_cv − TW) | +3.78 | [-5.01, +12.38] |

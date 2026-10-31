# E-LEAK — uji efek kembaran pada citra yang sama (selisih-dalam-selisih)
Tanggal: 2026-10-30 10:03
Iterasi bootstrap: 1000; L_cv = 333 citra, kontrol TW = 108 citra

| Subset | Model split acak | Model split per grup |
| --- | --- | --- |
| L_cv (kembaran dilihat model split acak): mAP@50–95 | 34.99% | 25.62% |
| L_cv (kembaran dilihat model split acak): F1 per citra | 48.15% | 43.25% |
| TW (grup 2 salinan, keduanya di validasi split acak): mAP@50–95 | 40.53% | 41.51% |
| TW (grup 2 salinan, keduanya di validasi split acak): F1 per citra | 44.40% | 44.48% |

| Selisih (poin) | Rata-rata | IK 95% |
| --- | --- | --- |
| mAP@50–95: sim − group pada L_cv | +9.23 | [+6.62, +12.37] |
| mAP@50–95: sim − group pada TW | -1.82 | [-8.94, +3.02] |
| mAP@50–95: efek kembaran (L_cv − TW) | +11.05 | [+5.41, +18.15] |
| mAP@50: sim − group pada L_cv | +9.67 | [+6.77, +13.04] |
| mAP@50: sim − group pada TW | -0.34 | [-8.77, +5.25] |
| mAP@50: efek kembaran (L_cv − TW) | +10.01 | [+3.46, +18.07] |
| F1 per citra: sim − group pada L_cv | +4.90 | [+2.68, +7.09] |
| F1 per citra: sim − group pada TW | -0.02 | [-6.16, +5.41] |
| F1 per citra: efek kembaran (L_cv − TW) | +4.92 | [-0.91, +11.42] |
| Recall per citra: sim − group pada L_cv | +6.78 | [+4.30, +9.26] |
| Recall per citra: sim − group pada TW | -0.87 | [-7.87, +5.85] |
| Recall per citra: efek kembaran (L_cv − TW) | +7.65 | [+0.44, +15.01] |

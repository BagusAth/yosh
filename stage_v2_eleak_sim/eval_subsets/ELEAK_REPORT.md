# E-LEAK (RQ1 bagian A) — Inflasi metrik akibat split per citra setelah augmentasi
Tanggal: 2026-10-28 21:37:32

| Model | Subset | Citra | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | --- | --- | --- | --- | --- | --- |
| sim | val_all | 2,567 | 46.42% | 46.35% | 43.69% | 33.46% |
| sim | val_leaked | 1,554 | 48.06% | 49.45% | 47.45% | 38.35% |
| sim | val_twinval | 432 | 49.64% | 47.43% | 47.34% | 38.59% |
| sim | val_single | 581 | 47.50% | 46.00% | 44.08% | 32.43% |
| group | val_all | 2,567 | 40.37% | 41.19% | 35.83% | 26.20% |
| group | val_twin | 1,972 | 39.71% | 40.68% | 35.24% | 26.44% |

Selisih mAP@50–95 (poin):
- Dalam model simulasi, populasi sama: bocor − kembar-di-validasi = -0.23
- Angka yang akan dilaporkan peneliti: validasi split acak − validasi split per grup = +7.26
- Antar model, populasi sama: simulasi bocor − split per grup (grup 2 citra) = +11.91

Catatan: subset bocor dan kembar-di-validasi berasal dari populasi yang sama (grup 2 citra hasil augmentasi Roboflow),
sehingga selisih pertama paling bersih mengukur efek kebocoran. Interval kepercayaan bootstrap berpasangan dihitung di Fase 5.

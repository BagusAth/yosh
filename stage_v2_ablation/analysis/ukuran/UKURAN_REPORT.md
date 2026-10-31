# AP dan recall per ukuran objek (Fase 5 A1)
Tanggal: 2026-10-31 20:28 · 2,567 citra validasi bersih · 15 model · bootstrap klaster 1000 iterasi

Protokol COCO: kecil < 32², sedang 32²–96², besar > 96² piksel (citra 640 px); maxDets 100; replikasi numpy dari pycocotools.

## Jumlah objek GT per ukuran

| Ukuran | CBB | CBSD | CGM | CMD | HEALTHY | Total |
| --- | --- | --- | --- | --- | --- | --- |
| small | 29 | 13 | 42 | 166 | 111 | 361 |
| medium | 95 | 103 | 444 | 699 | 1111 | 2452 |
| large | 537 | 574 | 1546 | 1961 | 2218 | 6836 |

## Nilai per model (rata-rata tiga seed)

| Model | AP semua (COCO) | AP kecil | AP sedang | AP besar | AR kecil | Recall kecil conf ≥ 0,25 | Recall sedang | Recall besar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | 26.01 | 0.36 | 6.68 | 32.61 | 11.58 | 0.28 | 13.42 | 51.53 |
| EXP-0R | 25.68 | 0.35 | 6.61 | 32.12 | 11.73 | 0.00 | 12.81 | 51.66 |
| EXP-1 | 24.39 | 0.20 | 5.92 | 31.06 | 10.97 | 0.00 | 14.93 | 53.33 |
| EXP-2 | 25.97 | 0.42 | 6.51 | 32.40 | 10.28 | 0.00 | 10.39 | 50.47 |
| EXP-3 | 25.27 | 0.13 | 6.50 | 31.92 | 9.85 | 0.00 | 14.26 | 52.99 |

## Kontras: AP objek kecil (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | +0.10 / -0.13 / -0.01 | -0.01 | -0.14 s.d. +0.14 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | -0.14 / +0.01 / -0.29 | -0.14 | -0.45 s.d. +0.17 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | -0.05 / -0.12 / -0.30 | -0.16 | -0.42 s.d. +0.11 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | +0.18 / -0.02 / +0.01 | +0.06 | -0.22 s.d. +0.86 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | -0.11 / -0.23 / -0.34 | -0.23 | -0.52 s.d. -0.05 | menurunkan secara konsisten |
| EXP-3 − EXP-2 | -0.30 / -0.21 / -0.35 | -0.29 | -1.20 s.d. -0.04 | menurunkan secara konsisten |
| EXP-3 − EXP-1 | -0.06 / -0.11 / -0.04 | -0.07 | -0.37 s.d. +0.06 | tidak terbukti berbeda |
| Interaksi | -0.25 / -0.09 / -0.05 | -0.13 | -1.13 s.d. +0.19 | tidak terbukti berbeda |

## Kontras: AP objek sedang (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | -0.12 / +0.15 / -0.25 | -0.07 | -0.71 s.d. +0.66 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | -0.69 / -0.77 / -0.61 | -0.69 | -1.49 s.d. +0.23 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | -0.81 / -0.62 / -0.86 | -0.76 | -1.55 s.d. +0.08 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | +0.10 / +0.08 / -0.70 | -0.17 | -0.76 s.d. +0.52 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | +0.13 / +0.82 / -1.48 | -0.18 | -0.66 s.d. +0.47 | tidak terbukti berbeda |
| EXP-3 − EXP-2 | +0.02 / +0.74 / -0.78 | -0.00 | -0.73 s.d. +0.82 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | +0.94 / +1.44 / -0.62 | +0.59 | -0.08 s.d. +1.25 | tidak terbukti berbeda |
| Interaksi | +0.84 / +1.37 / +0.08 | +0.76 | -0.30 s.d. +1.64 | tidak terbukti berbeda |

## Kontras: AP objek besar (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | -0.27 / -1.03 / -0.15 | -0.48 | -1.03 s.d. +0.07 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | -0.96 / -0.93 / -1.28 | -1.06 | -1.77 s.d. -0.30 | menurunkan secara konsisten |
| EXP-1 − EXP-0 | -1.24 / -1.97 / -1.43 | -1.54 | -2.31 s.d. -0.77 | menurunkan secara konsisten |
| EXP-2 − EXP-0 | -0.18 / -0.35 / -0.08 | -0.20 | -0.73 s.d. +0.37 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | -0.92 / -0.66 / -0.48 | -0.69 | -1.34 s.d. -0.06 | menurunkan secara konsisten |
| EXP-3 − EXP-2 | -0.74 / -0.31 / -0.40 | -0.48 | -1.13 s.d. +0.15 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | +0.31 / +1.31 / +0.95 | +0.86 | +0.21 s.d. +1.54 | meningkatkan secara konsisten |
| Interaksi | +0.49 / +1.66 / +1.03 | +1.06 | +0.16 s.d. +1.98 | meningkatkan secara konsisten |

## Kontras: AR objek kecil (maxDets 100) (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | -1.27 / +1.27 / +0.46 | +0.15 | -1.26 s.d. +1.61 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | -0.10 / -1.58 / -0.60 | -0.76 | -2.71 s.d. +0.98 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | -1.37 / -0.32 / -0.14 | -0.61 | -2.66 s.d. +1.85 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | -1.80 / -1.13 / -0.96 | -1.30 | -3.85 s.d. +0.80 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | -1.72 / -1.26 / -2.20 | -1.73 | -3.30 s.d. -0.18 | menurunkan secara konsisten |
| EXP-3 − EXP-2 | +0.08 / -0.13 / -1.24 | -0.43 | -1.94 s.d. +1.21 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | -0.35 / -0.94 / -2.06 | -1.12 | -3.09 s.d. +0.72 | tidak terbukti berbeda |
| Interaksi | +1.45 / +0.19 / -1.10 | +0.18 | -2.87 s.d. +3.07 | tidak terbukti berbeda |

## Kontras: Recall objek kecil, conf ≥ 0,25 (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | +0.00 / -0.83 / +0.00 | -0.28 | -0.85 s.d. +0.00 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | +0.00 / -0.83 / +0.00 | -0.28 | -0.85 s.d. +0.00 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | +0.00 / -0.83 / +0.00 | -0.28 | -0.85 s.d. +0.00 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | +0.00 / -0.83 / +0.00 | -0.28 | -0.85 s.d. +0.00 | tidak terbukti berbeda |
| EXP-3 − EXP-2 | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| Interaksi | +0.00 / +0.83 / +0.00 | +0.28 | +0.00 s.d. +0.85 | tidak terbukti berbeda |

## AP objek kecil per kelas (rata-rata tiga seed)

| Model | CBB | CBSD | CGM | CMD | HEALTHY |
| --- | --- | --- | --- | --- | --- |
| EXP-0 | 0.47 | 0.17 | 0.37 | 0.30 | 0.48 |
| EXP-0R | 0.36 | 0.19 | 0.47 | 0.29 | 0.42 |
| EXP-1 | 0.65 | 0.09 | 0.04 | 0.05 | 0.19 |
| EXP-2 | 1.09 | 0.18 | 0.17 | 0.30 | 0.35 |
| EXP-3 | 0.26 | 0.08 | 0.04 | 0.10 | 0.17 |

Catatan: AP COCO memakai maxDets 100 dan definisi area COCO, sehingga sedikit berbeda dari mAP@50–95 utama (prediksi hingga 300 per citra). IK berasal dari bootstrap klaster per grup foto, dirata-rata antar seed.

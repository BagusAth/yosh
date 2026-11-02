# AP dan recall per ukuran objek (Fase 5 A1)
Tanggal: 2026-11-02 20:30 · 1,282 citra test · 15 model · bootstrap klaster 1000 iterasi

Protokol COCO: kecil < 32², sedang 32²–96², besar > 96² piksel (citra 640 px); maxDets 100; replikasi numpy dari pycocotools.

## Jumlah objek GT per ukuran

| Ukuran | CBB | CBSD | CGM | CMD | HEALTHY | Total |
| --- | --- | --- | --- | --- | --- | --- |
| small | 23 | 4 | 15 | 75 | 73 | 190 |
| medium | 66 | 47 | 198 | 418 | 675 | 1404 |
| large | 250 | 279 | 755 | 954 | 1257 | 3495 |

## Nilai per model (rata-rata tiga seed)

| Model | AP semua (COCO) | AP kecil | AP sedang | AP besar | AR kecil | Recall kecil conf ≥ 0,25 | Recall sedang | Recall besar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | 24.05 | 0.20 | 5.16 | 31.11 | 6.02 | 0.00 | 10.94 | 50.56 |
| EXP-0R | 24.00 | 0.25 | 5.69 | 30.71 | 6.61 | 0.00 | 11.99 | 50.67 |
| EXP-1 | 22.81 | 0.61 | 5.10 | 29.72 | 6.80 | 0.35 | 13.25 | 51.67 |
| EXP-2 | 24.48 | 0.26 | 5.58 | 31.38 | 4.61 | 0.00 | 9.21 | 49.58 |
| EXP-3 | 23.13 | 0.26 | 5.12 | 30.00 | 5.58 | 0.00 | 13.11 | 51.18 |

## Kontras: AP objek kecil (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | -0.03 / +0.15 / +0.02 | +0.05 | -0.34 s.d. +0.47 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | +0.19 / +0.53 / +0.38 | +0.37 | +0.01 s.d. +0.98 | meningkatkan secara konsisten |
| EXP-1 − EXP-0 | +0.16 / +0.68 / +0.40 | +0.41 | -0.09 s.d. +1.28 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | -0.00 / +0.28 / -0.08 | +0.06 | -0.31 s.d. +0.41 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | +0.09 / -0.06 / +0.15 | +0.06 | -0.29 s.d. +0.46 | tidak terbukti berbeda |
| EXP-3 − EXP-2 | +0.09 / -0.33 / +0.23 | -0.00 | -0.19 s.d. +0.27 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | -0.07 / -0.73 / -0.25 | -0.35 | -1.03 s.d. +0.10 | tidak terbukti berbeda |
| Interaksi | -0.07 / -1.01 / -0.17 | -0.41 | -1.25 s.d. +0.14 | tidak terbukti berbeda |

## Kontras: AP objek sedang (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | +0.63 / +0.24 / +0.74 | +0.53 | +0.07 s.d. +1.17 | meningkatkan secara konsisten |
| EXP-1 − EXP-0R | -0.56 / -0.13 / -1.07 | -0.59 | -1.25 s.d. +0.07 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | +0.06 / +0.10 / -0.33 | -0.06 | -0.72 s.d. +0.80 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | +0.37 / +0.38 / +0.51 | +0.42 | -0.05 s.d. +1.13 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | -0.72 / +0.24 / +0.37 | -0.04 | -0.68 s.d. +0.64 | tidak terbukti berbeda |
| EXP-3 − EXP-2 | -1.09 / -0.14 / -0.13 | -0.45 | -1.16 s.d. +0.09 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | -0.78 / +0.14 / +0.71 | +0.02 | -0.55 s.d. +0.51 | tidak terbukti berbeda |
| Interaksi | -1.15 / -0.24 / +0.20 | -0.40 | -1.31 s.d. +0.24 | tidak terbukti berbeda |

## Kontras: AP objek besar (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | -0.34 / +0.14 / -1.00 | -0.40 | -1.13 s.d. +0.36 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | -1.62 / -1.35 / +0.02 | -0.98 | -1.86 s.d. -0.05 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | -1.96 / -1.21 / -0.99 | -1.38 | -2.36 s.d. -0.43 | menurunkan secara konsisten |
| EXP-2 − EXP-0 | -0.16 / +0.43 / +0.55 | +0.27 | -0.62 s.d. +1.01 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | -0.80 / -1.40 / -1.11 | -1.10 | -1.96 s.d. -0.32 | menurunkan secara konsisten |
| EXP-3 − EXP-2 | -0.64 / -1.83 / -1.66 | -1.38 | -2.22 s.d. -0.51 | menurunkan secara konsisten |
| EXP-3 − EXP-1 | +1.16 / -0.20 / -0.13 | +0.28 | -0.59 s.d. +1.00 | tidak terbukti berbeda |
| Interaksi | +1.32 / -0.62 / -0.67 | +0.01 | -1.03 s.d. +1.11 | tidak terbukti berbeda |

## Kontras: AR objek kecil (maxDets 100) (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | +1.16 / +1.50 / -0.90 | +0.59 | -5.25 s.d. +4.09 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | -0.48 / +2.80 / -1.76 | +0.19 | -3.19 s.d. +5.56 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | +0.68 / +4.30 / -2.66 | +0.78 | -1.14 s.d. +2.64 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | -0.16 / -0.87 / -3.21 | -1.41 | -6.10 s.d. +1.00 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | +0.87 / +1.09 / -3.28 | -0.44 | -5.50 s.d. +2.57 | tidak terbukti berbeda |
| EXP-3 − EXP-2 | +1.02 / +1.96 / -0.08 | +0.97 | -0.29 s.d. +2.87 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | +0.18 / -3.21 / -0.62 | -1.22 | -5.64 s.d. +1.39 | tidak terbukti berbeda |
| Interaksi | +0.34 / -2.34 / +2.58 | +0.19 | -1.20 s.d. +2.25 | tidak terbukti berbeda |

## Kontras: Recall objek kecil, conf ≥ 0,25 (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Kontras | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | +0.00 / +0.53 / +0.53 | +0.35 | +0.00 s.d. +1.10 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | +0.00 / +0.53 / +0.53 | +0.35 | +0.00 s.d. +1.10 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| EXP-3 − EXP-2 | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 s.d. +0.00 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | +0.00 / -0.53 / -0.53 | -0.35 | -1.10 s.d. +0.00 | tidak terbukti berbeda |
| Interaksi | +0.00 / -0.53 / -0.53 | -0.35 | -1.10 s.d. +0.00 | tidak terbukti berbeda |

## AP objek kecil per kelas (rata-rata tiga seed)

| Model | CBB | CBSD | CGM | CMD | HEALTHY |
| --- | --- | --- | --- | --- | --- |
| EXP-0 | 0.35 | 0.02 | 0.00 | 0.15 | 0.47 |
| EXP-0R | 0.10 | 0.10 | 0.01 | 0.10 | 0.93 |
| EXP-1 | 0.34 | 0.18 | 0.00 | 0.23 | 2.32 |
| EXP-2 | 0.08 | 0.13 | 0.00 | 0.42 | 0.69 |
| EXP-3 | 0.25 | 0.17 | 0.00 | 0.22 | 0.66 |

Catatan: AP COCO memakai maxDets 100 dan definisi area COCO, sehingga sedikit berbeda dari mAP@50–95 utama (prediksi hingga 300 per citra). IK berasal dari bootstrap klaster per grup foto, dirata-rata antar seed.

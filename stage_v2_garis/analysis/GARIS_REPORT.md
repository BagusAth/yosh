# Uji jalan pintas pinggiran bergaris — hasil
Tanggal: 2026-11-01 10:03 · model EXP-0, EXP-1, EXP-2 × seed 42/123/2026 · bootstrap klaster per grup 1000 iterasi

## Himpunan citra dan cek manipulasi

- Set G: 1,183 citra validasi tanpa baji hitam dan tanpa garis terdeteksi. Subset: G_tr 784, G_cb 390, G_tr_sens 349, G_cb_sens 243.
- Luas isian median 15.5% (p90 21.5%); citra dengan kotak GT tertutup ≥ 30%: 23.8%.
- Detektor petak: G_garis terdeteksi bergaris 76.3%, G_rata 9.4%; kuartil petak bergaris pada G_garis yang terdeteksi 3/5/6 (garis asli di validasi: 3/5/7).

## Nilai per varian (rata-rata tiga seed)

**EXP-0, G_tr** (784 citra)

| Varian | porsi_CBB | porsi_vt | map_train | ap_CGM | ap_CMD | ap_HEALTHY | map |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G_asli | 3.48 | 5.70 | 24.82 | 27.08 | 20.80 | 26.59 | 24.82 |
| G_garis | 4.51 | 7.00 | 19.38 | 23.01 | 16.78 | 18.34 | 19.38 |
| G_rata | 3.16 | 5.21 | 19.24 | 23.18 | 16.68 | 17.85 | 19.24 |

**EXP-0, G_cb** (390 citra)

| Varian | porsi_CBB | porsi_vt | ap_CBB | ap_CBSD | map |
| --- | --- | --- | --- | --- | --- |
| G_asli | 39.36 | 83.08 | 30.59 | 32.01 | 31.30 |
| G_garis | 46.11 | 87.61 | 28.23 | 28.12 | 28.18 |
| G_rata | 38.91 | 80.26 | 27.26 | 28.99 | 28.13 |

**EXP-1, G_tr** (784 citra)

| Varian | porsi_CBB | porsi_vt | map_train | ap_CGM | ap_CMD | ap_HEALTHY | map |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G_asli | 2.43 | 5.36 | 23.70 | 26.54 | 20.08 | 24.46 | 23.70 |
| G_garis | 4.09 | 7.02 | 18.78 | 22.54 | 16.31 | 17.50 | 18.78 |
| G_rata | 2.35 | 4.84 | 18.59 | 22.61 | 16.26 | 16.90 | 18.59 |

**EXP-1, G_cb** (390 citra)

| Varian | porsi_CBB | porsi_vt | ap_CBB | ap_CBSD | map |
| --- | --- | --- | --- | --- | --- |
| G_asli | 36.12 | 78.28 | 27.96 | 30.17 | 29.07 |
| G_garis | 44.18 | 83.95 | 25.56 | 26.63 | 26.10 |
| G_rata | 37.21 | 77.20 | 24.43 | 26.48 | 25.45 |

**EXP-2, G_tr** (784 citra)

| Varian | porsi_CBB | porsi_vt | map_train | ap_CGM | ap_CMD | ap_HEALTHY | map |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G_asli | 3.04 | 5.45 | 24.75 | 27.26 | 20.76 | 26.23 | 24.75 |
| G_garis | 4.71 | 7.21 | 19.37 | 22.74 | 16.74 | 18.62 | 19.37 |
| G_rata | 2.77 | 4.89 | 19.06 | 23.12 | 16.65 | 17.42 | 19.06 |

**EXP-2, G_cb** (390 citra)

| Varian | porsi_CBB | porsi_vt | ap_CBB | ap_CBSD | map |
| --- | --- | --- | --- | --- | --- |
| G_asli | 39.34 | 81.29 | 29.21 | 32.65 | 30.93 |
| G_garis | 46.23 | 86.56 | 27.39 | 28.35 | 27.87 |
| G_rata | 39.11 | 78.61 | 26.14 | 28.55 | 27.34 |

porsi_CBB = rata-rata per citra dari bagian massa keyakinan (prediksi conf ≥ 0,25) yang jatuh ke CBB; porsi_vt = ke CBB/CBSD; map_train = rata-rata AP CGM/CMD/HEALTHY. AP dalam poin persen (AP@50–95).

## Kontras (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Model | Peran | Kontras | Subset | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | primer | Garis − rata: porsi ke CBB | G_tr | +1.50 / +1.32 / +1.21 | +1.35 | +0.87 s.d. +1.86 | naik, terbukti |
| EXP-0 | primer | Garis − rata: rata-rata AP CGM/CMD/HEALTHY | G_tr | +0.22 / -0.10 / +0.31 | +0.14 | -0.18 s.d. +0.42 | tidak terbukti berbeda |
| EXP-0 | pendukung | Garis − rata: porsi ke CBB/CBSD | G_tr | +1.68 / +1.95 / +1.74 | +1.79 | +1.20 s.d. +2.43 | naik, terbukti |
| EXP-0 | sekunder | Garis − rata: AP CBB | G_cb | +0.96 / +1.06 / +0.89 | +0.97 | -0.06 s.d. +2.16 | tidak terbukti berbeda |
| EXP-0 | sekunder | Garis − rata: AP CBSD | G_cb | -0.80 / -0.22 / -1.59 | -0.87 | -1.65 s.d. +0.07 | tidak terbukti berbeda |
| EXP-0 | sekunder | Garis − rata: porsi ke CBB/CBSD (citra CBB/CBSD) | G_cb | +9.48 / +5.46 / +7.11 | +7.35 | +5.77 s.d. +9.19 | naik, terbukti |
| EXP-0 | deskriptif | Garis − asli: porsi ke CBB | G_tr | +1.25 / +0.84 / +1.00 | +1.03 | +0.49 s.d. +1.62 | naik, terbukti |
| EXP-0 | deskriptif | Rata − asli: porsi ke CBB | G_tr | -0.26 / -0.48 / -0.21 | -0.32 | -0.69 s.d. +0.06 | tidak terbukti berbeda |
| EXP-0 | deskriptif | Garis − asli: rata-rata AP CGM/CMD/HEALTHY | G_tr | -5.49 / -5.54 / -5.30 | -5.44 | -6.34 s.d. -4.65 | turun, terbukti |
| EXP-0 | deskriptif | Rata − asli: rata-rata AP CGM/CMD/HEALTHY | G_tr | -5.71 / -5.44 / -5.60 | -5.58 | -6.41 s.d. -4.84 | turun, terbukti |
| EXP-0 | deskriptif | Garis − asli: AP CBB | G_cb | -3.03 / -1.74 / -2.29 | -2.35 | -3.77 s.d. -0.86 | turun, terbukti |
| EXP-0 | deskriptif | Rata − asli: AP CBB | G_cb | -3.99 / -2.80 / -3.19 | -3.33 | -4.95 s.d. -1.77 | turun, terbukti |
| EXP-1 | primer | Garis − rata: porsi ke CBB | G_tr | +2.07 / +1.26 / +1.89 | +1.74 | +1.19 s.d. +2.38 | naik, terbukti |
| EXP-1 | primer | Garis − rata: rata-rata AP CGM/CMD/HEALTHY | G_tr | +0.05 / +0.47 / +0.07 | +0.20 | -0.01 s.d. +0.45 | tidak terbukti berbeda |
| EXP-1 | pendukung | Garis − rata: porsi ke CBB/CBSD | G_tr | +2.73 / +1.91 / +1.91 | +2.18 | +1.55 s.d. +2.86 | naik, terbukti |
| EXP-1 | sekunder | Garis − rata: AP CBB | G_cb | +1.62 / +0.71 / +1.07 | +1.13 | +0.23 s.d. +2.27 | naik, terbukti |
| EXP-1 | sekunder | Garis − rata: AP CBSD | G_cb | +0.33 / -0.05 / +0.18 | +0.15 | -0.49 s.d. +0.82 | tidak terbukti berbeda |
| EXP-1 | sekunder | Garis − rata: porsi ke CBB/CBSD (citra CBB/CBSD) | G_cb | +7.50 / +8.16 / +4.59 | +6.75 | +5.24 s.d. +8.48 | naik, terbukti |
| EXP-1 | deskriptif | Garis − asli: porsi ke CBB | G_tr | +1.85 / +1.34 / +1.81 | +1.67 | +1.10 s.d. +2.33 | naik, terbukti |
| EXP-1 | deskriptif | Rata − asli: porsi ke CBB | G_tr | -0.22 / +0.08 / -0.08 | -0.07 | -0.45 s.d. +0.25 | tidak terbukti berbeda |
| EXP-1 | deskriptif | Garis − asli: rata-rata AP CGM/CMD/HEALTHY | G_tr | -5.29 / -4.73 / -4.72 | -4.91 | -5.69 s.d. -4.20 | turun, terbukti |
| EXP-1 | deskriptif | Rata − asli: rata-rata AP CGM/CMD/HEALTHY | G_tr | -5.34 / -5.20 / -4.79 | -5.11 | -5.87 s.d. -4.43 | turun, terbukti |
| EXP-1 | deskriptif | Garis − asli: AP CBB | G_cb | -1.62 / -3.12 / -2.46 | -2.40 | -3.64 s.d. -0.98 | turun, terbukti |
| EXP-1 | deskriptif | Rata − asli: AP CBB | G_cb | -3.23 / -3.83 / -3.54 | -3.53 | -5.10 s.d. -2.11 | turun, terbukti |
| EXP-2 | primer | Garis − rata: porsi ke CBB | G_tr | +2.20 / +2.14 / +1.49 | +1.94 | +1.37 s.d. +2.71 | naik, terbukti |
| EXP-2 | primer | Garis − rata: rata-rata AP CGM/CMD/HEALTHY | G_tr | +0.35 / +0.28 / +0.29 | +0.31 | +0.04 s.d. +0.58 | naik, terbukti |
| EXP-2 | pendukung | Garis − rata: porsi ke CBB/CBSD | G_tr | +2.59 / +2.37 / +2.00 | +2.32 | +1.67 s.d. +3.12 | naik, terbukti |
| EXP-2 | sekunder | Garis − rata: AP CBB | G_cb | +1.30 / +1.16 / +1.29 | +1.25 | +0.19 s.d. +2.38 | naik, terbukti |
| EXP-2 | sekunder | Garis − rata: AP CBSD | G_cb | -0.20 / -0.46 / +0.05 | -0.20 | -0.91 s.d. +0.63 | tidak terbukti berbeda |
| EXP-2 | sekunder | Garis − rata: porsi ke CBB/CBSD (citra CBB/CBSD) | G_cb | +9.74 / +7.73 / +6.36 | +7.94 | +6.23 s.d. +9.86 | naik, terbukti |
| EXP-2 | deskriptif | Garis − asli: porsi ke CBB | G_tr | +2.08 / +1.62 / +1.29 | +1.67 | +1.13 s.d. +2.26 | naik, terbukti |
| EXP-2 | deskriptif | Rata − asli: porsi ke CBB | G_tr | -0.12 / -0.51 / -0.20 | -0.28 | -0.73 s.d. +0.15 | tidak terbukti berbeda |
| EXP-2 | deskriptif | Garis − asli: rata-rata AP CGM/CMD/HEALTHY | G_tr | -5.19 / -5.35 / -5.60 | -5.38 | -6.19 s.d. -4.66 | turun, terbukti |
| EXP-2 | deskriptif | Rata − asli: rata-rata AP CGM/CMD/HEALTHY | G_tr | -5.54 / -5.64 / -5.89 | -5.69 | -6.51 s.d. -4.98 | turun, terbukti |
| EXP-2 | deskriptif | Garis − asli: AP CBB | G_cb | -2.27 / -1.37 / -1.84 | -1.83 | -3.38 s.d. -0.26 | turun, terbukti |
| EXP-2 | deskriptif | Rata − asli: AP CBB | G_cb | -3.57 / -2.52 / -3.14 | -3.08 | -4.76 s.d. -1.65 | turun, terbukti |

## Selisih efek garis − rata terhadap EXP-0 (berpasangan per seed, resampel sama)

| Kontras | Subset | Metrik | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-1 − EXP-0 | G_tr | porsi_CBB | +0.57 / -0.06 / +0.68 | +0.40 | -0.02 s.d. +0.84 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | G_tr | map_train | -0.17 / +0.56 / -0.24 | +0.05 | -0.13 s.d. +0.31 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | G_cb | ap_CBB | +0.66 / -0.35 / +0.18 | +0.16 | -0.74 s.d. +1.05 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | G_cb | porsi_vt | -1.98 / +2.70 / -2.52 | -0.60 | -2.11 s.d. +0.84 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | G_tr | porsi_CBB | +0.69 / +0.82 / +0.27 | +0.60 | +0.11 s.d. +1.12 | naik, terbukti |
| EXP-2 − EXP-0 | G_tr | map_train | +0.13 / +0.38 / -0.02 | +0.16 | -0.04 s.d. +0.41 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | G_cb | ap_CBB | +0.34 / +0.09 / +0.40 | +0.28 | -0.56 s.d. +1.01 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | G_cb | porsi_vt | +0.26 / +2.28 / -0.75 | +0.59 | -1.01 s.d. +2.12 | tidak terbukti berbeda |

## Prediksi teratas CBB per citra (deskriptif, rata-rata tiga seed)

| Model | Subset | Varian | % citra |
| --- | --- | --- | --- |
| EXP-0 | G_tr | G_asli | 3.4 |
| EXP-0 | G_tr | G_garis | 4.7 |
| EXP-0 | G_tr | G_rata | 3.1 |
| EXP-0 | G_cb | G_asli | 39.4 |
| EXP-0 | G_cb | G_garis | 45.0 |
| EXP-0 | G_cb | G_rata | 38.2 |
| EXP-1 | G_tr | G_asli | 2.3 |
| EXP-1 | G_tr | G_garis | 4.0 |
| EXP-1 | G_tr | G_rata | 2.7 |
| EXP-1 | G_cb | G_asli | 36.0 |
| EXP-1 | G_cb | G_garis | 43.2 |
| EXP-1 | G_cb | G_rata | 36.6 |
| EXP-2 | G_tr | G_asli | 3.2 |
| EXP-2 | G_tr | G_garis | 4.9 |
| EXP-2 | G_tr | G_rata | 3.0 |
| EXP-2 | G_cb | G_asli | 38.5 |
| EXP-2 | G_cb | G_garis | 45.0 |
| EXP-2 | G_cb | G_rata | 38.3 |

## Sensitivitas garis − rata (estimasi titik, rata-rata tiga seed)

| Model | Subset | Metrik | Selisih |
| --- | --- | --- | --- |
| EXP-0 | G_tr_sens (349) | porsi_CBB | +1.69 |
| EXP-0 | G_tr_sens (349) | map_train | +0.20 |
| EXP-0 | G_cb_sens (243) | ap_CBB | +0.52 |
| EXP-0 | G_cb_sens (243) | porsi_vt | +6.58 |
| EXP-1 | G_tr_sens (349) | porsi_CBB | +2.04 |
| EXP-1 | G_tr_sens (349) | map_train | +0.35 |
| EXP-1 | G_cb_sens (243) | ap_CBB | +0.73 |
| EXP-1 | G_cb_sens (243) | porsi_vt | +5.71 |
| EXP-2 | G_tr_sens (349) | porsi_CBB | +2.38 |
| EXP-2 | G_tr_sens (349) | map_train | +0.63 |
| EXP-2 | G_cb_sens (243) | ap_CBB | +1.19 |
| EXP-2 | G_cb_sens (243) | porsi_vt | +7.83 |

Catatan: kotak GT tidak diubah; garis dan warna rata menutup area yang sama dengan warna rata-rata yang sama, sehingga kontras garis − rata hanya menguji tekstur garis. IK dari bootstrap klaster per grup foto (resampel sama untuk semua model dan varian), dirata-rata antar seed.

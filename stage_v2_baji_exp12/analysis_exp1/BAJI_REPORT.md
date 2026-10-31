# Uji jalan pintas baji hitam — hasil
Tanggal: 2026-10-31 20:35 · model EXP-1 seed 42/123/2026 · bootstrap klaster per grup 1000 iterasi

## Himpunan citra

- Set N (tanpa baji): 1,632 citra; subset primer N berkelas dominan CBB/CBSD: 620 citra.
- Set W (berbaji): 935 citra; 80 di antaranya hanya terdeteksi lewat estimasi sudut.
- Kelas dominan set W: CGM 301, CMD 291, HEALTHY 290, CBSD 45, (tanpa kotak) 6, CBB 2
- Kelas dominan set N: HEALTHY 338, CMD 333, CGM 329, CBB 310, CBSD 310, (tanpa kotak) 12

Objek GT per kelas: N_cb CBB 659, CBSD 594; W_semua CBB 2, CBSD 96, CGM 974, CMD 1339, HEALTHY 1562. AP CBB di set W tidak dilaporkan (hanya 2 objek).

## Nilai per varian (rata-rata tiga seed)

**N_cb** (620 citra)

| Varian | map | ap_CBB | ap_CBSD | porsi_train | porsi_vt |
| --- | --- | --- | --- | --- | --- |
| N_asli | 27.20 | 25.21 | 29.20 | 19.49 | 80.51 |
| N_baji_hitam | 25.27 | 23.18 | 27.37 | 32.71 | 67.29 |
| N_baji_rata | 26.84 | 24.98 | 28.71 | 20.54 | 79.46 |

**W_semua** (935 citra)

| Varian | map_train | ap_CGM | ap_CMD | ap_HEALTHY | ap_CBSD | porsi_vt | porsi_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W_asli | 26.07 | 26.55 | 21.78 | 29.87 | 38.17 | 6.96 | 93.04 |
| W_rata | 25.04 | 26.47 | 21.44 | 27.20 | 37.51 | 8.92 | 91.08 |
| W_inpaint | 24.92 | 26.36 | 21.41 | 26.99 | 36.53 | 9.28 | 90.72 |

**N_semua** (1,632 citra)

| Varian | map | ap_CBB | ap_CBSD | ap_CGM | ap_CMD | ap_HEALTHY | porsi_vt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N_asli | 23.38 | 22.11 | 27.69 | 24.56 | 19.51 | 23.03 | 34.03 |
| N_baji_hitam | 22.49 | 20.44 | 26.14 | 24.08 | 18.75 | 23.05 | 26.96 |
| N_baji_rata | 23.00 | 22.00 | 27.24 | 24.54 | 18.99 | 22.24 | 33.30 |

Porsi = rata-rata per citra dari bagian massa keyakinan (prediksi conf ≥ 0,25) yang jatuh ke kelompok kelas tersebut; porsi_train = CGM/CMD/HEALTHY, porsi_vt = CBB/CBSD. AP dalam poin persen (AP@50–95); map_train = rata-rata AP CGM/CMD/HEALTHY.

## Kontras (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Peran | Kontras | Subset | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| primer | Hitam − rata: AP CBB | N_cb | -1.55 / -1.87 / -2.00 | -1.80 | -2.60 s.d. -1.05 | turun, terbukti |
| primer | Hitam − rata: AP CBSD | N_cb | -1.21 / -1.48 / -1.33 | -1.34 | -2.00 s.d. -0.82 | turun, terbukti |
| primer | Hitam − rata: porsi ke CGM/CMD/HEALTHY | N_cb | +11.58 / +12.80 / +12.16 | +12.18 | +10.58 s.d. +13.85 | naik, terbukti |
| pendukung | Hitam − rata: mAP subset | N_cb | -1.38 / -1.67 / -1.66 | -1.57 | -2.09 s.d. -1.10 | turun, terbukti |
| deskriptif | Hitam − asli: AP CBB (efek total) | N_cb | -1.55 / -2.52 / -2.01 | -2.03 | -2.95 s.d. -1.10 | turun, terbukti |
| deskriptif | Hitam − asli: AP CBSD (efek total) | N_cb | -1.67 / -1.56 / -2.28 | -1.84 | -2.68 s.d. -1.08 | turun, terbukti |
| deskriptif | Hitam − asli: porsi ke CGM/CMD/HEALTHY | N_cb | +13.23 / +12.96 / +13.49 | +13.23 | +11.51 s.d. +14.94 | naik, terbukti |
| deskriptif | Rata − asli: AP CBB (oklusi) | N_cb | -0.00 / -0.65 / -0.01 | -0.22 | -0.63 s.d. +0.20 | tidak terbukti berbeda |
| deskriptif | Rata − asli: AP CBSD (oklusi) | N_cb | -0.47 / -0.08 / -0.95 | -0.50 | -1.10 s.d. +0.07 | tidak terbukti berbeda |
| deskriptif | Rata − asli: porsi ke CGM/CMD/HEALTHY | N_cb | +1.65 / +0.16 / +1.34 | +1.05 | +0.37 s.d. +1.71 | naik, terbukti |
| sekunder | Asli − rata: porsi ke CBB/CBSD | W_semua | -1.75 / -2.16 / -1.97 | -1.96 | -2.52 s.d. -1.45 | turun, terbukti |
| sekunder | Asli − inpaint: porsi ke CBB/CBSD | W_semua | -2.33 / -1.94 / -2.71 | -2.33 | -2.95 s.d. -1.75 | turun, terbukti |
| sekunder | Asli − rata: rata-rata AP CGM/CMD/HEALTHY | W_semua | +1.12 / +0.75 / +1.22 | +1.03 | +0.76 s.d. +1.29 | naik, terbukti |
| sekunder | Asli − inpaint: rata-rata AP CGM/CMD/HEALTHY | W_semua | +1.13 / +0.97 / +1.34 | +1.15 | +0.82 s.d. +1.45 | naik, terbukti |
| sekunder | Asli − rata: AP CGM | W_semua | -0.21 / +0.07 / +0.38 | +0.08 | -0.27 s.d. +0.39 | tidak terbukti berbeda |
| sekunder | Asli − rata: AP CMD | W_semua | +0.35 / +0.30 / +0.37 | +0.34 | +0.07 s.d. +0.63 | naik, terbukti |
| sekunder | Asli − rata: AP HEALTHY | W_semua | +3.22 / +1.88 / +2.92 | +2.67 | +1.94 s.d. +3.34 | naik, terbukti |
| sekunder | Asli − inpaint: AP CGM | W_semua | +0.04 / +0.14 / +0.39 | +0.19 | -0.21 s.d. +0.54 | tidak terbukti berbeda |
| sekunder | Asli − inpaint: AP CMD | W_semua | +0.36 / +0.56 / +0.18 | +0.36 | +0.00 s.d. +0.73 | naik, terbukti |
| sekunder | Asli − inpaint: AP HEALTHY | W_semua | +3.00 / +2.22 / +3.44 | +2.88 | +2.08 s.d. +3.68 | naik, terbukti |

## Perpindahan kelas teratas per citra (deskriptif, rata-rata tiga seed)

Persentase citra yang prediksi berkeyakinan tertinggi berada di kelompok CGM/CMD/HEALTHY.

| Subset | Varian | % citra |
| --- | --- | --- |
| N_cb | N_asli | 19.9 |
| N_cb | N_baji_hitam | 32.5 |
| N_cb | N_baji_rata | 20.9 |
| W_semua | W_asli | 92.8 |
| W_semua | W_rata | 90.9 |
| W_semua | W_inpaint | 90.6 |

## Sensitivitas (estimasi titik, rata-rata tiga seed)

| Subset | Kontras | Metrik | Selisih |
| --- | --- | --- | --- |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | ap_CBB | -1.86 |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | ap_CBSD | -1.37 |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | porsi_train | +12.44 |
| W_sens (883) | W_asli − W_rata | porsi_vt | -1.92 |
| W_sens (883) | W_asli − W_inpaint | porsi_vt | -2.29 |

Catatan: kotak GT tidak diubah oleh intervensi; baji dan warna rata menutup area yang sama, sehingga kontras hitam − rata mengendalikan oklusi. IK berasal dari bootstrap klaster per grup foto (resampel sama untuk semua model dan varian), dirata-rata antar seed.

# Uji jalan pintas baji hitam — hasil
Tanggal: 2026-10-31 20:40 · model EXP-2 seed 42/123/2026 · bootstrap klaster per grup 1000 iterasi

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
| N_asli | 30.41 | 27.90 | 32.93 | 17.05 | 82.95 |
| N_baji_hitam | 28.56 | 25.78 | 31.33 | 29.67 | 70.33 |
| N_baji_rata | 30.07 | 27.55 | 32.60 | 17.29 | 82.71 |

**W_semua** (935 citra)

| Varian | map_train | ap_CGM | ap_CMD | ap_HEALTHY | ap_CBSD | porsi_vt | porsi_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W_asli | 27.01 | 27.25 | 22.26 | 31.52 | 33.04 | 7.77 | 92.23 |
| W_rata | 25.98 | 27.44 | 21.77 | 28.73 | 32.16 | 10.12 | 89.88 |
| W_inpaint | 25.90 | 27.37 | 21.82 | 28.52 | 31.28 | 10.52 | 89.48 |

**N_semua** (1,632 citra)

| Varian | map | ap_CBB | ap_CBSD | ap_CGM | ap_CMD | ap_HEALTHY | porsi_vt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N_asli | 25.19 | 25.17 | 30.76 | 25.64 | 19.95 | 24.42 | 34.64 |
| N_baji_hitam | 24.29 | 23.39 | 29.72 | 24.99 | 19.23 | 24.13 | 28.06 |
| N_baji_rata | 24.78 | 24.83 | 30.53 | 25.77 | 19.33 | 23.42 | 34.46 |

Porsi = rata-rata per citra dari bagian massa keyakinan (prediksi conf ≥ 0,25) yang jatuh ke kelompok kelas tersebut; porsi_train = CGM/CMD/HEALTHY, porsi_vt = CBB/CBSD. AP dalam poin persen (AP@50–95); map_train = rata-rata AP CGM/CMD/HEALTHY.

## Kontras (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Peran | Kontras | Subset | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| primer | Hitam − rata: AP CBB | N_cb | -1.92 / -1.61 / -1.75 | -1.76 | -2.55 s.d. -0.97 | turun, terbukti |
| primer | Hitam − rata: AP CBSD | N_cb | -0.92 / -1.32 / -1.57 | -1.27 | -1.82 s.d. -0.74 | turun, terbukti |
| primer | Hitam − rata: porsi ke CGM/CMD/HEALTHY | N_cb | +11.63 / +11.86 / +13.63 | +12.37 | +10.65 s.d. +14.16 | naik, terbukti |
| pendukung | Hitam − rata: mAP subset | N_cb | -1.42 / -1.46 / -1.66 | -1.52 | -2.00 s.d. -1.05 | turun, terbukti |
| deskriptif | Hitam − asli: AP CBB (efek total) | N_cb | -2.01 / -1.93 / -2.40 | -2.11 | -3.07 s.d. -1.14 | turun, terbukti |
| deskriptif | Hitam − asli: AP CBSD (efek total) | N_cb | -1.29 / -1.77 / -1.75 | -1.60 | -2.41 s.d. -0.89 | turun, terbukti |
| deskriptif | Hitam − asli: porsi ke CGM/CMD/HEALTHY | N_cb | +12.29 / +11.76 / +13.79 | +12.61 | +10.76 s.d. +14.54 | naik, terbukti |
| deskriptif | Rata − asli: AP CBB (oklusi) | N_cb | -0.09 / -0.32 / -0.64 | -0.35 | -0.87 s.d. +0.14 | tidak terbukti berbeda |
| deskriptif | Rata − asli: AP CBSD (oklusi) | N_cb | -0.37 / -0.45 / -0.18 | -0.33 | -0.93 s.d. +0.14 | tidak terbukti berbeda |
| deskriptif | Rata − asli: porsi ke CGM/CMD/HEALTHY | N_cb | +0.65 / -0.10 / +0.16 | +0.24 | -0.42 s.d. +0.89 | tidak terbukti berbeda |
| sekunder | Asli − rata: porsi ke CBB/CBSD | W_semua | -2.51 / -1.95 / -2.59 | -2.35 | -3.02 s.d. -1.79 | turun, terbukti |
| sekunder | Asli − inpaint: porsi ke CBB/CBSD | W_semua | -2.85 / -2.68 / -2.72 | -2.75 | -3.51 s.d. -2.07 | turun, terbukti |
| sekunder | Asli − rata: rata-rata AP CGM/CMD/HEALTHY | W_semua | +0.89 / +1.09 / +1.12 | +1.03 | +0.77 s.d. +1.29 | naik, terbukti |
| sekunder | Asli − inpaint: rata-rata AP CGM/CMD/HEALTHY | W_semua | +0.98 / +1.23 / +1.12 | +1.11 | +0.84 s.d. +1.48 | naik, terbukti |
| sekunder | Asli − rata: AP CGM | W_semua | -0.23 / -0.16 / -0.18 | -0.19 | -0.53 s.d. +0.12 | tidak terbukti berbeda |
| sekunder | Asli − rata: AP CMD | W_semua | +0.48 / +0.58 / +0.43 | +0.50 | +0.21 s.d. +0.75 | naik, terbukti |
| sekunder | Asli − rata: AP HEALTHY | W_semua | +2.41 / +2.85 / +3.11 | +2.79 | +2.07 s.d. +3.46 | naik, terbukti |
| sekunder | Asli − inpaint: AP CGM | W_semua | -0.12 / -0.11 / -0.11 | -0.11 | -0.42 s.d. +0.26 | tidak terbukti berbeda |
| sekunder | Asli − inpaint: AP CMD | W_semua | +0.24 / +0.59 / +0.50 | +0.45 | +0.16 s.d. +0.77 | naik, terbukti |
| sekunder | Asli − inpaint: AP HEALTHY | W_semua | +2.82 / +3.20 / +2.97 | +3.00 | +2.17 s.d. +3.88 | naik, terbukti |

## Perpindahan kelas teratas per citra (deskriptif, rata-rata tiga seed)

Persentase citra yang prediksi berkeyakinan tertinggi berada di kelompok CGM/CMD/HEALTHY.

| Subset | Varian | % citra |
| --- | --- | --- |
| N_cb | N_asli | 18.1 |
| N_cb | N_baji_hitam | 30.3 |
| N_cb | N_baji_rata | 18.3 |
| W_semua | W_asli | 91.9 |
| W_semua | W_rata | 89.7 |
| W_semua | W_inpaint | 89.3 |

## Sensitivitas (estimasi titik, rata-rata tiga seed)

| Subset | Kontras | Metrik | Selisih |
| --- | --- | --- | --- |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | ap_CBB | -1.81 |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | ap_CBSD | -1.29 |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | porsi_train | +12.68 |
| W_sens (883) | W_asli − W_rata | porsi_vt | -2.36 |
| W_sens (883) | W_asli − W_inpaint | porsi_vt | -2.85 |

Catatan: kotak GT tidak diubah oleh intervensi; baji dan warna rata menutup area yang sama, sehingga kontras hitam − rata mengendalikan oklusi. IK berasal dari bootstrap klaster per grup foto (resampel sama untuk semua model dan varian), dirata-rata antar seed.

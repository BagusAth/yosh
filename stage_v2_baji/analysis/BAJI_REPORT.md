# Uji jalan pintas baji hitam — hasil
Tanggal: 2026-10-31 09:42 · model EXP-0 seed 42/123/2026 · bootstrap klaster per grup 1000 iterasi

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
| N_asli | 30.13 | 27.86 | 32.41 | 14.50 | 85.50 |
| N_baji_hitam | 28.11 | 25.41 | 30.81 | 27.87 | 72.13 |
| N_baji_rata | 29.78 | 27.41 | 32.15 | 16.10 | 83.90 |

**W_semua** (935 citra)

| Varian | map_train | ap_CGM | ap_CMD | ap_HEALTHY | ap_CBSD | porsi_vt | porsi_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W_asli | 27.12 | 27.53 | 22.30 | 31.54 | 33.06 | 7.15 | 92.85 |
| W_rata | 26.08 | 27.48 | 21.89 | 28.87 | 32.27 | 9.45 | 90.55 |
| W_inpaint | 25.90 | 27.37 | 21.96 | 28.38 | 32.10 | 10.42 | 89.58 |

**N_semua** (1,632 citra)

| Varian | map | ap_CBB | ap_CBSD | ap_CGM | ap_CMD | ap_HEALTHY | porsi_vt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N_asli | 25.15 | 25.03 | 30.59 | 25.46 | 19.77 | 24.89 | 36.21 |
| N_baji_hitam | 24.23 | 22.89 | 29.53 | 24.91 | 19.13 | 24.69 | 28.77 |
| N_baji_rata | 24.67 | 24.63 | 30.43 | 25.31 | 19.14 | 23.85 | 35.10 |

Porsi = rata-rata per citra dari bagian massa keyakinan (prediksi conf ≥ 0,25) yang jatuh ke kelompok kelas tersebut; porsi_train = CGM/CMD/HEALTHY, porsi_vt = CBB/CBSD. AP dalam poin persen (AP@50–95); map_train = rata-rata AP CGM/CMD/HEALTHY.

## Kontras (poin; per seed 42 / 123 / 2026, rata-rata, IK 95%)

| Peran | Kontras | Subset | Per seed | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| primer | Hitam − rata: AP CBB | N_cb | -1.83 / -2.72 / -1.45 | -2.00 | -2.84 s.d. -1.24 | turun, terbukti |
| primer | Hitam − rata: AP CBSD | N_cb | -1.21 / -1.99 / -0.82 | -1.34 | -1.95 s.d. -0.74 | turun, terbukti |
| primer | Hitam − rata: porsi ke CGM/CMD/HEALTHY | N_cb | +12.30 / +12.58 / +10.44 | +11.77 | +10.12 s.d. +13.54 | naik, terbukti |
| pendukung | Hitam − rata: mAP subset | N_cb | -1.52 / -2.36 / -1.14 | -1.67 | -2.15 s.d. -1.15 | turun, terbukti |
| deskriptif | Hitam − asli: AP CBB (efek total) | N_cb | -2.24 / -3.07 / -2.04 | -2.45 | -3.54 s.d. -1.49 | turun, terbukti |
| deskriptif | Hitam − asli: AP CBSD (efek total) | N_cb | -1.23 / -2.42 / -1.14 | -1.59 | -2.40 s.d. -0.74 | turun, terbukti |
| deskriptif | Hitam − asli: porsi ke CGM/CMD/HEALTHY | N_cb | +14.11 / +14.18 / +11.82 | +13.37 | +11.55 s.d. +15.20 | naik, terbukti |
| deskriptif | Rata − asli: AP CBB (oklusi) | N_cb | -0.41 / -0.34 / -0.59 | -0.45 | -0.89 s.d. -0.00 | turun, terbukti |
| deskriptif | Rata − asli: AP CBSD (oklusi) | N_cb | -0.02 / -0.43 / -0.32 | -0.26 | -0.78 s.d. +0.27 | tidak terbukti berbeda |
| deskriptif | Rata − asli: porsi ke CGM/CMD/HEALTHY | N_cb | +1.81 / +1.60 / +1.38 | +1.60 | +0.84 s.d. +2.35 | naik, terbukti |
| sekunder | Asli − rata: porsi ke CBB/CBSD | W_semua | -1.68 / -2.61 / -2.61 | -2.30 | -2.99 s.d. -1.70 | turun, terbukti |
| sekunder | Asli − inpaint: porsi ke CBB/CBSD | W_semua | -3.10 / -3.46 / -3.25 | -3.27 | -4.14 s.d. -2.43 | turun, terbukti |
| sekunder | Asli − rata: rata-rata AP CGM/CMD/HEALTHY | W_semua | +0.94 / +1.16 / +1.03 | +1.04 | +0.78 s.d. +1.30 | naik, terbukti |
| sekunder | Asli − inpaint: rata-rata AP CGM/CMD/HEALTHY | W_semua | +0.98 / +1.40 / +1.29 | +1.22 | +0.88 s.d. +1.54 | naik, terbukti |
| sekunder | Asli − rata: AP CGM | W_semua | +0.03 / +0.21 / -0.09 | +0.05 | -0.29 s.d. +0.37 | tidak terbukti berbeda |
| sekunder | Asli − rata: AP CMD | W_semua | +0.43 / +0.20 / +0.62 | +0.42 | +0.20 s.d. +0.70 | naik, terbukti |
| sekunder | Asli − rata: AP HEALTHY | W_semua | +2.36 / +3.08 / +2.56 | +2.67 | +1.92 s.d. +3.32 | naik, terbukti |
| sekunder | Asli − inpaint: AP CGM | W_semua | +0.08 / +0.47 / -0.07 | +0.16 | -0.24 s.d. +0.52 | tidak terbukti berbeda |
| sekunder | Asli − inpaint: AP CMD | W_semua | +0.28 / +0.11 / +0.65 | +0.35 | +0.04 s.d. +0.72 | naik, terbukti |
| sekunder | Asli − inpaint: AP HEALTHY | W_semua | +2.59 / +3.60 / +3.29 | +3.16 | +2.22 s.d. +4.05 | naik, terbukti |

## Perpindahan kelas teratas per citra (deskriptif, rata-rata tiga seed)

Persentase citra yang prediksi berkeyakinan tertinggi berada di kelompok CGM/CMD/HEALTHY.

| Subset | Varian | % citra |
| --- | --- | --- |
| N_cb | N_asli | 14.8 |
| N_cb | N_baji_hitam | 27.8 |
| N_cb | N_baji_rata | 16.9 |
| W_semua | W_asli | 92.5 |
| W_semua | W_rata | 90.3 |
| W_semua | W_inpaint | 89.1 |

## Sensitivitas (estimasi titik, rata-rata tiga seed)

| Subset | Kontras | Metrik | Selisih |
| --- | --- | --- | --- |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | ap_CBB | -2.06 |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | ap_CBSD | -1.34 |
| N_cb_sens (605) | N_baji_hitam − N_baji_rata | porsi_train | +12.01 |
| W_sens (883) | W_asli − W_rata | porsi_vt | -2.22 |
| W_sens (883) | W_asli − W_inpaint | porsi_vt | -3.25 |

Catatan: kotak GT tidak diubah oleh intervensi; baji dan warna rata menutup area yang sama, sehingga kontras hitam − rata mengendalikan oklusi. IK berasal dari bootstrap klaster per grup foto (resampel sama untuk semua model dan varian), dirata-rata antar seed.

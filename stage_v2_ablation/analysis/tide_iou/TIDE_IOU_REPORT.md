# Analisis pasca-hoc butir 6–7: TIDE dan AP per ambang IoU (validasi)

Tanggal: 2026-11-05 17:22 · 2,567 citra validasi · bootstrap klaster 1000 iterasi · spesifikasi: SPESIFIKASI.md (ditulis sebelum dijalankan)

**Status: pasca-hoc dan eksploratif.** Tidak mengubah verdict RQ1–RQ3. Test set tidak disentuh.

## Verifikasi

- Rata-rata AP 10 ambang vs mAP@50–95 `point.json` (15 model): selisih maks 1.1e-16.
- Resampel bootstrap identik dengan analisis utama (rata-rata 10 ambang per resampel vs kolom `|m`): seed 0 1.9e-16, seed 1 1.7e-16, seed 3 1.7e-16, seed 2 1.7e-16.
- **Cek terhadap kode tidecv asli belum dijalankan** (syarat pakai butir 6 di SPESIFIKASI.md); jalankan `jalankan_cek_tidecv.bat`. Sampai cek itu lolos, angka butir 6 bersifat sementara.
- Verifikasi antara: transkripsi Python murni alur tidecv (`_panduan_kerja/verif_tide_iou/transkripsi_tidecv.py`) vs implementasi numpy: 15 model, selisih maks 1.4e-14 poin. Ini menguji vektorisasi, bukan tafsir aturan tidecv.

## Butir 6 — kontribusi jenis error dalam satuan AP50 (TIDE), rata-rata 3 seed

| Model | AP50 TIDE | Cls | Loc | Both | Dupe | Bkg | Miss | FalsePos | FalseNeg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | 35.92 | 2.08 | 8.67 | 0.71 | 0.48 | 11.28 | 2.29 | 50.35 | 5.45 |
| EXP-0R | 35.72 | 2.10 | 8.93 | 0.74 | 0.42 | 10.83 | 2.34 | 50.08 | 5.67 |
| EXP-1 | 34.14 | 2.23 | 8.53 | 0.60 | 0.43 | 11.84 | 2.33 | 51.14 | 5.65 |
| EXP-2 | 35.84 | 2.15 | 8.85 | 0.75 | 0.51 | 10.59 | 2.47 | 50.04 | 5.70 |
| EXP-3 | 35.16 | 2.18 | 8.83 | 0.61 | 0.47 | 11.49 | 2.37 | 50.78 | 5.51 |

Jumlah error per citra menurut TIDE (deskriptif, rata-rata 3 seed; max_dets 100, tanpa ambang keyakinan):

| Model | Cls | Loc | Both | Dupe | Bkg | Miss |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | 3.76 | 16.60 | 5.66 | 3.03 | 58.04 | 0.26 |
| EXP-0R | 3.73 | 16.52 | 5.82 | 2.96 | 58.09 | 0.26 |
| EXP-1 | 3.81 | 16.47 | 5.95 | 2.97 | 60.10 | 0.27 |
| EXP-2 | 3.87 | 16.77 | 5.92 | 3.05 | 56.96 | 0.27 |
| EXP-3 | 3.82 | 16.92 | 6.04 | 3.10 | 59.54 | 0.26 |

### EXP-1 − EXP-0R (poin dAP; per seed · rata-rata (IK 95%) · verdict)

| Besaran | Per seed | Rata-rata (IK 95%) | Verdict |
| --- | --- | --- | --- |
| AP50_tide | -1.33 / -1.68 / -1.74 | -1.58 (-2.28 s.d. -0.75) | menurun konsisten |
| dAP_Cls | +0.16 / +0.14 / +0.08 | +0.13 (-0.16 s.d. +0.49) | tidak terbukti berbeda |
| dAP_Loc | -0.41 / -0.43 / -0.37 | -0.40 (-0.73 s.d. +0.01) | tidak terbukti berbeda |
| dAP_Both | +0.02 / -0.18 / -0.24 | -0.14 (-0.22 s.d. -0.05) | tidak terbukti berbeda |
| dAP_Dupe | +0.03 / +0.05 / -0.05 | +0.01 (-0.03 s.d. +0.05) | tidak terbukti berbeda |
| dAP_Bkg | +0.13 / +1.64 / +1.27 | +1.01 (+0.62 s.d. +1.39) | meningkat konsisten |
| dAP_Miss | +0.21 / -0.08 / -0.18 | -0.01 (-0.14 s.d. +0.12) | tidak terbukti berbeda |
| dAP_FalsePos | +0.74 / +0.49 / +1.94 | +1.06 (+0.15 s.d. +1.87) | meningkat konsisten |
| dAP_FalseNeg | -0.00 / +0.25 / -0.29 | -0.01 (-0.25 s.d. +0.22) | tidak terbukti berbeda |

### EXP-2 − EXP-0 (poin dAP; per seed · rata-rata (IK 95%) · verdict)

| Besaran | Per seed | Rata-rata (IK 95%) | Verdict |
| --- | --- | --- | --- |
| AP50_tide | +0.19 / -0.13 / -0.31 | -0.08 (-0.64 s.d. +0.51) | tidak terbukti berbeda |
| dAP_Cls | -0.21 / +0.17 / +0.25 | +0.07 (-0.15 s.d. +0.30) | tidak terbukti berbeda |
| dAP_Loc | +0.25 / +0.39 / -0.08 | +0.19 (-0.14 s.d. +0.49) | tidak terbukti berbeda |
| dAP_Both | +0.07 / -0.03 / +0.08 | +0.04 (-0.03 s.d. +0.11) | tidak terbukti berbeda |
| dAP_Dupe | -0.05 / +0.11 / +0.03 | +0.03 (-0.01 s.d. +0.07) | tidak terbukti berbeda |
| dAP_Bkg | -0.22 / -1.41 / -0.43 | -0.69 (-0.94 s.d. -0.40) | menurun konsisten |
| dAP_Miss | +0.18 / +0.29 / +0.06 | +0.18 (+0.04 s.d. +0.32) | meningkat konsisten |
| dAP_FalsePos | -0.39 / -0.27 / -0.29 | -0.31 (-1.07 s.d. +0.27) | tidak terbukti berbeda |
| dAP_FalseNeg | +0.13 / +0.34 / +0.25 | +0.24 (+0.02 s.d. +0.43) | meningkat konsisten |

Semua kontras (rata-rata (IK 95%)) untuk AP50 TIDE, dAP_Bkg, dAP_Miss, dAP_FalsePos, dAP_FalseNeg:

| Kontras | AP50 TIDE | dAP_Bkg | dAP_Miss | dAP_FalsePos | dAP_FalseNeg |
| --- | --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | -0.19 (-0.74 s.d. +0.35) | -0.45 (-0.70 s.d. -0.20) | +0.05 (-0.06 s.d. +0.17) | -0.27 (-0.91 s.d. +0.39) | +0.21 (+0.00 s.d. +0.43) |
| EXP-1 − EXP-0R | -1.58 (-2.28 s.d. -0.75) | +1.01 (+0.62 s.d. +1.39) | -0.01 (-0.14 s.d. +0.12) | +1.06 (+0.15 s.d. +1.87) | -0.01 (-0.25 s.d. +0.22) |
| EXP-1 − EXP-0 | -1.78 (-2.51 s.d. -0.90) | +0.56 (+0.14 s.d. +0.99) | +0.04 (-0.08 s.d. +0.18) | +0.79 (-0.11 s.d. +1.59) | +0.20 (-0.04 s.d. +0.43) |
| EXP-2 − EXP-0 | -0.08 (-0.64 s.d. +0.51) | -0.69 (-0.94 s.d. -0.40) | +0.18 (+0.04 s.d. +0.32) | -0.31 (-1.07 s.d. +0.27) | +0.24 (+0.02 s.d. +0.43) |
| EXP-3 − EXP-0 | -0.76 (-1.38 s.d. -0.07) | +0.21 (-0.07 s.d. +0.45) | +0.08 (-0.05 s.d. +0.20) | +0.43 (-0.44 s.d. +1.14) | +0.06 (-0.15 s.d. +0.28) |
| EXP-3 − EXP-2 | -0.68 (-1.37 s.d. +0.00) | +0.90 (+0.58 s.d. +1.19) | -0.10 (-0.24 s.d. +0.02) | +0.74 (-0.03 s.d. +1.52) | -0.19 (-0.39 s.d. +0.06) |
| EXP-3 − EXP-1 | +1.02 (+0.32 s.d. +1.71) | -0.35 (-0.73 s.d. +0.01) | +0.04 (-0.10 s.d. +0.15) | -0.36 (-1.20 s.d. +0.42) | -0.14 (-0.34 s.d. +0.09) |
| Interaksi | +1.10 (+0.19 s.d. +1.96) | +0.34 (-0.13 s.d. +0.79) | -0.14 (-0.32 s.d. +0.01) | -0.04 (-1.02 s.d. +1.01) | -0.38 (-0.64 s.d. -0.06) |

## Butir 7 — AP per ambang IoU (hitung ulang, poin), rata-rata 3 seed

| Model | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | 35.92 | 34.70 | 33.10 | 31.50 | 29.44 | 27.26 | 24.37 | 20.96 | 15.81 | 6.79 |
| EXP-0R | 35.73 | 34.47 | 32.85 | 31.12 | 29.41 | 27.00 | 23.95 | 20.44 | 15.35 | 6.26 |
| EXP-1 | 34.15 | 32.79 | 31.11 | 29.58 | 27.81 | 25.67 | 22.59 | 19.31 | 14.51 | 5.90 |
| EXP-2 | 35.86 | 34.57 | 33.05 | 31.46 | 29.63 | 27.39 | 24.36 | 20.85 | 15.68 | 6.63 |
| EXP-3 | 35.16 | 33.77 | 32.25 | 30.68 | 28.87 | 26.72 | 23.80 | 20.36 | 14.94 | 5.91 |

### EXP-1 − EXP-0R per ambang IoU

| Ambang | Per seed | Rata-rata (IK 95%) | Verdict |
| --- | --- | --- | --- |
| 0.50 | -1.37 / -1.67 / -1.71 | -1.58 (-2.28 s.d. -0.77) | menurun konsisten |
| 0.55 | -1.33 / -1.66 / -2.03 | -1.68 (-2.36 s.d. -0.84) | menurun konsisten |
| 0.60 | -1.49 / -1.67 / -2.05 | -1.74 (-2.44 s.d. -0.92) | menurun konsisten |
| 0.65 | -1.48 / -1.37 / -1.77 | -1.54 (-2.20 s.d. -0.72) | menurun konsisten |
| 0.70 | -1.57 / -1.46 / -1.74 | -1.59 (-2.28 s.d. -0.83) | menurun konsisten |
| 0.75 | -1.36 / -1.07 / -1.56 | -1.33 (-2.03 s.d. -0.59) | menurun konsisten |
| 0.80 | -1.26 / -1.24 / -1.56 | -1.36 (-2.04 s.d. -0.66) | menurun konsisten |
| 0.85 | -0.80 / -0.85 / -1.74 | -1.13 (-1.78 s.d. -0.51) | menurun konsisten |
| 0.90 | -0.72 / -0.68 / -1.11 | -0.84 (-1.50 s.d. -0.23) | menurun konsisten |
| 0.95 | -0.45 / -0.37 / -0.26 | -0.36 (-0.81 s.d. +0.01) | tidak terbukti berbeda |
| G (0,75–0,95 − 0,50–0,70) | +0.53 / +0.73 / +0.62 | +0.62 (+0.20 s.d. +0.90) | meningkat konsisten |

### EXP-1 − EXP-0 per ambang IoU

| Ambang | Per seed | Rata-rata (IK 95%) | Verdict |
| --- | --- | --- | --- |
| 0.50 | -1.32 / -2.08 / -1.89 | -1.77 (-2.49 s.d. -0.88) | menurun konsisten |
| 0.55 | -1.44 / -2.15 / -2.13 | -1.91 (-2.66 s.d. -1.01) | menurun konsisten |
| 0.60 | -1.39 / -2.21 / -2.39 | -2.00 (-2.74 s.d. -1.12) | menurun konsisten |
| 0.65 | -1.43 / -2.08 / -2.24 | -1.92 (-2.64 s.d. -1.05) | menurun konsisten |
| 0.70 | -1.29 / -1.87 / -1.73 | -1.63 (-2.35 s.d. -0.80) | menurun konsisten |
| 0.75 | -1.48 / -1.62 / -1.65 | -1.58 (-2.31 s.d. -0.75) | menurun konsisten |
| 0.80 | -1.76 / -2.14 / -1.42 | -1.77 (-2.52 s.d. -0.99) | menurun konsisten |
| 0.85 | -1.52 / -1.75 / -1.66 | -1.64 (-2.39 s.d. -0.94) | menurun konsisten |
| 0.90 | -0.78 / -1.74 / -1.38 | -1.30 (-2.02 s.d. -0.68) | menurun konsisten |
| 0.95 | -0.77 / -1.46 / -0.45 | -0.89 (-1.35 s.d. -0.49) | menurun konsisten |
| G (0,75–0,95 − 0,50–0,70) | +0.11 / +0.33 / +0.77 | +0.40 (-0.01 s.d. +0.73) | tidak terbukti berbeda |

### EXP-2 − EXP-0 per ambang IoU

| Ambang | Per seed | Rata-rata (IK 95%) | Verdict |
| --- | --- | --- | --- |
| 0.50 | +0.29 / -0.17 / -0.29 | -0.05 (-0.60 s.d. +0.54) | tidak terbukti berbeda |
| 0.55 | +0.23 / -0.22 / -0.42 | -0.14 (-0.69 s.d. +0.49) | tidak terbukti berbeda |
| 0.60 | +0.37 / -0.26 / -0.27 | -0.05 (-0.61 s.d. +0.52) | tidak terbukti berbeda |
| 0.65 | +0.20 / -0.21 / -0.11 | -0.04 (-0.58 s.d. +0.55) | tidak terbukti berbeda |
| 0.70 | +0.25 / +0.24 / +0.09 | +0.19 (-0.38 s.d. +0.78) | tidak terbukti berbeda |
| 0.75 | +0.11 / +0.32 / -0.04 | +0.13 (-0.41 s.d. +0.73) | tidak terbukti berbeda |
| 0.80 | -0.30 / +0.20 / +0.07 | -0.01 (-0.54 s.d. +0.56) | tidak terbukti berbeda |
| 0.85 | -0.48 / +0.15 / +0.02 | -0.10 (-0.62 s.d. +0.40) | tidak terbukti berbeda |
| 0.90 | -0.17 / -0.08 / -0.13 | -0.13 (-0.57 s.d. +0.38) | tidak terbukti berbeda |
| 0.95 | -0.32 / -0.26 / +0.09 | -0.16 (-0.58 s.d. +0.26) | tidak terbukti berbeda |
| G (0,75–0,95 − 0,50–0,70) | -0.50 / +0.19 / +0.20 | -0.04 (-0.36 s.d. +0.29) | tidak terbukti berbeda |

Selisih relatif EXP-1 terhadap EXP-0R (R_t = AP EXP-1 / AP EXP-0R − 1, AP dijumlah antar seed):

| Ambang | R_t | IK 95% |
| --- | --- | --- |
| 0.50 | -4.4% | -6.3% s.d. -2.1% |
| 0.55 | -4.9% | -6.9% s.d. -2.5% |
| 0.60 | -5.3% | -7.5% s.d. -2.8% |
| 0.65 | -4.9% | -7.0% s.d. -2.3% |
| 0.70 | -5.4% | -7.6% s.d. -2.8% |
| 0.75 | -4.9% | -7.4% s.d. -2.2% |
| 0.80 | -5.7% | -8.3% s.d. -2.8% |
| 0.85 | -5.5% | -8.6% s.d. -2.5% |
| 0.90 | -5.5% | -9.4% s.d. -1.5% |
| 0.95 | -5.7% | -12.3% s.d. +0.2% |
| G_rel | -0.5 poin persen | -2.5 s.d. +1.3 |

Catatan:
- dAP jenis utama dipotong pada 0 per model (aturan tidecv); dAP FalsePos/FalseNeg tidak dipotong. Jenis error TIDE tidak saling lepas dalam satuan AP: jumlah dAP tidak sama dengan selisih ke AP sempurna.
- AP50 TIDE memakai pencocokan TIDE dan max_dets 100, sehingga berbeda dari mAP@50 hitung ulang (max_det 300).
- Verdict memakai bahasa aturan keputusan penelitian, tetapi analisis ini pasca-hoc dan tanpa koreksi multiplisitas.

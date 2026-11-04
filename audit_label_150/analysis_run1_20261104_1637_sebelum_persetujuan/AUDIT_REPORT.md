# Laporan analisis audit label (lembar v2)

Dibuat 2026-11-04T09:37:01. Rencana analisis dikunci 2 November 2026 (docstring skrip).

## Kelengkapan

- Rater 1: 150 baris, 150 dengan Kategori utama, 0 isian tidak dikenal, 39 "Kurang yakin".
- Strata: tinggi 100, acak 50.

## Analisis utama

| Luaran | Tinggi | IK 95% | Acak | IK 95% | Selisih (IK 95% Newcombe) | p Fisher | Verdict |
|---|---|---|---|---|---|---|---|
| Y_label (utama) | 22/100 (22.0%) | 15.0% s.d. 31.1% | 36/50 (72.0%) | 58.3% s.d. 82.5% | -50.0 poin (-62.6 s.d. -33.6) | 0.000 | berlawanan |
| S1: tanpa "Kurang yakin" | 11/88 (12.5%) | 7.1% s.d. 21.0% | 10/23 (43.5%) | 25.6% s.d. 63.2% | -31.0 poin (-51.4 s.d. -11.2) | 0.002 | berlawanan |
| S2: Kategori utama saja | 22/100 (22.0%) | 15.0% s.d. 31.1% | 36/50 (72.0%) | 58.3% s.d. 82.5% | -50.0 poin (-62.6 s.d. -33.6) | 0.000 | berlawanan |
| Y_hilang (sekunder) | 11/91 (12.1%) | 6.9% s.d. 20.4% | 29/45 (64.4%) | 49.8% s.d. 76.8% | -52.4 poin (-65.7 s.d. -35.6) | 0.000 | berlawanan |
| Y_presisi (sekunder) | 14/100 (14.0%) | 8.5% s.d. 22.1% | 22/49 (44.9%) | 31.9% s.d. 58.7% | -30.9 poin (-45.7 s.d. -15.5) | 0.000 | berlawanan |

## Kategori (utama atau tambahan), eksploratif

| Kategori | Tinggi | Acak | Selisih (IK 95%) | p Fisher |
|---|---|---|---|---|
| Daun tanpa kotak | 11 (11.0%) | 30 (60.0%) | -49.0 (-62.3 s.d. -33.2) | 0.000 |
| Kotak bersarang/ganda | 4 (4.0%) | 5 (10.0%) | -6.0 (-17.6 s.d. +2.1) | 0.161 |
| Objek bukan daun | 1 (1.0%) | 0 (0.0%) | +1.0 (-6.2 s.d. +5.4) | 1.000 |
| Kotak tidak presisi | 10 (10.0%) | 8 (16.0%) | -6.0 (-19.3 s.d. +4.7) | 0.297 |
| Artefak citra | 6 (6.0%) | 2 (4.0%) | +2.0 (-8.0 s.d. +9.1) | 0.719 |
| Citra padat/sulit | 1 (1.0%) | 9 (18.0%) | -17.0 (-29.8 s.d. -7.6) | 0.000 |
| Tidak ada masalah terlihat | 72 (72.0%) | 13 (26.0%) | +46.0 (+29.5 s.d. +58.8) | 0.000 |
| Lainnya | 0 (0.0%) | 0 (0.0%) | +0.0 (-7.1 s.d. +3.7) | 1.000 |

## Sebaran "Daun tanpa kotak"

| Nilai | Tinggi | Acak |
|---|---|---|
| 0 | 80 | 16 |
| 1-2 | 7 | 10 |
| 3-5 | 1 | 6 |
| >5 | 3 | 13 |
| Ragu | 9 | 5 |

## Kesepakatan antar penilai

| Ukuran | n | Setuju | Kappa (IK 95%) |
|---|---|---|---|
| Y_label | 30 | 90.0% | 0.80 (0.55 s.d. 1.00) |
| Kategori utama | 30 | 90.0% | 0.84 (0.63 s.d. 1.00) |
| Daun tanpa kotak (berbobot) | 28 | 89.3% | 0.88 (0.61 s.d. 1.00) |

## Validasi skor otomatis OOF

| Skor vs luaran [sampel] | n | positif | AUC (IK 95%) | Valid |
|---|---|---|---|---|
| 1 - label_quality vs Y_label [semua] | 150 | 58 | 0.50 (0.40 s.d. 0.60) | tidak |
| 1 - label_quality vs Y_label [acak] | 50 | 36 | 0.78 (0.65 s.d. 0.89) | ya |
| missing_score vs Y_hilang [semua] | 136 | 40 | 0.66 (0.57 s.d. 0.75) | tidak |
| missing_score vs Y_hilang [acak] | 45 | 29 | 0.72 (0.58 s.d. 0.85) | ya |
| badloc_score vs Y_presisi [semua] | 149 | 36 | 0.64 (0.56 s.d. 0.71) | tidak |
| badloc_score vs Y_presisi [acak] | 49 | 22 | 0.68 (0.56 s.d. 0.81) | tidak |

## Paparan EXP-1 terhadap citra bermasalah label (eksploratif)

- Logistik P(Y_label | d_image): b0 = 1.88, b1 = -2.86.
- Prevalensi terprediksi di 8980 citra latih: uniform 58.6%, tertimbang EXP-1 53.9%; rasio paparan 0.919 (IK 95% 0.899 s.d. 0.945).
- Citra dengan label_quality < 0,5: 1996 citra, porsi uniform 22.2%, porsi tarikan EXP-1 23.7% (rasio 1.07); TIDAK ditafsirkan karena skor otomatis tidak valid.

## Catatan

- Benar-salahnya kelas tidak dinilai (penilai bukan ahli penyakit tanaman); keterbatasan penelitian.
- Hanya analisis utama yang konfirmatori; bagian lain eksploratif tanpa koreksi multiplisitas.

# Analisis pasca-hoc audit label (EKSPLORATIF, bukan pra-registrasi)

Dibuat 2026-11-04T13:13:50. Disetujui pengguna 4 November 2026 20.09; spesifikasi di docstring analyze_audit_pascahoc.py ditulis sebelum dijalankan, setelah hasil utama audit diketahui.

## P1 Komposisi strata

| Ciri | Tinggi | Acak | Populasi latih |
|---|---|---|---|
| Citra (objek) | 100 (114) | 50 (203) | 8980 (34219) |
| Kelas dominan CBB | 28,0% | 8,0% | 12,2% |
| Kelas dominan CBSD | 18,0% | 16,0% | 13,4% |
| Kelas dominan CGM | 16,0% | 14,0% | 24,6% |
| Kelas dominan CMD | 26,0% | 22,0% | 24,5% |
| Kelas dominan HEALTHY | 12,0% | 40,0% | 24,6% |
| Kelas dominan (tanpa kotak) | 0,0% | 0,0% | 0,7% |
| Jumlah kotak, median [Q1, Q3] | 1 [1, 1] | 3 [1, 6] | 3 [1, 5] |
| Citra 1 kotak | 87,0% | 28,0% | 32,9% |
| Citra >= 7 kotak | 0,0% | 18,0% | 16,5% |
| Objek Small (% objek) | 7,0% | 5,4% | 3,4% |
| Objek Medium (% objek) | 0,9% | 30,0% | 27,3% |
| Objek Large (% objek) | 92,1% | 64,5% | 69,3% |
| Citra dengan >= 1 objek kecil | 7,0% | 12,0% | 8,8% |
| OOF True_Positive (% objek) | 0,0% | 40,9% | 35,7% |
| OOF False_Negative_Missed (% objek) | 7,0% | 44,8% | 48,2% |
| OOF Localization_Error (% objek) | 0,0% | 12,3% | 10,2% |
| OOF Classification_Confusion (% objek) | 93,0% | 0,5% | 3,1% |
| OOF Localization_and_Class_Error (% objek) | 0,0% | 1,5% | 2,7% |
| Citra dengan >= 1 objek OOF salah kelas | 94,0% | 2,0% | 10,1% |
| Asal Roboflow test | 22,0% | 8,0% | 11,8% |
| Asal Roboflow train | 54,0% | 76,0% | 76,5% |
| Asal Roboflow valid | 24,0% | 16,0% | 11,7% |
| Baji hitam | 26,0% | 38,0% | 36,6% |
| Pinggiran bergaris (citra tanpa baji) | 36,5% | 19,4% | 26,5% |
| Artefak rotasi apa pun | 53,0% | 50,0% | 53,4% |
| Sinyal OOF missing>0 | 6,0% | 14,0% | 12,7% |
| Sinyal OOF swap>0 | 94,0% | 2,0% | 10,1% |
| Sinyal OOF badloc>0 | 0,0% | 30,0% | 21,7% |
| Median label_quality | 0,576 | 0,661 | 0,697 |

Artefak populasi dari sampel acak 1000 citra latih (Fase 5).

## P2 Y_label per kelas dominan

| Kelas | Tinggi | Acak |
|---|---|---|
| CBB | 6/28 | 1/4 |
| CBSD | 1/18 | 2/8 |
| CGM | 2/16 | 5/7 |
| CMD | 7/26 | 9/11 |
| HEALTHY | 6/12 | 19/20 |
| (tanpa kotak) | 0/0 | 0/0 |

Selisih tertimbang Mantel-Haenszel (tinggi - acak): -40,1 poin (IK bootstrap -55,3 s.d. -24,0).

## P3 D_image dan Y_label di strata acak

AUC 0,712 (IK 0,549 s.d. 0,862).

## P4 Paparan EXP-1 tanpa asumsi monoton

| Pita | Batas D_image | Citra latih | Porsi citra | Porsi tarikan EXP-1 | Y_label audit |
|---|---|---|---|---|---|
| 0 | min s.d. 0.3289 | 2215 | 24,7% | 14,3% | 7/14 |
| 1 | 0.3289 s.d. 0.5331 | 2213 | 24,6% | 21,7% | 12/16 |
| 2 | 0.5331 s.d. 0.7022 | 2215 | 24,7% | 27,8% | 8/10 |
| 3 | 0.7022 s.d. 0.9899 | 2215 | 24,7% | 33,9% | 9/10 |
| 4 | 0.9899 s.d. maks | 122 | 1,4% | 2,2% | 22/100 |

| Estimator | Rasio paparan (IK 95%) | Porsi uniform | Porsi EXP-1 | Tafsiran |
|---|---|---|---|---|
| E1 | 1,051 (0,999 s.d. 1,106) | 73,0% | 76,7% | belum dapat ditentukan |
| E2 | 1,060 (1,007 s.d. 1,115) | 72,9% | 77,3% | lebih sering |
| E3 | 0,994 (0,993 s.d. 0,995) | 71,3% | 70,9% | lebih jarang |
| E1-S1 | 1,103 (0,911 s.d. 1,215) | 46,2% | 50,9% | belum dapat ditentukan |

E1 = 4 pita kuartil Sisa + Puncak (utama pasca-hoc); E2 = logistik pada citra acak untuk Sisa; E3 = Sisa tanpa gradien; E1-S1 = E1 tanpa "Kurang yakin".

Semua hasil di laporan ini pasca-hoc dan eksploratif; tidak menggantikan analisis yang dipra-registrasi.

# Komposisi sampel yang ditekankan EXP-1 (Fase 5 A3)
Tanggal: 2026-11-01 10:12 · 8,980 citra latih · bobot relatif w = N × P(k), λ = 0,70

Sebaran bobot: min 0.32, p10 0.56, median 1.02, p90 1.38, maks 1.64; ESS relatif 0.910. 10% citra berbobot tertinggi menerima 15.0% tarikan (uniform 10%); 10% terendah 4.7%.

Rasio paparan citra = rata-rata w di kelompok (1 = sama dengan uniform). Objek mewarisi bobot citranya; rasio objek dinormalkan terhadap rata-rata bobot semua objek (= porsi EXP-1 / porsi uniform). Secara total EXP-1 menarik 11.5% lebih banyak objek per epoch daripada uniform karena citra berobjek banyak berbobot lebih tinggi.

## Citra per kelas dominan

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| CBB | 1,095 | 12.2% | 12.1% | 0.995 (0.975–1.013) |
| CBSD | 1,207 | 13.4% | 12.9% | 0.957 (0.933–0.979) |
| CGM | 2,208 | 24.6% | 22.9% | 0.933 (0.912–0.952) |
| CMD | 2,198 | 24.5% | 25.4% | 1.038 (1.021–1.055) |
| HEALTHY | 2,206 | 24.6% | 26.0% | 1.057 (1.042–1.072) |

## Citra per split asal Roboflow

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| train | 6,868 | 76.5% | 77.0% | 1.007 (0.997–1.017) |
| valid | 1,053 | 11.7% | 11.4% | 0.969 (0.949–0.989) |
| test | 1,059 | 11.8% | 11.6% | 0.985 (0.964–1.006) |

## Citra dengan target kecil (1) dan tanpa (0)

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| 1 | 755 | 8.4% | 11.5% | 1.372 (1.362–1.382) |
| 0 | 8,225 | 91.6% | 88.5% | 0.966 (0.957–0.974) |

## Citra per jumlah objek

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| 1 | 2,958 | 32.9% | 27.0% | 0.820 (0.805–0.835) |
| 2-3 | 2,523 | 28.1% | 28.6% | 1.019 (1.005–1.033) |
| 4-6 | 1,955 | 21.8% | 24.0% | 1.102 (1.088–1.116) |
| 7+ | 1,478 | 16.5% | 19.7% | 1.195 (1.182–1.207) |

## Objek per ukuran

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| Small | 1,149 | 3.4% | 4.2% | 1.239 (1.226–1.253) |
| Medium | 9,358 | 27.3% | 30.6% | 1.119 (1.112–1.126) |
| Large | 23,712 | 69.3% | 65.2% | 0.941 (0.938–0.945) |

## Objek per kelas

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| CBB | 2,534 | 7.4% | 7.4% | 0.996 (0.975–1.015) |
| CBSD | 2,307 | 6.7% | 6.3% | 0.939 (0.917–0.962) |
| CGM | 6,747 | 19.7% | 19.3% | 0.978 (0.960–0.994) |
| CMD | 9,408 | 27.5% | 28.1% | 1.021 (1.009–1.032) |
| HEALTHY | 13,223 | 38.6% | 38.9% | 1.008 (0.998–1.018) |

## Objek per taksonomi OOF

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| TP | 12,216 | 35.7% | 29.7% | 0.831 (0.823–0.838) |
| kotak meleset | 3,503 | 10.2% | 10.2% | 0.993 (0.983–1.003) |
| salah kelas | 1,069 | 3.1% | 4.1% | 1.298 (1.286–1.312) |
| meleset + salah kelas | 925 | 2.7% | 3.1% | 1.142 (1.123–1.160) |
| tidak terdeteksi | 16,506 | 48.2% | 53.0% | 1.099 (1.094–1.105) |

## Objek dengan OOF lebih yakin pada kelas lain (1) dan lainnya (0)

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) |
| --- | --- | --- | --- | --- |
| 1 | 1,150 | 3.4% | 4.3% | 1.289 (1.276–1.303) |
| 0 | 33,069 | 96.6% | 95.7% | 0.990 (0.989–0.991) |

## Objek per kelas × ukuran (rasio paparan)

| Kelas | Kecil | Sedang | Besar |
| --- | --- | --- | --- |
| CBB | 1.265 (n 122) | 1.188 (n 474) | 0.932 (n 1,938) |
| CBSD | 1.325 (n 38) | 1.228 (n 238) | 0.898 (n 2,031) |
| CGM | 1.245 (n 266) | 1.131 (n 1,532) | 0.916 (n 4,949) |
| CMD | 1.245 (n 248) | 1.126 (n 2,373) | 0.976 (n 6,787) |
| HEALTHY | 1.219 (n 475) | 1.099 (n 4,741) | 0.941 (n 8,007) |

## Artefak rotasi (sampel acak 1.000 citra latih, IK bootstrap citra)

| Kelompok | n | Porsi uniform | Porsi EXP-1 | Rasio paparan (IK 95%) | Median D_image |
| --- | --- | --- | --- | --- | --- |
| baji hitam | 366 | 36.6% | 36.6% | 0.990 (0.958–1.021) | 0.534 |
| bergaris, tanpa baji | 168 | 16.8% | 16.7% | 0.983 (0.936–1.031) | 0.500 |
| tanpa artefak terdeteksi | 466 | 46.6% | 46.7% | 0.991 (0.963–1.019) | 0.526 |

Rasio per kelas (baji hitam / bergaris / bersih): CBB 1.02 / 0.92 / 1.00; CBSD 0.86 / 1.02 / 0.89; CGM 0.88 / 0.90 / 0.97; CMD 1.05 / 1.03 / 1.06; HEALTHY 1.05 / 1.09 / 1.03

## Desil skor kesulitan D_image (rasio paparan)

D1: 0.47 · D2: 0.62 · D3: 0.75 · D4: 0.86 · D5: 0.97 · D6: 1.07 · D7: 1.16 · D8: 1.25 · D9: 1.35 · D10: 1.50

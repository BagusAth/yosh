# Prevalensi pinggiran bergaris dan baji hitam (Fase 5 A5)
Tanggal: 2026-10-31 20:29 · validasi bersih 2,567 citra · sampel latih acak 1,000 citra (seed 2031)

Dua artefak augmentasi rotasi Roboflow: **baji hitam** (isian konstan hitam; kriteria uji baji: sudut 24 px atau IoU rotasi ≥ 0,5) dan **pinggiran bergaris** (isian replikasi tepi; detektor petak cek_pinggiran_bergaris.py). Persentase disertai IK 95% Wilson.

## Kalibrasi detektor pinggiran bergaris

40 citra validasi acak tanpa baji hitam dilabeli visual: 13 positif, 2 ragu, 25 negatif.
Pada 38 citra yang pasti: TP 11, FN 2, FP 0, TN 25 → sensitivitas 85%, spesifisitas 100%. Dari 2 citra ragu, 1 terdeteksi.
Pada citra yang juga berbaji hitam, tepi baji bisa ikut terbaca sebagai garis; cek visual 12 citra W bergaris memperlihatkan sekitar 9 yang benar-benar bergaris. Angka "bergaris dengan baji" karena itu hanya indikatif.

## Validasi bersih

- Baji hitam: 935/2,567 = 36.4% (34.6–38.3)
- Pinggiran bergaris pada citra tanpa baji hitam: 449/1,632 = 27.5% (25.4–29.7)
- Pinggiran bergaris pada citra berbaji hitam (indikatif, rotasi ganda): 192/935 = 20.5% (18.1–23.2)
- Artefak rotasi apa pun (baji hitam atau bergaris): 1,384/2,567 = 53.9% (52.0–55.8)
- Tanpa artefak yang terdeteksi: 1,183/2,567 = 46.1% (44.2–48.0)

Dikoreksi untuk sensitivitas detektor (~85%), pinggiran bergaris pada citra tanpa baji sekitar 33%.

| Kelas dominan | n | Baji hitam | Bergaris (tanpa baji) | Artefak apa pun |
| --- | --- | --- | --- | --- |
| CBB | 312 | 0.6% | 42.3% (131/310) | 42.6% |
| CBSD | 355 | 12.7% | 31.9% (99/310) | 40.6% |
| CGM | 630 | 47.8% | 20.4% (67/329) | 58.4% |
| CMD | 624 | 46.6% | 22.2% (74/333) | 58.5% |
| HEALTHY | 628 | 46.2% | 22.2% (75/338) | 58.1% |

| Split asal Roboflow | n | Baji hitam | Bergaris (tanpa baji) | Artefak apa pun |
| --- | --- | --- | --- | --- |
| train | 1,972 | 46.9% | 20.7% (217/1048) | 57.9% |
| valid | 310 | 1.3% | 48.0% (147/306) | 48.7% |
| test | 285 | 2.5% | 30.6% (85/278) | 32.3% |

Komposisi kelas per split asal (jumlah citra): train: CBB 0, CBSD 78, CGM 630, CMD 624, HEALTHY 628; valid: CBB 187, CBSD 120, CGM 0, CMD 0, HEALTHY 0; test: CBB 125, CBSD 157, CGM 0, CMD 0, HEALTHY 0

## Sampel latih

- Baji hitam: 366/1,000 = 36.6% (33.7–39.6)
- Pinggiran bergaris pada citra tanpa baji hitam: 168/634 = 26.5% (23.2–30.1)
- Pinggiran bergaris pada citra berbaji hitam (indikatif, rotasi ganda): 61/366 = 16.7% (13.2–20.8)
- Artefak rotasi apa pun (baji hitam atau bergaris): 534/1,000 = 53.4% (50.3–56.5)
- Tanpa artefak yang terdeteksi: 466/1,000 = 46.6% (43.5–49.7)

| Kelas dominan | n | Baji hitam | Bergaris (tanpa baji) | Artefak apa pun |
| --- | --- | --- | --- | --- |
| CBB | 129 | 1.6% | 42.5% (54/127) | 43.4% |
| CBSD | 126 | 13.5% | 20.2% (22/109) | 31.0% |
| CGM | 242 | 45.0% | 21.8% (29/133) | 57.0% |
| CMD | 237 | 50.6% | 23.1% (27/117) | 62.0% |
| HEALTHY | 262 | 44.3% | 24.7% (36/146) | 58.0% |

| Split asal Roboflow | n | Baji hitam | Bergaris (tanpa baji) | Artefak apa pun |
| --- | --- | --- | --- | --- |
| train | 772 | 46.9% | 22.4% (92/410) | 58.8% |
| valid | 115 | 0.9% | 45.6% (52/114) | 46.1% |
| test | 113 | 2.7% | 21.8% (24/110) | 23.9% |

Komposisi kelas per split asal (jumlah citra): train: CBB 0, CBSD 28, CGM 242, CMD 237, HEALTHY 262; valid: CBB 73, CBSD 42, CGM 0, CMD 0, HEALTHY 0; test: CBB 56, CBSD 56, CGM 0, CMD 0, HEALTHY 0


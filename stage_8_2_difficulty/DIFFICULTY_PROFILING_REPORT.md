# Laporan Hasil Stage 8.2 — Profil Kesulitan Objek & Citra OOF
Tanggal: 2026-10-08 19:01:48

## 1. Status Sanity Gate
- **Status Akhir**: **PASS**
- **Proporsi Citra Hard ($D_\text{image} \ge 0.80$)**: 8.95% (Kriteria: $\le 60\%$ PASS, $60-80\%$ REVIEW, $>80\%$ FAIL)
- **Dynamic Range ($P90 - P10$)**: 0.6035 (Kriteria diagnostik: $\ge 0.30$)
- **Effective Sample Size (ESS)**: 8172.8 / 8980 ($ESS_\text{ratio} = 0.9101$)
- **Clipping Rate ($D_i = 1.0$ pada Wrong Class)**: 8.11% (374/4612)

## 2. Taksonomi Error Ground Truth (OOF Validation Matching)
| Taksonomi Error | Jumlah Objek | Persentase |
| :--- | :---: | :---: |
| True Positive (TP) | 27,588 | 80.62% |
| Localization Error | 1,164 | 3.40% |
| Classification Confusion | 4,251 | 12.42% |
| Localization and Class Error | 361 | 1.05% |
| False Negative (Missed) | 855 | 2.50% |
| **Total Ground Truth** | **34,219** | **100.00%** |

## 3. Distribusi Kuantil Kesulitan Citra ($D_\text{image}$)
- **Min**: 0.0154
- **P10**: 0.1883
- **P25**: 0.3348
- **Median (P50)**: 0.4810
- **P75**: 0.6762
- **P90**: 0.7918
- **Max**: 1.0000
- **Mean $\pm$ Std**: 0.4936

## 4. Kesiapan Menuju Stage 8.3 & 8.4
Manifest `train_image_difficulty_manifest.csv` telah memuat probabilitas sampling $P(k)$ terkalibrasi
dan identitas lesi target $i^*$ untuk augmentasi on-the-fly.

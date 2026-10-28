# Laporan Hasil Stage 8.2 — Profil Kesulitan Objek & Citra OOF (Hardened)
Tanggal: 2026-10-28 19:19:10
Versi pencocokan: v2 (tau = 0.25)

## 1. Status Sanity Gate
- **Status Akhir**: **PASS**
- **Proporsi Citra Hard ($D_\text{image} \ge 0.80$)**: 14.35% (Kriteria: $\le 60\%$ PASS, $60-80\%$ REVIEW, $>80\%$ FAIL)
- **Dynamic Range ($P90 - P10$)**: 0.6173 (Kriteria diagnostik: $\ge 0.30$)
- **Effective Sample Size (ESS)**: 8170.5 / 8980 ($ESS_\text{ratio} = 0.9099$)
- **Clipping Rate ($D_i = 1.0$ pada Wrong Class)**: 32.52% (374/1150)

## 2. Taksonomi Error Ground Truth (OOF Validation Matching)
| Taksonomi Error | Jumlah Objek | Persentase |
| :--- | :---: | :---: |
| True Positive (TP) | 12,216 | 35.70% |
| Localization Error | 3,503 | 10.24% |
| Classification Confusion | 1,069 | 3.12% |
| Localization and Class Error | 925 | 2.70% |
| False Negative (Missed) | 16,506 | 48.24% |
| **Total Ground Truth** | **34,219** | **100.00%** |

## 3. Distribusi Kuantil Kesulitan Citra ($D_\text{image}$)
- **Min**: 0.0158
- **P10**: 0.1902
- **P25**: 0.3317
- **Median (P50)**: 0.5378
- **P75**: 0.7101
- **P90**: 0.8076
- **Max**: 1.0000
- **Mean $\pm$ Std**: 0.5216

## 4. Kesiapan Menuju Stage 8.3 & 8.4
Manifest `train_image_difficulty_manifest.csv` telah memuat probabilitas sampling $P(k)$ terkalibrasi
dan identitas lesi target $i^*$ untuk augmentasi on-the-fly.

# Laporan Eksperimen Ablasi Stage 8 (EXP-0 s.d. EXP-3)
Tanggal Pembuatan: 2026-10-09 03:22:12

## 1. Tabel Komparasi Hasil Validasi (Validation Set: 2.567 citra)

| ID | Konfigurasi | Sampling | Augmentasi | Precision | Recall | mAP@50 | mAP@50-95 | Δ mAP@50 | Δ mAP@50-95 | Durasi |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | Baseline (Re-run) | Uniform (1/N) | Standard YOLO11n | 40.57% | 43.78% | **36.66%** | **26.58%** | 0.00% (Ref) | 0.00% (Ref) | 126.41 m |
| **EXP-1** | Mixture Adaptive Sampling | Mixture Adaptive P(k) | Standard YOLO11n | 39.34% | 40.34% | **34.20%** | **24.42%** | -2.46% | -2.16% | 120.82 m |
| **EXP-2** | Targeted Context Augmentation | Uniform (1/N) | On-the-Fly Context Crop (2× bbox) | 39.63% | 42.58% | **36.12%** | **26.24%** | -0.54% | -0.34% | 121.81 m |
| **EXP-3** | Full Proposed (Sampling + Augmentation) | Mixture Adaptive P(k) | On-the-Fly Context Crop (2× bbox) | 39.46% | 40.90% | **35.09%** | **25.25%** | -1.57% | -1.33% | 122.83 m |

## 2. Analisis Performa Per-Kelas (mAP@50)

| ID | CBB | CBSD | CGM | CMD | HEALTHY | Mean mAP@50 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | 23.57% | 31.15% | 27.52% | 21.02% | 29.63% | **36.66%** |
| **EXP-1** | 21.08% | 28.55% | 26.20% | 20.22% | 26.04% | **34.20%** |
| **EXP-2** | 24.92% | 29.99% | 26.42% | 21.07% | 28.79% | **36.12%** |
| **EXP-3** | 24.48% | 28.89% | 25.25% | 20.41% | 27.19% | **35.09%** |

## 3. Evaluasi Pertanyaan Penelitian & Hipotesis
- **RQ1 (Efek Mixture Adaptive Sampling):** Komparasi EXP-1 terhadap EXP-0.
- **RQ2 (Efek Targeted Context Augmentation):** Komparasi EXP-2 terhadap EXP-0.
- **RQ3 (Kombinasi Kedua Mekanisme):** Komparasi EXP-3 terhadap EXP-1 dan EXP-2.

*Catatan Metodologis:* Hasil seleksi model terbaik berdasarkan hierarki validasi (Primary: mAP@50-95) akan diteruskan ke evaluasi akhir satu pintu pada Locked Test Set (Stage 8.6).

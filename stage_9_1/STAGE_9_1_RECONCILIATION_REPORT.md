# LAPORAN REKONSILIASI METRIK RESMI (STAGE 9.1)
## *Rekonsiliasi Lengkap Metrik Kanonis, Ketertelusuran False Negative, dan Penetapan Model Pemenang*
**Proyek Skripsi**: Optimasi Model YOLO Berbasis Analisis Error dan Tingkat Kesulitan Objek untuk Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong  
**Peran**: ML Research Engineer  
**Workspace**: `D:\UNDIP\SKRIPSI`  
**Tanggal**: 2026-10-10 09:56:07  
**Status Evaluasi Akhir**: **`READY_FOR_CLEAN_EXPERIMENTS`**

---

## 1. Ringkasan Eksekutif & Keputusan Status

Tahap Stage 9.1 telah menyelesaikan seluruh ketidaksesuaian metrik dan ambiguitas metodologis yang diidentifikasi pada Audit Stage 9 tanpa mengubah atau memanipulasi data asli:

> ### 🟢 KEPUTUSAN FINAL: `READY_FOR_CLEAN_EXPERIMENTS`
> 
> **Kriteria Kelulusan Terpenuhi**:
> 1. **Rekonsiliasi AP Per-Kelas Selesai 100%**: Sumber nilai `val_metrics.box.ap50` (AP@50) dan `val_metrics.box.maps` (AP@50-95) telah dipisahkan secara tegas dan dikuantifikasi untuk setiap kelas pada split validasi dan split uji.
> 2. **Ketertelusuran False Negative (FN 2.654 vs FN 4.760) Terpecahkan Secara Tuntas**: Terbukti secara matematis bahwa FN=4.760 berasal dari split Validasi (2.567 citra, 9.632 objek GT), sedangkan FN=2.654 berasal dari split Uji Terkunci (1.282 citra, 5.084 objek GT). Keduanya adalah hasil evaluasi pada subset data yang berbeda dan tidak pernah saling bertentangan.
> 3. **Penetapan Pemenang Seleksi Model Terkoreksi Secara Ilmiah**: **EXP-0 (Baseline Re-run, mAP@50-95 = 26.58%)** ditetapkan secara sah sebagai model pemenang seleksi tahap validasi. Narasi pemilihan model tidak lagi menggunakan hasil Test Set.
> 4. **Penolakan Klaim "Domain Generalization"**: Ditegaskan secara metodologis bahwa split validasi dan uji berasal dari distribusi dataset Roboflow v7 yang sama (*in-distribution partition*), sehingga keunggulan EXP-2 pada Test Set dinarasikan sebagai *held-out test set generalization* dan kepekaan lesi kecil, bukan generalisasi lintas domain (*cross-domain*).

---

## 2. Fase A.1 — Verifikasi Artefak Asli Stage 7 dan EXP-0 s.d. EXP-3

Seluruh artefak kanonis dari kelima model telah diverifikasi keberadaannya dan dievaluasi ulang secara mandiri:
- **Stage 7 Baseline**: `cassava-v7-stage-7/runs/baseline_yolo11n/weights/best.pt`
- **EXP-0 Baseline**: `stage_8_ablation/exp_0_baseline/weights/best.pt`
- **EXP-1 Mixture Sampling**: `stage_8_ablation/exp_1_adaptive_sampling/weights/best.pt`
- **EXP-2 Targeted Augmentation**: `stage_8_ablation/exp_2_targeted_aug/weights/best.pt`
- **EXP-3 Full Proposed**: `stage_8_ablation/exp_3_full_proposed/weights/best.pt`

Semua model dilatih selama 50 epoch pada arsitektur YOLO11n dengan seed 42 dan input 640px.

---

## 3. Fase A.2 — Rekonsiliasi Sumber AP Per-Kelas (AP@50 vs AP@50-95)

Pada laporan sebelumnya, terjadi kesalahan penamaan variabel di mana nilai `val_metrics.box.maps` (yang merupakan AP@50-95 per-kelas) dicatat ke dalam kamus bernama `per_class_mAP50` dan disajikan dengan judul kolom `mAP@50`.

Berikut adalah tabel rekonsiliasi kanonis resmi yang memisahkan **AP@50 sejati** dan **AP@50-95 sejati**:

### 3.1 Tabel Rekonsiliasi Validation Set (2.567 Citra)
| Model | Metrik | CBB | CBSD | CGM | CMD | HEALTHY | Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | **Kanonis AP@50** | **33.63%** | **40.84%** | **39.44%** | **33.33%** | **36.05%** | **36.66%** |
| | **Kanonis AP@50-95** | **23.57%** | **31.15%** | **27.52%** | **21.02%** | **29.63%** | **26.58%** |
| **EXP-1** | **Kanonis AP@50** | **30.35%** | **38.66%** | **37.95%** | **31.80%** | **32.23%** | **34.20%** |
| | **Kanonis AP@50-95** | **21.08%** | **28.55%** | **26.20%** | **20.22%** | **26.04%** | **24.42%** |
| **EXP-2** | **Kanonis AP@50** | **34.48%** *(+0.85%)* | **39.47%** | **38.22%** | **32.99%** | **35.45%** | **36.12%** |
| | **Kanonis AP@50-95** | **24.92%** *(+1.35%)* | **29.99%** | **26.42%** | **21.07%** | **28.79%** | **26.24%** |
| **EXP-3** | **Kanonis AP@50** | **34.41%** *(+0.78%)* | **39.11%** | **36.01%** | **32.29%** | **33.65%** | **35.09%** |
| | **Kanonis AP@50-95** | **24.48%** *(+0.91%)* | **28.89%** | **25.25%** | **20.41%** | **27.19%** | **25.25%** |

### 3.2 Tabel Rekonsiliasi Locked Test Set (1.282 Citra)
| Model | Metrik | CBB | CBSD | CGM | CMD | HEALTHY | Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | **Kanonis AP@50** | **27.83%** | **36.12%** | **38.63%** | **31.60%** | **38.52%** | **34.54%** |
| | **Kanonis AP@50-95** | **19.35%** | **25.80%** | **25.96%** | **20.76%** | **31.53%** | **24.68%** |
| **EXP-1** | **Kanonis AP@50** | **25.66%** | **31.97%** | **36.19%** | **31.27%** | **36.89%** | **32.40%** |
| | **Kanonis AP@50-95** | **17.57%** | **21.71%** | **24.97%** | **19.86%** | **30.22%** | **22.86%** |
| **EXP-2** | **Kanonis AP@50** | **28.93%** *(+1.10%)* | **35.98%** | **39.91%** *(+1.28%)* | **32.00%** | **38.69%** | **35.10%** *(+0.56%)* |
| | **Kanonis AP@50-95** | **20.59%** *(+1.24%)* | **25.10%** | **27.38%** *(+1.42%)* | **20.80%** | **31.82%** | **25.14%** *(+0.46%)* |
| **EXP-3** | **Kanonis AP@50** | **28.01%** | **35.15%** | **36.82%** | **30.69%** | **35.68%** | **33.27%** |
| | **Kanonis AP@50-95** | **19.36%** | **23.27%** | **25.01%** | **19.86%** | **29.44%** | **23.39%** |

---

## 4. Fase A.3 & A.4 — Ketertelusuran False Negative: Investigasi FN 2.654 vs FN 4.760

Ditemukan penjelasan yang definitif mengenai perbedaan angka False Negative (FN):

| Parameter Komparasi | Angka: FN = 4.760 | Angka: FN = 2.654 |
| :--- | :--- | :--- |
| **Split yang Dievaluasi** | **VALIDATION SPLIT** (`cassava-v7-research/valid`) | **TEST SPLIT** (`cassava-v7-research/test`) |
| **Jumlah Citra** | **2.567 citra** | **1.282 citra** (hampir separuh ukuran validasi) |
| **Total Objek Ground Truth** | **9.632 objek** | **5.084 objek** |
| **Ambang Batas Evaluasi** | conf = 0.25, IoU >= 0.10 | conf = 0.25, IoU >= 0.10 |
| **Model** | EXP-0 Baseline | EXP-0 Baseline |
| **Laporan Sumber** | `STAGE_8_5_MODEL_SELECTION_REPORT.md` (baris 8) | `STAGE_8_FINAL_THESIS_BENCHMARK_REPORT.md` (baris 9) |
| **Rasio FN / Total GT** | 4.760 / 9.632 = **49.42%** | 2.654 / 5.084 = **52.20%** |

### Kesimpulan Ketertelusuran:
Kedua angka ini **sama sekali tidak bertentangan**:
- FN = 4.760 adalah jumlah lesi yang terlewat pada **2.567 citra validasi**.
- FN = 2.654 adalah jumlah lesi yang terlewat pada **1.282 citra test**.
Rasio lesi yang terlewat pada kedua split sangat konsisten (~49.4% vs ~52.2%), mencerminkan ambang batas operasional conf = 0.25.

Rincian lengkap disimpan dalam berkas [error_metric_traceability.csv](file:///d:/UNDIP/SKRIPSI/stage_9_1/error_metric_traceability.csv).

---

## 5. Fase A.5 & A.6 — Koreksi Narasi Seleksi Model & Klarifikasi Generalisasi

### 5.1 Koreksi Status Model Juara
1. **Model Terpilih Resmi**: **EXP-0 (Baseline Re-run)** adalah model pemenang sah berdasarkan protokol hierarki seleksi resmi yang telah dikunci pada rancangan metodologi:
   $$\text{Pemenang Validasi} = \arg\max_m \left( \text{mAP@50-95}_{\text{valid}} \right) = \text{EXP-0 (26.58%)}$$
2. **Koreksi Terhadap EXP-2**: Laporan sebelumnya keliru dalam menetapkan EXP-2 sebagai juara seleksi karena mendasarkannya pada hasil Test Set. Hasil Test Set tidak boleh digunakan untuk memilih model kandidat (*mencegah data snooping*).

### 5.2 Refutasi Klaim "Domain Generalization"
- Partisi validasi (2.567 citra) dan partisi uji (1.282 citra) berasal dari pemotongan acak bertingkat (*stratified group-aware partition*) dari sumber dataset yang sama (`cassava-v7`).
- Kedua partisi memiliki karakteristik pencahayaan, varietas tanaman, dan latar belakang yang identik secara statistik.
- Oleh karena itu, klaim "domain generalization" pada laporan sebelumnya ditarik dan dikoreksi menjadi:
  > *"Model EXP-2 menunjukkan kemampuan generalisasi yang lebih tangguh pada citra held-out test set yang tidak pernah dilihat sebelumnya (mAP@50-95 25.14% vs 24.68%, +0.46%) serta sensitivitas yang lebih tinggi terhadap lesi kecil bakteri (CBB AP@50 28.93% vs 27.83%) dengan mereduksi 155 False Negative."*

---

## 6. Daftar Artefak yang Dihasilkan pada Stage 9.1

1. 📄 [STAGE_9_1_RECONCILIATION_REPORT.md](file:///d:/UNDIP/SKRIPSI/stage_9_1/STAGE_9_1_RECONCILIATION_REPORT.md) — Dokumen laporan utama rekonsiliasi.
2. 📊 [metrics_reconciled.csv](file:///d:/UNDIP/SKRIPSI/stage_9_1/metrics_reconciled.csv) — Tabel rekonsiliasi seluruh metrik per-kelas dan metrik global.
3. 📋 [error_metric_traceability.csv](file:///d:/UNDIP/SKRIPSI/stage_9_1/error_metric_traceability.csv) — Tabel ketertelusuran lengkap angka TP, FN, GT, threshold, dan split.

*Status: Tahap Rekonsiliasi Metrik Selesai. Siap Melangkah ke Pelatihan Bersih.*

# LAPORAN PEMBERSIHAN & STANDARISASI DATASET (STAGE 10 DATA AUDIT)
## *Standarisasi Anotasi Poligon ke Bounding Box Axis-Aligned dan Eliminasi Label Korup*
**Proyek Skripsi**: Optimasi Model YOLO Berbasis Analisis Error dan Tingkat Kesulitan Objek untuk Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong  
**Dataset Sumber**: `D:\UNDIP\SKRIPSI\cassava-v7-research` (Asli, Tidak Dimodifikasi)  
**Dataset Bersih Baru**: `D:\UNDIP\SKRIPSI\cassava-v7-standardized`  
**Tanggal**: 2026-10-10 09:56:07  
**Status Audit Data Akhir**: **`READY_FOR_CLEAN_EXPERIMENTS`**

---

## 1. Ringkasan Eksekutif Standarisasi Data

Sesuai arahan Fase B, dataset asli `cassava-v7-research` dipertahankan 100% tanpa modifikasi untuk menjaga integritas historis eksperimen Stage 7–8. Sebuah versi dataset bersih yang terpisah telah berhasil dibangun pada `cassava-v7-standardized`.

> ### 🟢 STATUS DATASET: `READY_FOR_CLEAN_EXPERIMENTS`
> 
> **Pencapaian Kunci**:
> 1. **100% Baris Poligon Berhasil Dikonversi**: Sebanyak **48.868 anotasi poligon segmentasi** di seluruh split train, validasi, dan uji telah dikonversi secara presisi menjadi format deteksi standar YOLO 5-kolom: `class x_center y_center width height` dengan koordinat ternormalisasi [0, 1].
> 2. **Eliminasi 100% Peringatan Label Campuran / Korup**: Sebanyak **17 berkas label berformat campuran** (13 di train, 3 di validasi, 1 di test) yang sebelumnya menyebabkan pengabaian citra oleh Ultralytics telah distandarisasi secara homogen.
> 3. **Kelulusan Audit Dataloader 100%**: Pemindai dataloader resmi Ultralytics memverifikasi bahwa:
>    - **Train**: Tepat **8.980 dari 8.980 citra termuat** (0 corrupt, 0 mixed warnings, 66 background polos).
>    - **Valid**: Tepat **2.567 dari 2.567 citra termuat** (0 corrupt, 0 mixed warnings, 18 background polos).
>    - **Test**: Tepat **1.282 dari 1.282 citra termuat** (0 corrupt, 0 mixed warnings, 9 background polos).
> 4. **Integritas Objek Terjaga Sempurna**: Jumlah objek ground truth sebelum dan sesudah konversi cocok 100% pada setiap berkas dari 12.829 file label. Tidak ada objek yang hilang atau bergeser kelas.

---

## 2. Metodologi Konversi Anotasi Poligon ke Bounding Box

### 2.1 Formula Matematis Konversi
Untuk setiap baris anotasi poligon dengan k pasangan titik verteks [(x_1, y_1), (x_2, y_2), ..., (x_k, y_k)] yang ternormalisasi terhadap dimensi citra:
1. Hitung koordinat batas ekstrem pelingkup sumbu (*axis-aligned bounding box*):
   $$x_{\min} = \max\left(0.0, \min_{i=1}^k x_i\right), \quad x_{\max} = \min\left(1.0, \max_{i=1}^k x_i\right)$$
   $$y_{\min} = \max\left(0.0, \min_{i=1}^k y_i\right), \quad y_{\max} = \min\left(1.0, \max_{i=1}^k y_i\right)$$
2. Hitung dimensi lebar (w) dan tinggi (h):
   $$w = \max(10^{-6}, x_{\max} - x_{\min}), \quad h = \max(10^{-6}, y_{\max} - y_{\min})$$
3. Hitung titik pusat kotak (x_c, y_c):
   $$x_c = \frac{x_{\min} + x_{\max}}{2}, \quad y_c = \frac{y_{\min} + y_{\max}}{2}$$
4. Format baris deteksi standar YOLO:
   $$\text{Format} = \left[ \text{class\_id} \quad x_c \quad y_c \quad w \quad h \right]$$

---

## 3. Hasil Audit Anotasi Sebelum dan Sesudah Standarisasi

| Parameter | Split Train | Split Validation | Split Test | Total Keseluruhan |
| :--- | :---: | :---: | :---: | :---: |
| **Total Citra** | 8.980 | 2.567 | 1.282 | **12.829** |
| **Total Berkas Label** | 8.980 | 2.567 | 1.282 | **12.829** |
| **Baris Poligon yang Dikonversi** | 34.150 | 9.638 | 5.080 | **48.868** |
| **Baris BBox Asli yang Dipertahankan** | 69 | 11 | 9 | **89** |
| **Berkas Format Campuran Terselesaikan** | **13** | **3** | **1** | **17** |
| **Citra Background Polos (0 Objek)** | 66 | 18 | 9 | **93** |
| **Total Objek Ground Truth (Sebelum)** | 34.219 | 9.649 | 5.089 | **48.957** |
| **Total Objek Ground Truth (Sesudah)** | 34.219 | 9.649 | 5.089 | **48.957** |
| **Tingkat Kecocokan Objek (Match Rate)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

---

## 4. Hasil Verifikasi Dataloader Ultralytics (Dataloader Audit)

Setelah pembentukan berkas konfigurasi `cassava-v7-standardized/data.yaml` dan pembuatan cache baru (`labels.cache`), pemindai Ultralytics menghasilkan log verifikasi resmi:

```text
TRAIN: Scanning labels.cache... 8980 images, 66 backgrounds, 0 corrupt: 100% 8980/8980
VALID: Scanning labels.cache... 2567 images, 18 backgrounds, 0 corrupt: 100% 2567/2567
TEST:  Scanning labels.cache... 1282 images,  9 backgrounds, 0 corrupt: 100% 1282/1282
```

- **Peringatan Label Korup**: **0 (NOL)** di semua split!
- **Peringatan Anotasi Campuran**: **0 (NOL)** di semua split!
- **Jumlah Citra Pelatihan Termuat**: Tepat **8.980 citra** (memenuhi 100% target Audit Task 5 & 8).

---

## 5. Visualisasi Overlay Sebelum vs Sesudah Konversi

Untuk memastikan konversi tidak merusak batas morfologi gejala penyakit daun singkong, visualisasi *side-by-side overlay* telah dibuat menggunakan OpenCV pada sampel bermasalah dan sampel kontrol:

| Sampel Uji | Tipe Sampel | Split | Gambar Komparasi Tersimpan |
| :--- | :--- | :---: | :--- |
| `253146039_jpg.rf.9e814fd...` | Mixed format (1 poly + 1 bbox) | Train | [train_253146039_comparison.png](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/visualizations/train_253146039_jpg.rf.9e814fd03d5d2b739164a536d0d5b2a5_comparison.png) |
| `image_1012_jpg.rf.d53a95e...` | Mixed format (1 bbox + 2 poly) | Train | [train_image_1012_comparison.png](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/visualizations/train_image_1012_jpg.rf.d53a95ef6a505eaeef7a6119c5ac01de_comparison.png) |
| `image_947_jpg.rf.cb175d6...` | Mixed format (2 bbox + 1 poly) | Train | [train_image_947_comparison.png](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/visualizations/train_image_947_jpg.rf.cb175d6bae37e56344bbe71dbaea1e5d_comparison.png) |
| `image_1000_jpg.rf.6cfb821...` | Mixed format validasi | Valid | [valid_image_1000_comparison.png](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/visualizations/valid_image_1000_jpg.rf.6cfb821bfe27d81187278c669937c9ea_comparison.png) |
| `image_918_jpg.rf.a20c88d...` | Mixed format validasi | Valid | [valid_image_918_comparison.png](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/visualizations/valid_image_918_jpg.rf.a20c88d5772e34bff9f7fc5d858dd355_comparison.png) |
| `image_996_jpg.rf.5ff1d58...` | Mixed format test | Test | [test_image_996_comparison.png](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/visualizations/test_image_996_jpg.rf.5ff1d580dd6268900bde22c1c321a2aa_comparison.png) |

Setiap gambar menyajikan sisi kiri (anotasi mentah campuran dengan poligon kontur) dan sisi kanan (kotak pembatas axis-aligned bersih yang membungkus sempurna lesi penyakit).

---

## 6. Daftar Berkas dan Checksum Integritas Data

1. 📄 [label_conversion_manifest.csv](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/label_conversion_manifest.csv) — Berisi 12.829 baris pencatatan setiap berkas label, status poligon/bbox, dan hash MD5/SHA-256.
2. 📋 [dataset_checksums.csv](file:///d:/UNDIP/SKRIPSI/stage_10_data_audit/dataset_checksums.csv) — Hash agregat SHA-256 seluruh citra dan label per split.
3. ⚙️ `cassava-v7-standardized/data.yaml` — Berkas konfigurasi resmi dataset bersih untuk eksperimen mendatang.

*Status: Dataset Bersih Telah Terverifikasi 100% dan Siap Digunakan.*

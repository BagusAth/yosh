# STAGE 9.3 — CODE FIX AND REGRESSION TESTING REPORT

**Penelitian**: Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong Berbasis YOLO11n dengan *Error-Driven Adaptive Optimization*  
**Institusi**: Departemen Ilmu Komputer / Informatika, Universitas Diponegoro  
**Peran**: Senior ML Research Engineer  
**Tanggal Eksekusi**: 15 Oktober 2026  
**Status Akhir**: **`READY_FOR_CLEAN_PIPELINE`**  

---

## 1. Ringkasan Eksekutif

Tahap Stage 9.3 berhasil menyelesaikan seluruh temuan audit kode dari Stage 9.2 (CRITICAL-01, HIGH-01, HIGH-02, MED-01, MED-02, dan LOW-01). Perbaikan kode diterapkan secara terverifikasi pada 6 skrip penelitian dengan preservasi penuh terhadap artefak eksperimen historis (Stage 7–9.2).

Kebenaran fungsional dan integritas matematika dari setiap patch telah dibuktikan melalui rangkaian **Regression Testing 4-Tingkat** dengan total **16 uji regresi otomatis**, mencakup static analysis, pengujian unit matematis/geometris, verifikasi interaksi komponen eksperimen (EXP-0 s.d. EXP-3), dan pengujian smoke pada dataloader citra riil tanpa menjalankan *full training*.

### Matriks Status Defect

| Defect ID | Tingkat Keparahan | Berkas Target | Deskripsi Masalah | Status Akhir | Bukti Verifikasi |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **CRITICAL-01** | CRITICAL | `stage_8_3_adaptive_pipeline.py` | Silent NO-OP pada `apply_context_crop()` karena akses `label.get("bboxes")` yang salah. | **RESOLVED** | 8 Unit Tests Pass; Crop aktif pada sampel riil; Patch 640x640; Diagnostik terekam. |
| **HIGH-01** | HIGH | `stage_8_4_run_ablation_experiments.py`<br>`stage_8_6_final_test_benchmark.py` | Vektor `box.maps` (mAP@50-95 per kelas) salah diberi label sebagai mAP@50 per kelas. | **RESOLVED** | Ekstraksi terpisah `per_class_ap50` dan `per_class_map50_95`; Rekonsiliasi mean lulus. |
| **HIGH-02** | HIGH | `stage_8_5_model_selection_and_validation.py` | Aturan seleksi model memakai margin 0.5% yang mengizinkan model ber-mAP@50-95 lebih rendah menang. | **RESOLVED** | Enforced primary dominance mAP@50-95; Tie-break hanya aktif pada selisih $\le 10^{-5}$. |
| **MED-01** | MEDIUM | `stage_7_baseline_and_error_analysis.py` | Evaluator Stage 7 berpotensi memasangkan satu prediksi ke beberapa ground-truth (pred reuse). | **RESOLVED** | Implementasi 1-to-1 bipartite greedy matching berbasis descending IoU; Pred reuse 0. |
| **MED-02** | MEDIUM | `stage_7`, `stage_8_2`, `stage_8_5`, `stage_8_6` | Asumsi dimensi gambar konstan 640.0 pada konversi koordinat kotak pembatas. | **RESOLVED** | Koordinat dinamis `(img_width, img_height)`; Sumbu X memakai W, sumbu Y memakai H. |
| **LOW-01** | LOW | `stage_8_5_model_selection_and_validation.py` | Escape sequence `\l` pada LaTeX markdown memicu `SyntaxWarning` pada Python 3.12+. | **RESOLVED** | Sintaks dibersihkan; Skrip lulus `python -W error -m py_compile` dengan 0 warning. |

---

## 2. Audit Preservasi dan Lingkungan Runtime

### 2.1 Lingkungan Runtime Terdokumentasi
- **Sistem Operasi**: Windows 11 Home Single Language (Build 26100)
- **Python**: 3.13.7 (64-bit)
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4, cuDNN 90100)
- **Akselerator GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.188 MiB VRAM)
- **Ultralytics**: 8.4.173
- **OpenCV**: 4.13.0

### 2.2 Integritas Berkas Historis
Sebelum manipulasi kode dilakukan, seluruh salinan *pristine* dari ke-7 berkas target telah diamankan di direktori:  
`stage_9_3_code_fix/pre_fix_backup/`  

Semua hash SHA-256 dicatat secara ketat pada [code_changes_manifest.csv](file:///d:/UNDIP/SKRIPSI/stage_9_3_code_fix/code_changes_manifest.csv). Dataset `cassava-v7-standardized` dan `cassava-v7-research`, serta artefak historis `stage_8_ablation/` tidak disentuh atau ditimpa.

---

## 3. Rincian Teknis Perbaikan Kode

### 3.1 Perbaikan CRITICAL-01 (`apply_context_crop`)

#### Masalah & Akar Penyebab
Pada Ultralytics 8.4.173, kamus `label` sebelum melewati `Format()` tidak memiliki kunci `"bboxes"`. Seluruh kotak pembatas tersimpan di dalam objek `Instances` pada `label["instances"]`. Akibatnya, `label.get("bboxes")` selalu menghasilkan `None`, memicu *early return* tanpa transformasi sama sekali.

#### Solusi Implementasi
1. Mengimpor `Instances` dari `ultralytics.utils.instance`.
2. Mengakses kotak melalui `label["instances"].bboxes` (format normalized `xywh`).
3. Mengonversi koordinat target lesion $i^*$ ke koordinat absolut kanvas dan menghitung perluasan jendela $2\times$ dimensi lesi:
   $$\text{crop\_window} = 2.0 \times (w_i, h_i)$$
4. Melakukan *clamping* koordinat jendela crop pada batas citra $[0, W] \times [0, H]$ dan memotong patch citra menggunakan NumPy slicing, lalu me-resize ke $640 \times 640$ via `cv2.resize`.
5. Menghitung transformasi kotak pembatas target dan sekunder ke dalam ruang koordinat patch baru:
   $$x_{\text{crop}} = (x_{\text{orig}} - x_{\min}) \times \frac{640}{w_{\text{crop}}}, \quad y_{\text{crop}} = (y_{\text{orig}} - y_{\min}) \times \frac{640}{h_{\text{crop}}}$$
6. Melakukan penyaringan lesi sekunder dengan batas visibilitas area minimal $\ge 0.30$. Lesi dengan sisa area $< 0.30$ dibuang.
7. Mengemas kembali kotak ke dalam objek `Instances(bboxes=..., bbox_format="xywh", normalized=True)` baru.
8. Menghapus kunci internal `_is_context_cropped` sebelum masuk ke `crop_transforms` agar tidak menyebabkan pergeseran urutan kunci kamus pada `collate_fn`.
9. Menambahkan pencatatan diagnostik terinci ke kamus `self.crop_diagnostics`.

---

### 3.2 Perbaikan HIGH-01 (Pemisahan AP Per-Kelas)

#### Masalah & Akar Penyebab
Pada `stage_8_4_run_ablation_experiments.py` baris 236 dan `stage_8_6_final_test_benchmark.py` baris 143, skrip mengambil `val_metrics.box.maps` dan menyimpannya ke kamus `per_class_map50`. Di Ultralytics, `box.maps` adalah mAP@50-95 per kelas ($\frac{1}{10}\sum_{\text{IoU}=0.50}^{0.95} \text{AP}_{\text{IoU}}$), sedangkan AP@50 per kelas tersimpan di `box.ap50`.

#### Solusi Implementasi
1. Mengekstrak `per_class_ap50` dari `val_metrics.box.ap50`.
2. Mengekstrak `per_class_map50_95` dari `val_metrics.box.maps`.
3. Menyimpan kedua entri secara independen pada luaran JSON dan menyajikan dua tabel terpisah pada laporan komparasi ablasi:
   - Tabel AP@50 per-kelas dengan kolom rata-rata `Mean AP@50` (yang identik dengan `box.map50`).
   - Tabel AP@50-95 per-kelas dengan kolom rata-rata `Mean AP@50-95` (yang identik dengan `box.map`).
4. Menyediakan alias backward-compatibility `per_class_mAP50` yang merujuk ke data AP@50 yang benar.

---

### 3.3 Perbaikan HIGH-02 (Protokol Seleksi Model Resmi)

#### Masalah & Akar Penyebab
Skrip `stage_8_5_model_selection_and_validation.py` menyertakan toleransi `(top_candidate["mAP50_95"] - r["mAP50_95"]) <= 0.005` (margin 0.50%). Aturan ini memungkinkan model dengan performa deteksi umum (mAP@50-95) lebih rendah memenangkan seleksi model utama hanya karena memiliki nilai `recall_small` lebih tinggi. Hal ini bertentangan dengan metodologi formal penelitian di mana metrik seleksi utama adalah mAP@50-95 validasi tertinggi.

#### Solusi Implementasi
1. Menghapus seluruh margin toleransi arbitrer 0.50%.
2. Menetapkan dominasi mutlak mAP@50-95 validasi: kandidat dengan mAP@50-95 tertinggi SELALU terpilih.
3. Kriteria *tie-breaking* (mAP@50 $\rightarrow$ Recall_Small $\rightarrow$ False Negative terendah) HANYA dievaluasi jika selisih numerik antar-kandidat berada dalam batas floating-point tie ($\le 10^{-5}$).
4. Setiap keputusan seleksi menyertakan catatan rasional eksplisit dan ringkasan metrik seluruh kandidat.

---

### 3.4 Perbaikan MED-01 (One-to-One Bipartite Matching Stage 7)

#### Masalah & Akar Penyebab
Pada `stage_7_baseline_and_error_analysis.py`, loop evaluasi membiarkan setiap objek ground-truth mencari prediksi dengan IoU terbesar tanpa menandai prediksi yang telah diambil oleh objek ground-truth lain. Hal ini membuka peluang satu prediksi yang berada di antara dua lesi berdekatan dihitung sebagai True Positive untuk kedua lesi sekaligus.

#### Solusi Implementasi
Mengimplementasikan algoritma *one-to-one bipartite greedy matching*:
1. Mengumpulkan seluruh pasangan `(IoU, gt_idx, pred_idx)` dengan $\text{IoU} \ge 0.10$.
2. Mengurutkan daftar pasangan berdasarkan IoU secara *descending*.
3. Mengalokasikan pasangan secara eksklusif ke dalam `matched_gt_map` dan `assigned_preds`. Objek GT atau Prediksi yang telah teralokasi tidak dapat dipasangkan kembali.
4. Objek GT yang tidak mendapatkan pasangan langsung diklasifikasikan sebagai `False_Negative_Missed` ($\text{IoU} = 0.0, \text{conf} = 0.0$).

---

### 3.5 Perbaikan MED-02 (Dimensi Dinamis Koordinat BBox)

#### Masalah & Akar Penyebab
Fungsi `parse_yolo_labels` pada `stage_7`, `stage_8_2`, `stage_8_5`, dan `stage_8_6` mengalikan koordinat normalized dengan skalar tetap `640`. Jika skrip mengevaluasi citra non-persegi atau beresolusi berbeda, terjadi distorsi penskalaan kotak pembatas.

#### Solusi Implementasi
Mengubah tanda tangan fungsi menjadi:
`parse_yolo_labels(lbl_path, img_width=640.0, img_height=640.0)`
Koordinat horizontal ($x_c, w$) selalu diskalakan dengan `img_width`, sedangkan koordinat vertikal ($y_c, h$) selalu diskalakan dengan `img_height`. Nilai default `640.0` dipertahankan untuk memastikan *backward compatibility* 100% pada dataset singkong 640×640 saat ini.

---

### 3.6 Perbaikan LOW-01 (SyntaxWarning Python 3.12+)

#### Masalah & Akar Penyebab
Karakter LaTeX `$\le 0.50\%$` pada multiline f-string baris 393 `stage_8_5_model_selection_and_validation.py` menyebabkan `SyntaxWarning: invalid escape sequence '\l'`.

#### Solusi Implementasi
Mengganti string dengan format kompatibel tanpa *invalid escape sequence*. Verifikasi dengan `python -W error -m py_compile` pada seluruh skrip Python menghasilkan nol peringatan/kesalahan.

---

## 4. Hasil Pengujian Regresi Komprehensif (Level 1–4)

Pengujian regresi dijalankan secara otomatis melalui [test_stage_9_3_regression.py](file:///d:/UNDIP/SKRIPSI/test_stage_9_3_regression.py). Luaran lengkap terekam di [regression_test_results.txt](file:///d:/UNDIP/SKRIPSI/stage_9_3_code_fix/regression_test_results.txt) dan matriks pengujian di [regression_test_matrix.csv](file:///d:/UNDIP/SKRIPSI/stage_9_3_code_fix/regression_test_matrix.csv).

```
================================================================================
STAGE 9.3 — REGRESSION TEST SUITE EXECUTION REPORT
================================================================================
Date/Time          : 2026-10-15 19:49:21
Python Version     : 3.13.7
PyTorch Version    : 2.6.0+cu124
CUDA Available     : True
Total Tests Run    : 16
Failures           : 0
Errors             : 0
Execution Duration : 2.35 seconds
Overall Outcome    : PASS
================================================================================
```

### Rincian Hasil Tiap Test Suite

1. **`TestCritical01ContextCrop` (8 Kasus Uji Geometric & Mathematical — PASS)**
   - `test_case_1_center_target`: Target di tengah menghasilkan patch crop $640\times 640$ dengan kotak target tepat di tengah ($0.5, 0.5$) dan dimensi $2\times$ lebih besar.
   - `test_case_2_edges_and_corners`: Target di tepi pojok kiri-atas dan kanan-bawah dipotong dengan *clamping* kanvas aman tanpa nilai koordinat negatif atau $> 1.0$.
   - `test_case_3_no_target_objects`: Citra tanpa target lesi sulit di-skip dengan benar (`no_target_skips: 1`).
   - `test_case_4_multiple_objects_and_visibility_threshold`: Lesi sekunder di dalam crop dipertahankan, lesi di luar dan lesi tepi dengan visibilitas $< 30\%$ dibuang secara akurat (`visibility_skips >= 2`).
   - `test_case_5_invalid_boxes`: Kotak $w \le 0$ atau $h \le 0$ ditangani secara aman (`invalid_box_skips: 1`).
   - `test_case_6_empty_labels`: Label kosong tidak memicu *exception* dan tercatat pada penghitung diagnostik.
   - `test_case_7_manual_calculation_verification`: Luaran transformasi koordinat presisi identik hingga 4 desimal dengan kalkulasi manual.
   - `test_case_8_original_label_immutability`: Objek `label["instances"].bboxes` input terbukti tidak termutasi secara *in-place*.

2. **`TestHigh01CanonicalMetrics` (Reconciliation Test — PASS)**
   - Menguji bahwa rata-rata `per_class_ap50` sama persis dengan `val_metrics.box.map50` (toleransi numerik $< 10^{-6}$).
   - Menguji bahwa rata-rata `per_class_map50_95` sama persis dengan `val_metrics.box.map` (toleransi numerik $< 10^{-6}$).
   - Terbukti tidak terjadi pertukaran metrik (*no cross-metric conflation*).

3. **`TestHigh02SelectionProtocol` (Decision Rule Test — PASS)**
   - `test_higher_map50_95_always_wins`: Model dengan mAP@50-95 lebih tinggi selalu menang, membuktikan margin 0.5% telah dieliminasi.
   - `test_strict_tie_break_hierarchy`: Pada kondisi seri matematis ($|\Delta| \le 10^{-5}$), sistem mengevaluasi hierarki tie-break mAP@50 $\rightarrow$ Recall_Small $\rightarrow$ Lowest FN secara deterministik.

4. **`TestMed01BipartiteMatching` (One-to-One Matching Test — PASS)**
   - `test_adjacent_gts_single_prediction_no_reuse`: Dua objek GT berdekatan di dekat satu prediksi hanya mengizinkan 1 GT yang cocok (IoU tertinggi). GT kedua dipastikan menjadi `False_Negative_Missed`, membuktikan tidak ada *prediction reuse*.

5. **`TestMed02CoordinateParser` (Dynamic Dimensions Test — PASS)**
   - `test_dynamic_dimensions`: Pada resolusi $1920 \times 1080$, sumbu horizontal memakai 1920 dan vertikal memakai 1080 dengan luaran piksel yang tepat.
   - `test_backward_compat_default_640`: Panggilan default tanpa argumen dimensi menghasilkan koordinat identik 640×640.

6. **`TestComponentIntegrationAndSmoke` (Integration & Smoke Tests — PASS)**
   - `test_component_toggles_matrix`: Mengonfirmasi konfigurasi EXP-0 s.d. EXP-3 mengaktifkan dan menonaktifkan sampling adaptif serta targeted context augmentation secara presisi sesuai matriks desain eksperimen.
   - `test_smoke_dataloader_and_runtime_crop`: Memuat batch riil dari dataset `cassava-v7-research` melalui `AdaptiveYOLODataset` dan `InfiniteDataLoader`. Batch berhasil dikolasi menjadi tensor `torch.Size([4, 3, 640, 640])` dengan diagnostik runtime terekam.

---

## 5. Bukti Diagnostik Runtime Targeted Context Crop

Dari eksekusi smoke test pada batch citra riil dataset singkong standar, diagnostik dicatat pada [runtime_crop_diagnostics.csv](file:///d:/UNDIP/SKRIPSI/stage_9_3_code_fix/runtime_crop_diagnostics.csv):

| Metric Diagnostik | Nilai Aktual | Keterangan Verifikasi |
| :--- | :---: | :--- |
| `crop_attempts` | 1 | Percobaan crop terpicu oleh adanya hard small lesion pada batch |
| `crops_applied` | 1 | Transformasi crop kontekstual $2\times$ berhasil diaplikasikan secara aktif |
| `no_target_skips` | 0 | Tidak ada skip akibat ketidakhadiran target |
| `invalid_box_skips` | 0 | Seluruh kotak lesi valid secara geometris |
| `visibility_skips` | 5 | 5 lesi sekunder yang terpotong di bawah threshold 30% berhasil dibuang |

Hasil ini membuktikan secara empiris bahwa **`apply_context_crop()` BUKAN lagi silent NO-OP**, melainkan aktif memotong citra dan mentransformasikan koordinat kotak pembatas.

---

## 6. Daftar Skrip yang Siap Digunakan (Clean Pipeline Ready)

Seluruh skrip berikut telah diverifikasi aman, bebas cacat, dan siap digunakan untuk regenerasi OOF dan training ablasi bersih:

1. [`stage_6_group_aware_resplit.py`](file:///d:/UNDIP/SKRIPSI/stage_6_group_aware_resplit.py) — Partisi dataset bebas kebocoran kelompok.
2. [`stage_7_baseline_and_error_analysis.py`](file:///d:/UNDIP/SKRIPSI/stage_7_baseline_and_error_analysis.py) — Baseline YOLO11n dengan pencocokan bipartite 1-ke-1.
3. [`stage_8_2_build_difficulty_manifest.py`](file:///d:/UNDIP/SKRIPSI/stage_8_2_build_difficulty_manifest.py) — Profil kesulitan objek dan citra dengan koordinat dinamis.
4. [`stage_8_3_adaptive_pipeline.py`](file:///d:/UNDIP/SKRIPSI/stage_8_3_adaptive_pipeline.py) — Dataloader adaptif dan Targeted Context-Preserving Crop aktif.
5. [`stage_8_4_run_ablation_experiments.py`](file:///d:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py) — Runner ablasi EXP-0 s.d. EXP-3 dengan metrik kanonik terpisah.
6. [`stage_8_5_model_selection_and_validation.py`](file:///d:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py) — Protokol seleksi model resmi (dominasi mAP@50-95).
7. [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py) — Benchmark satu pintu Locked Test Set dengan metrik kanonik.

---

## 7. Keputusan Resmi & Rekomendasi Selanjutnya

### Keputusan: **`READY_FOR_CLEAN_PIPELINE`**

**Dasar Keputusan**:
1. Temuan berkategori **CRITICAL** (`CRITICAL-01`) telah diperbaiki secara tuntas dan dibuktikan aktif melalui pengujian unit dan pengujian smoke dataloader.
2. Seluruh temuan berkategori **HIGH** (`HIGH-01`, `HIGH-02`) telah direkonsiliasi dan diselaraskan penuh dengan protokol formal penelitian.
3. Temuan pendukung **MEDIUM** (`MED-01`, `MED-02`) dan **LOW** (`LOW-01`) telah diperbaiki dan lulus pengujian regresi.
4. Rangkaian uji regresi 16/16 lulus dengan tingkat keberhasilan 100% tanpa regresi baru.
5. Seluruh artefak historis tetap utuh dan terlindungi dari modifikasi.

### Rekomendasi Tahap Berikutnya (Stage 10 / Clean Training Run):
1. **Regenerasi OOF Profiling (Stage 8.1–8.2)**: Jalankan profiling 4-fold OOF pada dataset bersih `cassava-v7-research` untuk menghasilkan manifest kesulitan yang segar.
2. **Eksekusi 4 Eksperimen Bersih (Stage 8.4)**:
   - EXP-0: Baseline YOLO11n (Uniform Sampling + Augmentasi Standar)
   - EXP-1: Mixture Adaptive Sampling ($P(k)$) + Augmentasi Standar
   - EXP-2: Uniform Sampling + Targeted Context Augmentation
   - EXP-3: Full Proposed Method ($P(k)$ + Targeted Context Augmentation)
3. **Seleksi Model & Evaluasi Akhir (Stage 8.5–8.6)**: Lakukan seleksi model validasi kanonik dan evaluasi satu kali pada locked test set 1.282 citra untuk memperoleh hasil final naskah skripsi Bab 4.

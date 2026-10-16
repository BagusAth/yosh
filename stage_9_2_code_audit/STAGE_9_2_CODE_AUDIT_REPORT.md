# LAPORAN AUDIT KODE & PIPELINE KOMPREHENSIF (STAGE 9.2)
## *Comprehensive Code, Pipeline, and Mathematical Contract Audit*

**Proyek Skripsi**: Optimasi Model YOLO11n Berbasis Analisis Error dan Tingkat Kesulitan Objek untuk Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong  
**Peran Auditor**: Senior ML Research Engineer & Code Auditor  
**Workspace**: `D:\UNDIP\SKRIPSI`  
**Dataset Standar**: [`cassava-v7-standardized`](file:///D:/UNDIP/SKRIPSI/cassava-v7-standardized) (100% homogen 5-kolom YOLO)  
**Tanggal Audit**: 2026-10-14  
**Mode Audit**: **STRICT READ-ONLY** (Tidak ada skrip sumber, checkpoint, anotasi, atau konfigurasi yang diubah)  

---

## 1. Ringkasan Eksekutif & Keputusan Status Akhir

Berdasarkan audit statis, pengujian unit matematis dengan data sintetis, verifikasi integrasi antarmodul, dan audit aliran data Ultralytics 8.4, tahap **Stage 9.2** menghasilkan keputusan status resmi:

> ### 🔴 KEPUTUSAN STATUS AUDIT: `NEEDS_CODE_FIX`
> 
> **Pernyataan Penilaian Kritis**:
> Meskipun pipeline data Stage 6 (split tanpa kebocoran) dan Stage 8.1–8.2 (OOF profiling dan manifest kesulitan) terbukti valid secara matematis, audit menemukan **1 Temuan Kritis (CRITICAL)** dan **2 Temuan Tinggi (HIGH)** pada pipeline eksperimen Stage 8.3–8.5:
>
> 1. **[CRITICAL] Pemicu Augmentasi Terarah Mengalami Silent NO-OP ([FINDING-CRIT-01](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/code_findings.csv))**:
>    Pada [`stage_8_3_adaptive_pipeline.py`](file:///D:/UNDIP/SKRIPSI/stage_8_3_adaptive_pipeline.py#L197), metode `apply_context_crop()` memanggil `label.get("bboxes")`. Pada arsitektur internal Ultralytics 8.4, kamus `label` sebelum transformasi **tidak memiliki kunci `"bboxes"`**, melainkan menyimpan objek instans di `label["instances"].bboxes`. Akibatnya, `orig_boxes` selalu bernilai array kosong (`len == 0`), `_is_context_cropped` tidak pernah aktif (`False`), dan fungsi mengembalikan sampel tanpa crop. **Dengan demikian, pada eksperimen historis EXP-2 dan EXP-3, Targeted Context Augmentation tidak pernah dieksekusi selama training (berjalan sebagai replika baseline murni).**
>
> 2. **[HIGH] Per-Class AP@50-95 Salah Dicatat sebagai AP@50 ([FINDING-HIGH-01](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/code_findings.csv))**:
>    Pada [`stage_8_4_run_ablation_experiments.py`](file:///D:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py#L236) dan [`stage_8_6_final_test_benchmark.py`](file:///D:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py#L143), atribut `val_metrics.box.maps` (yang merupakan representasi COCO AP@[50:95] per kelas) dicatat ke dalam kamus `per_class_map50`, sehingga metrik AP@50-95 ditampilkan pada kolom AP@50 di laporan ringkasan. (Telah direkonsiliasi pada Stage 9.1).
>
> 3. **[HIGH] Deviasi Aturan Seleksi Model terhadap Protokol Validasi ([FINDING-HIGH-02](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/code_findings.csv))**:
>    Pada [`stage_8_5_model_selection_and_validation.py`](file:///D:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py#L345), kondisi ambang batas ad-hoc $\le 0.5\%$ mengaktifkan tie-breaking yang memenangkan EXP-2 atas EXP-0, bertentangan dengan protokol seleksi hierarkis resmi yang menetapkan validasi mAP@50-95 murni ($\text{EXP-0} = 26,58\% > \text{EXP-2} = 26,24\%$).
>
> **Kebijakan Eksekusi**: Sesuai mandat penelitian, **pelatihan bersih (Clean Training) TIDAK BOLEH dimulai** sebelum temuan `CRITICAL` dan `HIGH` diperbaiki pada kode sumber dan divalidasi dengan uji regresi.

---

## 2. Bagian A — Inventarisasi Kode dan Dependensi Lingkungan

### 2.1 Lingkungan Runtime Eksekusi
Berdasarkan berkas manifes [`dependency_versions.txt`](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/dependency_versions.txt):
- **Sistem Operasi**: Windows 11 / 10 x64 (AMD64)
- **Python Runtime**: `3.13.7` (tags/v3.13.7:bcee1c3, Aug 14 2025)
- **PyTorch**: `2.6.0+cu124` (CUDA 12.4 terakselerasi aktif)
- **Perangkat Keras**: NVIDIA GeForce RTX 4060 Laptop GPU (VRAM 8GB)
- **Ultralytics**: `8.4.173`
- **NumPy**: `2.5.3`
- **Pandas**: `3.0.6`
- **OpenCV**: `5.0.0.93`
- **SciPy & Albumentations**: *Not Installed* (semua operasi augmentasi dan spasial bergantung pada internal Ultralytics dan OpenCV murni).

### 2.2 Inventarisasi Berkas Skrip & Konfigurasi
Seluruh skrip dari Stage 6 hingga Stage 9.1 telah dipetakan pada [`code_inventory.csv`](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/code_inventory.csv):

| Nama Modul / Skrip | Tahap | LOC | Kelas Utama | Fungsi Kunci | Status Audit |
| :--- | :---: | :---: | :--- | :--- | :---: |
| [`stage_6_group_aware_resplit.py`](file:///D:/UNDIP/SKRIPSI/stage_6_group_aware_resplit.py) | Stage 6 | 389 | - | `stratified_group_split`, `link_or_copy` | 🟢 PASS |
| [`verify_stage_6_compliance.py`](file:///D:/UNDIP/SKRIPSI/verify_stage_6_compliance.py) | Stage 6 | 313 | - | `run_verification`, `parse_yolo_file` | 🟢 PASS |
| [`post_split_audit.py`](file:///D:/UNDIP/SKRIPSI/post_split_audit.py) | Stage 6 | 546 | - | `find_cross_visual_pairs`, `hamming_hex` | 🟢 PASS |
| [`execute_phase_b_standardization.py`](file:///D:/UNDIP/SKRIPSI/execute_phase_b_standardization.py) | Stage 9.1 | 293 | - | `convert_polygon_to_bbox`, `standardize` | 🟢 PASS |
| [`verify_standardized_dataloader.py`](file:///D:/UNDIP/SKRIPSI/verify_standardized_dataloader.py) | Stage 9.1 | 39 | - | Ultralytics DataLoader Scanner | 🟢 PASS |
| [`stage_7_baseline_and_error_analysis.py`](file:///D:/UNDIP/SKRIPSI/stage_7_baseline_and_error_analysis.py) | Stage 7 | 531 | - | `calculate_iou_xyxy`, `parse_yolo_labels` | 🟡 REVIEW |
| [`stage_8_1_group_oof_profiling.py`](file:///D:/UNDIP/SKRIPSI/stage_8_1_group_oof_profiling.py) | Stage 8.1 | 709 | - | `build_group_stratified_folds`, `train_and_predict_fold` | 🟢 PASS |
| [`stage_8_2_build_difficulty_manifest.py`](file:///D:/UNDIP/SKRIPSI/stage_8_2_build_difficulty_manifest.py) | Stage 8.2 | 571 | - | `greedy_one_to_one_matching`, `process_difficulty_manifest` | 🟢 PASS |
| [`stage_8_3_adaptive_pipeline.py`](file:///D:/UNDIP/SKRIPSI/stage_8_3_adaptive_pipeline.py) | Stage 8.3 | 444 | `AdaptiveYOLODataset`, `AdaptiveDetectionTrainer` | `get_sampling_weights`, `apply_context_crop`, `get_dataloader` | 🔴 DEFECT |
| [`stage_8_4_run_ablation_experiments.py`](file:///D:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py) | Stage 8.4 | 379 | - | `run_single_experiment`, `main` | 🟡 REVIEW |
| [`stage_8_5_model_selection_and_validation.py`](file:///D:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py) | Stage 8.5 | 428 | - | `evaluate_model_on_validation`, `main` | 🟡 REVIEW |
| [`stage_8_6_final_test_benchmark.py`](file:///D:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py) | Stage 8.6 | 363 | - | `evaluate_test_set`, `main` | 🟡 REVIEW |

---

## 3. Bagian B — Audit Dataset dan Konsistensi Koordinat

### 3.1 Verifikasi Resolusi dan Aspek Rasio Citra Aktual
Pengujian empiris ([TEST-02](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/audit_test_results.txt)) terhadap seluruh **12.829 berkas citra** menghasilkan:
- **Train Split (8.980 citra)**: Tepat **$640 \times 640$ piksel** ($100\%$).
- **Valid Split (2.567 citra)**: Tepat **$640 \times 640$ piksel** ($100\%$).
- **Test Split (1.282 citra)**: Tepat **$640 \times 640$ piksel** ($100\%$).
- **Aspek Rasio**: $1.000$ (persegi sempurna) di semua split.

> [!NOTE]
> **Dampak terhadap Audit Portabilitas ([FINDING-MED-02](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/code_findings.csv))**:
> Karena seluruh citra pada dataset singkong ini berdimensi $640 \times 640$, pengalian koordinat ternormalisasi dengan angka konstan `640.0` pada skrip parser kustom tidak menimbulkan distorsi geometris numerik pada dataset saat ini. Namun, untuk portabilitas kode penelitian ke resolusi sembarang, koordinat wajib diskalakan dengan `(W, H)` citra secara dinamis.

### 3.2 Audit Parser Anotasi & Standarisasi Poligon
1. **Rumus Konversi Bounding Box**:
   $$x_{\min} = \max(0.0, \min x_i), \quad x_{\max} = \min(1.0, \max x_i), \quad y_{\min} = \max(0.0, \min y_i), \quad y_{\max} = \min(1.0, \max y_i)$$
   $$xc = \frac{x_{\min} + x_{\max}}{2}, \quad yc = \frac{y_{\min} + y_{\max}}{2}, \quad w = x_{\max} - x_{\min}, \quad h = y_{\max} - y_{\min}$$
   Telah diuji dengan kasus batas (koordinat di tepi `0.0` dan `1.0`, poligon kecil dengan $\Delta < 10^{-5}$) dan terbukti stabil matematis.
2. **Kecocokan Gambar-Label**:
   Sebanyak **48.868 baris poligon** berhasil distandarisasi ke format 5-kolom YOLO. Integritas jumlah objek sebelum dan sesudah konversi cocok **$100,00\%$** (34.219 di train, 9.649 di valid, 5.089 di test).

---

## 4. Bagian C — Audit Stage 8.1: OOF Difficulty Profiling

Pengujian audit isolasi kebocoran ([TEST-03](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/audit_test_results.txt) & [TEST-04](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/audit_test_results.txt)) membuktikan:
1. **Pencegahan Kebocoran Grup 100%**:
   - Total citra latih: 8.980 citra yang berasal dari 5.546 grup atomik (`clean_group_id`).
   - Distribusi fold: Fold 1 (2.246 citra), Fold 2 (2.244 citra), Fold 3 (2.245 citra), Fold 4 (2.245 citra).
   - **Intersection antar-fold untuk grup**: Tepat **0 (NOL)** pada seluruh kombinasi pasangan fold:
     $$\text{Fold}_1 \cap \text{Fold}_2 = \emptyset, \quad \text{Fold}_1 \cap \text{Fold}_3 = \emptyset, \quad \dots \quad \text{Fold}_3 \cap \text{Fold}_4 = \emptyset$$
2. **Integritas Inferensi Diagnostik**:
   - Seluruh **1.402.872 prediksi OOF mentah** pada `oof_predictions_raw.csv` berasal secara eksklusif dari model fold yang tidak pernah melihat grup citra tersebut.
   - Ambang batas diagnostik $\text{conf}=0.001$ berhasil menangkap seluruh deteksi berbobot rendah untuk matching objek.

---

## 5. Bagian D — Audit Stage 8.2: Difficulty Score dan Error Taxonomy

### 5.1 Verifikasi Algoritma Matching
Algoritma pada [`stage_8_2_build_difficulty_manifest.py`](file:///D:/UNDIP/SKRIPSI/stage_8_2_build_difficulty_manifest.py#L204-L238) mengimplementasikan *Greedy Global Bipartite Matching*:
- Pasangan kandidat dengan $\text{IoU} \ge 0.10$ diurutkan secara menurun (*descending*).
- Penugasan eksklusif satu-ke-satu (`assigned_gt` dan `assigned_pred`) menjamin tidak ada objek GT yang berbagi prediksi atau sebaliknya ([TEST-05](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/audit_test_results.txt): PASS).

### 5.2 Verifikasi Formula Tingkat Kesulitan ($D_i$)
Pengujian matematis dengan kalkulasi manual ([TEST-06](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/audit_test_results.txt)):
- **Objek Matched & Kelas Benar**:
  $$D_i = 0.40(1 - C_i) + 0.40(1 - \text{IoU}_i) + 0.20 S_i$$
  Contoh: $C_i = 0.90, \text{IoU} = 1.00, S_i = 0.50 \implies D_i = 0.1400$ (Kode: 0.1400, Selisih: 0.0).
- **Objek Matched & Salah Kelas (Penalti 0.35)**:
  $$D_i = \min\left(1.0, 0.40(1 - C_i) + 0.40(1 - \text{IoU}_i) + 0.20 S_i + 0.35\right)$$
  Contoh: $C_i = 0.80, \text{IoU} = 0.80, S_i = 1.00 \implies D_i = 0.7100$ (Kode: 0.7100, Selisih: 0.0).
- **Objek False Negative (Unmatched)**:
  $$D_i = 0.80 + 0.20 S_i \implies \text{Small FN} = 1.0000, \quad \text{Medium FN} = 0.9000, \quad \text{Large FN} = 0.8000$$

Seluruh formula terbukti terbebas dari pembagian dengan nol, nilai NaN/inf, dan selalu berada dalam rentang $[0.0, 1.0]$.

---

## 6. Bagian E — Audit Stage 8.3: Adaptive Sampling & Targeted Context Crop

### 6.1 Audit Adaptive Sampling
1. **Distribusi Probabilitas Campuran (Mixture Sampling)**:
   $$P(k) = 0.30 \cdot \frac{1}{N} + 0.70 \cdot \frac{D_k}{\sum_{j=1}^N D_j}$$
   - Total bobot sampling pada 8.980 citra latih ter-normalisasi sempurna: $\sum P(k) = 1.000000$.
   - **Floor Probability**: $P_{\min} = 3.584 \times 10^{-5} \ge 3.341 \times 10^{-5}$, menjamin setiap citra tetap memiliki peluang terpilih minimal.
   - **Simulasi Empiris (100.000 penarikan)**: Korelasi empiris terhadap teoretis mencapai **$0.736284$** ([TEST-07](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/audit_test_results.txt)).
2. **Integrasi Dataloader**:
   `AdaptiveDetectionTrainer.get_dataloader` berhasil mengintegrasikan `WeightedRandomSampler` dengan `InfiniteDataLoader` pada mode `train` saja. Dataloader validasi dan uji tetap menggunakan `shuffle=False` deterministik murni.

### 6.2 Audit Kritis Targeted Context Augmentation (TEMUAN UTAMA)

> [!CAUTION]
> ### BUKTI FORENSIK DEFECT: `FINDING-CRIT-01`
> Pada [`stage_8_3_adaptive_pipeline.py`](file:///D:/UNDIP/SKRIPSI/stage_8_3_adaptive_pipeline.py#L197):
> ```python
> # KODE ASLI DI stage_8_3_adaptive_pipeline.py
> orig_boxes = label.get("bboxes", np.zeros((0, 4), dtype=np.float32))
> orig_cls = label.get("cls", np.zeros((0, 1), dtype=np.float32))
> ```
> 
> **Analisis Kerusakan**:
> 1. Pada Ultralytics 8.4, keluaran `self.get_image_and_label(index)` menghasilkan kamus dengan kunci:
>    `['im_file', 'cls', 'img', 'ori_shape', 'resized_shape', 'ratio_pad', 'instances']`
> 2. **Kunci `"bboxes"` TIDAK ADA dalam kamus tersebut.** Koordinat kotak disimpan di dalam objek instans: `label["instances"].bboxes`.
> 3. Karena memanggil `label.get("bboxes", np.zeros((0, 4)))`, `orig_boxes` selalu berupa array kosong berukuran $(0, 4)$.
> 4. Akibatnya, perulangan `for idx in range(len(orig_boxes))` tidak pernah berjalan!
> 5. Blok penanda:
>    ```python
>    if len(new_bboxes) > 0:
>        label["_is_context_cropped"] = True
>    ```
>    **TIDAK PERNAH TEREKSEKUSI.**
> 6. Pada metode `__getitem__`:
>    ```python
>    if label.get("_is_context_cropped", False):
>        return self.crop_transforms(label)
>    return self.transforms(label)  # <-- SELALU JATUH KE SINI!
>    ```
> 7. **Dampak Langsung**: Seluruh citra latih pada EXP-2 dan EXP-3 selalu kembali ke transformasi standar baseline (`self.transforms`). Augmentasi pembesaran lesi kecil $2\times$ yang dirancang tidak pernah diaplikasikan pada citra latih manapun.

---

## 7. Bagian F — Audit Integrasi Trainer dan Studi Ablasi

Berdasarkan temuan `FINDING-CRIT-01`, eksekusi aktual dari keempat konfigurasi eksperimen historis adalah:

| Eksperimen | Rancangan Awal | Eksekusi Riil di GPU | Status Validitas Konsep |
| :---: | :--- | :--- | :---: |
| **EXP-0** | Baseline (Uniform + Std Aug) | Uniform Sampling + Standard Transforms | 🟢 **100% Valid Baseline** |
| **EXP-1** | Adaptive Sampling + Std Aug | Mixture Sampling + Standard Transforms | 🟢 **100% Valid Sampling** |
| **EXP-2** | Uniform + Context Aug | Uniform Sampling + Standard Transforms *(NO-OP)* | 🔴 **Gagal Eksekusi Augmentasi (Menjadi Re-run EXP-0)** |
| **EXP-3** | Adaptive Sampling + Context Aug | Mixture Sampling + Standard Transforms *(NO-OP)* | 🔴 **Gagal Eksekusi Augmentasi (Menjadi Re-run EXP-1)** |

> [!IMPORTANT]
> **Penjelasan Fenomena Metrik Stage 8**:
> Temuan ini secara sempurna menjawab mengapa metrik EXP-0 dan EXP-2 pada validasi sangat mirip (mAP@50: $36,66\%$ vs $36,12\%$, mAP@50-95: $26,58\%$ vs $26,24\%$), serta EXP-1 dan EXP-3 sangat mirip (mAP@50: $34,20\%$ vs $35,09\%$, mAP@50-95: $24,42\%$ vs $25,25\%$). Perbedaan kecil yang muncul semata-mata diakibatkan oleh pergeseran deret bilangan acak (*random state perturbation*) akibat pemanggilan `random.random() < self.aug_prob` di baris 273.

---

## 8. Bagian G — Audit Evaluasi, Pelaporan, dan Seleksi Model

1. **Pemisahan Sumber AP Per-Kelas**:
   Kesalahan ekstraksi `val_metrics.box.maps` ke dalam variabel `per_class_map50` pada skrip evaluasi telah diklasifikasikan sebagai `FINDING-HIGH-01`. Pada Stage 9.1, metrik kanonis telah berhasil dipisahkan menjadi `AP@50` (`val_metrics.box.ap50`) dan `AP@50-95` (`val_metrics.box.maps`).
2. **Koreksi Hierarki Seleksi Model**:
   Hierarki seleksi kandidat model telah dikembalikan ke aturan obyektif validasi mAP@50-95:
   $$\text{Model Pemenang Validasi} = \arg\max_{m} \left(\text{mAP@50-95}_{\text{valid}}\right) = \mathbf{EXP\text{-}0} \quad (26,58\%)$$
   Aturan toleransi margin $0.5\%$ pada baris 345 [`stage_8_5_model_selection_and_validation.py`](file:///D:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py#L345) dinyatakan tidak berlaku untuk penetapan model pemenang.

---

## 9. Matriks Pengujian & Ketertelusuran Metode (Traceability)

Seluruh 13 pengujian multi-level telah didokumentasikan pada [`test_matrix.csv`](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/test_matrix.csv):
- **Level 1 (Static Analysis)**: TEST-01 (PASS), TEST-11 (DEFECT), TEST-12 (DISCREPANCY).
- **Level 2 (Unit Tests)**: TEST-05 (PASS), TEST-06 (PASS), TEST-07 (PASS), TEST-08 (PASS).
- **Level 3 (Integration Tests)**: TEST-02 (PASS), TEST-03 (PASS), TEST-04 (PASS), TEST-09 (DEFECT), TEST-10 (DEFECT).
- **Level 4 (Smoke Tests)**: TEST-13 (PASS).

Seluruh rumus matematika dan kontrak metode telah dipetakan ke fungsi implementasinya pada [`method_to_code_traceability.csv`](file:///D:/UNDIP/SKRIPSI/stage_9_2_code_audit/method_to_code_traceability.csv).

---

## 10. Rekomendasi Perbaikan Kode Konkret Sebelum Clean Training

Sebelum menjalankan eksperimen baru pada dataset bersih [`cassava-v7-standardized`](file:///D:/UNDIP/SKRIPSI/cassava-v7-standardized), modifikasi kode sumber berikut wajib diimplementasikan:

### 10.1 Perbaikan `apply_context_crop()` pada `stage_8_3_adaptive_pipeline.py`
Ubah pembacaan dan pembaruan bounding box agar sesuai dengan struktur `Instances` Ultralytics:
```python
# PERBAIKAN DIREKOMENDASIKAN UNTUK stage_8_3_adaptive_pipeline.py:
from ultralytics.utils.instance import Instances

# 1. Ekstrak bounding box dari label["instances"] (format xywh normalized)
instances = label.get("instances")
if instances is None or len(instances) == 0:
    return label

orig_boxes = instances.bboxes  # ndarray shape (N, 4) in normalized xywh
orig_cls = label.get("cls", np.zeros((len(orig_boxes), 1), dtype=np.float32))

# ... [proses crop geometri dan penyaringan visibilitas] ...

# 2. Perbarui label["instances"] dengan kotak hasil transformasi
if len(new_bboxes) > 0:
    new_bboxes_arr = np.array(new_bboxes, dtype=np.float32)
    label["img"] = resized_crop
    label["cls"] = np.array(new_cls, dtype=np.float32).reshape(-1, 1)
    label["instances"] = Instances(
        bboxes=new_bboxes_arr,
        segments=None,
        keypoints=None,
        bbox_format="xywh",
        normalized=True,
    )
    label["ori_shape"] = (640, 640)
    label["resized_shape"] = (640, 640)
    label["_is_context_cropped"] = True
```

### 10.2 Perbaikan Ekstraksi Metrik pada Skrip Evaluasi
Pada [`stage_8_4_run_ablation_experiments.py`](file:///D:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py) dan [`stage_8_6_final_test_benchmark.py`](file:///D:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py):
```python
# Ekstrak kedua metrik secara tegas dan terpisah
per_class_ap50 = {}
per_class_map50_95 = {}
for idx, cname in enumerate(CLASS_NAMES):
    if hasattr(val_metrics.box, "ap50") and idx < len(val_metrics.box.ap50):
        per_class_ap50[cname] = float(val_metrics.box.ap50[idx])
    if hasattr(val_metrics.box, "maps") and idx < len(val_metrics.box.maps):
        per_class_map50_95[cname] = float(val_metrics.box.maps[idx])
```

---

## 11. Kesimpulan Akhir & Langkah Lanjutan

1. Tahap audit Stage 9.2 telah berhasil menemukan akar permasalahan metodologis paling mendasar dalam pipeline eksperimen penelitian: **Targeted Context Augmentation belum pernah teruji secara nyata karena implementasi pemanggilan kotak yang tidak kompatibel dengan struktur internal Ultralytics 8.4.**
2. Status resmi diputuskan: **`NEEDS_CODE_FIX`**.
3. Sesuai instruksi read-only, kode sumber belum dimodifikasi pada tahap ini. Setelah Anda meninjau laporan ini, langkah selanjutnya adalah memasuki tahap perbaikan kode (Stage 9.3 Code Fix) dan menjalankan unit test verifikasi sebelum memulai pelatihan bersih baru.

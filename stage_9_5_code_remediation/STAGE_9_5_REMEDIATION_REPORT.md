# STAGE 9.5 — TARGETED REMEDIATION, ARTIFACT INTEGRITY, AND INDEPENDENT PIPELINE VERIFICATION REPORT

**Penelitian**: Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong Berbasis YOLO11n dengan Error-Driven Adaptive Optimization  
**Repository**: [yosh](https://github.com/BagusAth/yosh)  
**Workspace**: `D:\UNDIP\SKRIPSI`  
**Commit Dasar Diaudit**: `a74eec540c649959d319a192d77442bff2a0f3f7`  
**Active Working Branch**: `stage-9-5-targeted-remediation`  
**Auditor**: Senior Machine Learning Engineer, Computer Vision Research Engineer, Python Test Engineer, and Independent Research Code Auditor  
**Tanggal Audit**: 17 Oktober 2026  
**Keputusan Gerbang Akhir (Final Gate Decision)**: **`READY_FOR_CLEAN_OOF`**  

---

## 1. Eksekutif Ringkasan

Tahap **Stage 9.5** dilaksanakan sebagai audit independen, remediasi terarah terhadap defect yang tersisa pasca-Stage 9.4, pembuktian integritas artefak, serta verifikasi empiris kesiapan pipeline secara menyeluruh sebelum pelaksanaan eksperimen bersih berikutnya.

### Kepatuhan terhadap Batasan Wajib (Strict Boundaries)
1. **Zero Full Training**: Tidak ada pelatihan YOLO penuh, tidak ada 4-fold OOF probe training 12-epoch, dan tidak ada ablation training 50-epoch yang dijalankan.
2. **Zero Test Set Snooping**: Tidak ada evaluasi terhadap locked test set (`cassava-v7-standardized/test/` maupun `cassava-v7-research/test/`). File evaluasi akhir [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py) tetap terkunci di balik guard `--confirm-final-evaluation`.
3. **Immutability Dataset**: Dataset historis `cassava-v7-research` dan dataset terstandarisasi `cassava-v7-standardized` tidak diubah, tidak ditimpa, dan tidak dibangun ulang.
4. **Preservasi Artefak Historis**: Seluruh checkpoint, prediksi, manifest, dan laporan historis Stage 7 s.d. Stage 9.4 dipertahankan utuh tanpa penimpaan.
5. **Isolasi Output Bersih**: Seluruh artefak, log eksekusi, backup pre-fix, dan laporan Stage 9.5 ditempatkan secara eksklusif pada direktori terpisah [`stage_9_5_code_remediation/`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/).
6. **Empirical Evidence First**: Seluruh status "PASS", kesetaraan geometri 100%, paritas dataset, dan ketiadaan kebocoran grup dibuktikan melalui eksekusi nyata pada lingkungan Python lokal dengan exit code 0 dan output terverifikasi.

---

## 2. Audit Awal, Baseline Commit, dan Pelestarian Pre-Fix

### A. Identitas Git & Lingkungan Eksekusi
- **Base Commit**: `a74eec540c649959d319a192d77442bff2a0f3f7` (status clean pada branch `main` sebelum pembuatan branch kerja).
- **Working Branch**: `stage-9-5-targeted-remediation` (dibuat khusus untuk Stage 9.5).
- **Python**: `3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]`
- **PyTorch**: `2.9.1+cpu`
- **Ultralytics**: `8.3.235`
- **OS**: Windows 11 Pro 64-bit

### B. Inventarisasi dan Pencadangan Pre-Fix
Sebelum melakukan penyuntingan kode, seluruh file target dicadangkan ke direktori [`stage_9_5_code_remediation/pre_fix_backup/`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pre_fix_backup/) dan dicatat hash SHA-256 baseline-nya ke dalam [`stage_9_5_code_remediation/pre_fix_inventory.json`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pre_fix_inventory.json):

| File Target | Hash SHA-256 Pre-Fix |
|---|---|
| `stage_8_1_group_oof_profiling.py` | `97841ca06e7847998205ab1b14a92cc87a5c344afe94bc3810171021a02c0311` |
| `stage_8_2_build_difficulty_manifest.py` | `5251406923a2eadb9901862a1fbb4ee49c088277671f0da5c6362b1488b46f04` |
| `stage_8_3_adaptive_pipeline.py` | `4210cd8f18ab7f925df5f6bed88d28ae40cca0dc0d658150a79c20987d161f71` |
| `stage_8_4_run_ablation_experiments.py` | `7765236e62b8f6fd5d45ed1cdd33c06a771dc136d0a6c0ad37dfe7881a4773ac` |
| `stage_8_5_model_selection_and_validation.py` | `ba9f1bd8a0d8b4dd861396bbc14a02b102d2bea11f76a07a689dffc415a3b4b4` |
| `stage_8_6_final_test_benchmark.py` | `2d92b3c71ef3327c9e7b02ea92517ba7329c353936f47043e3e41eed6d1fae69` |
| `stage_7_baseline_and_error_analysis.py` | `d72473f3c4c95f190e24ec1725586a147e4eb13038a8e1e3263df9fa20b5a19c` |
| `research_utils/detection_metrics.py` | `8457b7dc4b35b8ff14584234821203da4bddc72fc80fd2145e1ecf14a68742ac` |
| `execute_phase_b_standardization.py` | `855d8cbe79f88258f0fe153e1d70a68fdaab0e00ff5811fd69cdc6be2e4bb46d` |
| `run_stage_8_profiling.py` | `5c9e7967f0293977d29fd9e035ddf3d42424c92fe56ea15b675556e5e827eb7e` |

---

## 3. P0 — Remediasi Penggunaan Ulang Checkpoint pada Stage 8.1

### A. Temuan & Analisis Akar Masalah
Pada Stage 9.4, `expected_metadata` mencatat checksum dasar, tetapi logika reuse pada `--skip-training-if-done` memiliki kelemahan kritis:
1. Pengecekan `meta_match` hanya membandingkan subset konfigurasi, mengabaikan hash skrip, hash YAML data fold, dan hash daftar citra fold;
2. Ketika metadata mismatch terjadi, sistem hanya mencetak warning dan tetap melanjutkan eksekusi pelatihan. Namun blok di bawahnya menggunakan kondisi `if not eval_weight_path.exists():`, sehingga jika file bobot lama `best.pt` masih ada di folder output, checkpoint yang tidak valid tersebut langsung digunakan untuk inferensi diagnostik OOF tanpa pernah dilatih ulang;
3. Jumlah citra fold di-hardcode ke angka 8.980 daripada dihitung dari inventaris fisik;
4. Belum ada verifikasi bahwa setiap citra outer holdout diprediksi tepat satu kali.

### B. Implementasi Perbaikan
1. **Konfigurasi Identitas Run Kanonis (20 Field Eksplisit)**:
   Mengimplementasikan fungsi `build_fold_run_identity()` yang mengikat:
   - SHA-256 skrip `stage_8_1_group_oof_profiling.py`;
   - SHA-256 YAML data fold (`fold_k_data.yaml`);
   - SHA-256 daftar citra inner-train, inner-validation, dan outer holdout;
   - Dataset inventory fingerprint (jumlah citra, label, dan data.yaml SHA256);
   - Mode pemilihan checkpoint (`nested_inner_val` atau `fixed_epoch_last`);
   - Fold ID, epochs, batch size, imgsz, seed, OOF confidence threshold;
   - Konfigurasi model dan hyperparameter (`model_cfg`, `hyp_cfg`);
   - Versi Python, PyTorch, dan Ultralytics;
   - SHA-256 aktual checkpoint yang digunakan untuk inferensi (`best.pt` / `last.pt`);
   - SHA-256 file prediksi OOF (`predictions.csv`).
2. **Fungsi Verifikasi Reuse Berlapis (`verify_fold_reuse_safety()`)**:
   - Memvalidasi seluruh field identitas run secara eksplisit;
   - Memvalidasi keberadaan fisik checkpoint dan file prediksi;
   - Memvalidasi kecocokan SHA-256 checkpoint aktual dan prediksi terhadap metadata tersimpan;
   - Memvalidasi bahwa jumlah baris prediksi outer holdout tepat sama dengan jumlah citra outer holdout.
3. **Runtime Fail-Fast Anti-Reuse Guard**:
   - Jika checkpoint lama ada tetapi metadata tidak cocok atau verifikasi reuse gagal, fungsi `run_oof_probe_training()` langsung melempar `RuntimeError` penolakan, menghentikan eksekusi dan mencegah penggunaan checkpoint lama secara diam-diam.
4. **Verifikasi Jalur Split & Hitungan Dinamis**:
   - Menghitung `total_train_images` secara dinamis dari file daftar citra;
   - Memvalidasi keberadaan citra fisik di disk dan menolak split yang mengarah ke `cassava-v7-research` saat menggunakan dataset terstandarisasi.

### C. Pembuktian Uji Regresi
- `test_identical_metadata_and_valid_checkpoint_allows_reuse`: PASS
- `test_script_hash_change_rejects_reuse`: PASS
- `test_yaml_hash_change_rejects_reuse`: PASS
- `test_inner_train_hash_change_rejects_reuse`: PASS
- `test_inner_val_hash_change_rejects_reuse`: PASS
- `test_outer_holdout_hash_change_rejects_reuse`: PASS
- `test_checkpoint_hash_mismatch_rejects_reuse`: PASS
- `test_missing_metadata_rejects_reuse`: PASS
- `test_missing_checkpoint_rejects_reuse`: PASS
- `test_mismatched_checkpoint_runtime_guard_raises_error`: PASS
- `test_historical_output_dir_never_overwritten`: PASS
- `test_holdout_prediction_count_mismatch_rejects_reuse`: PASS

---

## 4. P1 — Remediasi Fingerprint & `--skip-if-done` pada Stage 8.4

### A. Temuan & Analisis Akar Masalah
Meskipun Stage 9.4 telah menghitung `script_sha256`, `data_yaml_sha256`, dan `manifest_sha256`, pengecekan reuse sebelum skip eksperimen belum membandingkan seluruh fingerprint tersebut terhadap konfigurasi run sebelumnya, sehingga modifikasi skrip atau manifest kesulitan berpotensi melewati eksekusi.

### B. Implementasi Perbaikan
1. Mengimplementasikan struktur identitas kanonis melalui `build_experiment_run_identity()` yang menyatukan: `exp_id`, `exp_name`, `epochs`, `batch`, `imgsz`, `seed`, `device`, `script_sha256`, `data_yaml_sha256`, `manifest_sha256`, `pretrained_weight_sha256`, `train_flags`, dan `software_versions`.
2. Mengimplementasikan `verify_ablation_reuse_safety()` yang memeriksa kesesuaian seluruh sidik jari konfigurasi serta memverifikasi SHA-256 aktual `best.pt`.
3. Mempertahankan proteksi anti-overwrite pada direktori historis `stage_8_ablation/` (melempar `PermissionError`).
4. Menambahkan guard anti-penimpaan pada output bersih jika artefak tujuan tidak cocok dengan konfigurasi run aktif.

### C. Pembuktian Uji Regresi
- `test_ablation_identical_metadata_allows_reuse`: PASS
- `test_ablation_script_hash_change_rejects_reuse`: PASS
- `test_ablation_data_yaml_hash_change_rejects_reuse`: PASS
- `test_ablation_manifest_hash_change_rejects_reuse`: PASS
- `test_ablation_pretrained_weight_change_rejects_reuse`: PASS
- `test_ablation_hyperparameters_change_rejects_reuse`: PASS
- `test_ablation_experiment_type_mismatch_rejects_reuse`: PASS
- `test_ablation_historic_dir_overwrite_raises_permission_error`: PASS

---

## 5. P1 — Unifikasi Evaluator Stage 7 dengan Evaluator Kanonis

### A. Temuan & Analisis Akar Masalah
Stage 7 masih menggunakan implementasi parsing YOLO dan algoritma matching lokal yang terpisah dari `research_utils/detection_metrics.py`. Selain berpotensi menghasilkan taksonomi error yang tidak konsisten antarstage, kode Stage 7 lama memiliki bug runtime `NameError: free variable 'matched_preds' referenced before assignment`.

### B. Implementasi Perbaikan
1. **Refaktor Penuh Menggunakan SSOT**:
   Mengganti seluruh parsing dan matching di [`stage_7_baseline_and_error_analysis.py`](file:///d:/UNDIP/SKRIPSI/stage_7_baseline_and_error_analysis.py) dengan fungsi kanonis dari `research_utils.detection_metrics`: `calculate_iou_xyxy`, `classify_object_size`, `parse_yolo_labels`, dan `greedy_bipartite_matching`.
2. **Perbaikan Bug Matching**:
   Menghapus variabel bebas `matched_preds` dan menyelaraskan struktur data matching dengan tuple kembalian `(tp_matches, fn_unmatched, fp_unmatched, error_taxonomy)`.
3. **Penetapan Versi Evaluator**:
   Menetapkan konstanta `EVALUATOR_VERSION = "Stage-9.5-Canonical-v1"` dan mencatatnya dalam metadata evaluasi.
4. **Pemisahan Tegas Metrik Resmi vs Metrik Diagnostik**:
   Mendokumentasikan bahwa metrik resmi Ultralytics (mAP@50, mAP@50-95, AP per kelas) adalah metrik resmi model, sedangkan global one-to-one matching bersifat diagnostik error decomposition.

### C. Pembuktian Uji Regresi
- `test_competing_one_pred_two_gt`: PASS
- `test_competing_two_preds_one_gt`: PASS
- `test_iou_below_association_threshold`: PASS
- `test_iou_between_association_and_eval_threshold`: PASS
- `test_high_iou_wrong_class`: PASS
- `test_unmatched_fp_and_tp`: PASS
- `test_empty_image_with_false_positive`: PASS
- `test_stage_7_and_stage_8_matching_parity`: PASS

---

## 6. P1 — Eliminasi Asumsi Dimensi Tetap 640

### A. Temuan & Analisis Akar Masalah
Beberapa modul (`research_utils/detection_metrics.py`, `stage_8_2_build_difficulty_manifest.py`, `stage_8_5_model_selection_and_validation.py`, `stage_8_6_final_test_benchmark.py`) memanggil parser label dengan argumen default `img_width=640.0, img_height=640.0` tanpa membaca resolusi fisik citra. Selain itu, batas partisi objek Medium sebelumnya menggunakan `<= 96*96`, sehingga bounding box dengan luas tepat $96^2$ tergolong Medium bukannya Large.

### B. Implementasi Perbaikan
1. **Fungsi Pembacaan Dimensi Dinamis (`get_image_dimensions`)**:
   Menambahkan pembacaan resolusi fisik citra langsung via Pillow (`Image.open().size`) pada `research_utils/detection_metrics.py` dan mengintegrasikannya ke seluruh modul evaluasi.
2. **Koreksi Batas Klasifikasi Strata Ukuran**:
   - Small: $\text{area} < 32^2$ ($1.024$ px$^2$);
   - Medium: $32^2 \le \text{area} < 96^2$ ($1.024 \le \text{area} < 9.216$ px$^2$);
   - Large: $\text{area} \ge 96^2$ ($\ge 9.216$ px$^2$).
3. **Penyimpanan Metadata Dimensi pada Manifest**:
   Manifest kesulitan Stage 8.2 kini menyimpan kolom `norm_xyxy`, `img_width`, dan `img_height` untuk setiap objek.

### C. Pembuktian Uji Regresi
- `test_non_square_dimensions_parsing_and_metrics`: PASS (diuji pada resolusi sintetis 1920x1080)
- `test_object_size_classification_boundaries`: PASS (Small $< 1024$, Medium $1024 \le A < 9216$, Large $\ge 9216$)
- `test_get_image_dimensions_dynamic`: PASS

---

## 7. P1 — Penguatan Standardisasi & Verifikasi Integritas Anotasi

### A. Temuan & Analisis Akar Masalah
`execute_phase_b_standardization.py` sebelumnya menjepit koordinat poligon yang berada di luar rentang valid $[0.0, 1.0]$ secara diam-diam (silent clamping). Selain itu, mode verifikasi belum menguji kesetaraan geometri bounding box secara numerik.

### B. Implementasi Perbaikan
1. **Penolakan Koordinat Invalid**:
   Menghapus silent clamping pada `convert_polygon_to_bbox()`. Koordinat di luar batas $[0.0, 1.0]$ atau bernilai non-finite dilaporkan sebagai invalid.
2. **Verifikasi Geometri Presisi Tinggi (Toleransi $\le 10^{-4}$)**:
   Menambahkan perbandingan geometris terhadap setiap bounding box antara sumber dan tujuan.
3. **Eksekusi Mode `--verify-only` pada Dataset Nyata**:
   Dijalankan langsung pada command line tanpa memodifikasi satu file pun:
   ```bash
   python execute_phase_b_standardization.py --verify-only --src-root cassava-v7-research --dst-root cassava-v7-standardized
   ```
   **Hasil Eksekusi Nyata**:
   - Exit code: `0`
   - Total citra diverifikasi: **12.829 citra** (Train: 8.980, Valid: 2.567, Test: 1.282)
   - Total label diverifikasi: **12.829 file label**
   - Total anotasi bounding box: **48.957 objek**
   - Citra hilang / orphan: **0**
   - Label hilang / orphan: **0**
   - Presisi kesetaraan geometri: **100.0% PASS** (semua koordinat cocok dalam toleransi $\le 10^{-4}$)
   - Zero files written / zero files overwritten.

### C. Pembuktian Uji Regresi
- `test_valid_box_preserved`: PASS
- `test_valid_polygon_converted`: PASS
- `test_non_finite_coordinates_rejected`: PASS
- `test_out_of_bounds_coordinates_rejected`: PASS
- `test_invalid_column_count_rejected`: PASS
- `test_class_id_out_of_bounds_rejected`: PASS
- `test_missing_destination_label_fails`: PASS
- `test_orphan_destination_label_fails`: PASS
- `test_image_size_mismatch_fails`: PASS
- `test_geometry_mismatch_fails_audit`: PASS

---

## 8. P1 — Tes Integrasi Crop Melalui DataLoader Nyata Ultralytics

### A. Temuan & Analisis Akar Masalah
Unit test Stage 9.4 hanya memanggil `apply_context_crop()` secara langsung pada dictionary sintetis. Hal ini belum membuktikan bahwa augmentasi adaptif berfungsi saat dipanggil di dalam siklus data loader nyata Ultralytics, di mana `BaseDataset` melakukan pemuatan citra secara malas (lazy loading).

### B. Implementasi Perbaikan
1. **Dukungan Lazy Image Loading di `apply_context_crop()`**:
   Jika `label.get("img") is None` tetapi `label.get("im_file")` tersedia, citra dimuat langsung dari disk melalui `cv2.imread(label["im_file"])`.
2. **Pembersihan Parameter Trainer di `AdaptiveYOLODataset`**:
   Mengekstrak argumen `enable_mixture_sampling` sebelum meneruskan `kwargs` ke `super().__init__()` untuk mencegah `TypeError`.
3. **Integration Test Nyata pada Mini Dataset Sintetis**:
   Membangun dataset mini sintetis di direktori temporer, menginisialisasi `AdaptiveYOLODataset`, dan menjalankan pipeline collate/batch Ultralytics.

### C. Pembuktian Uji Regresi
- `test_crop_integration_with_real_ultralytics_dataloader`: PASS (membuktikan crop diterapkan, resolusi output sesuai target, koordinat ternormalisasi valid, dan visibilitas lesi sekunder dihormati)
- `test_crop_skipped_when_no_manifest_target`: PASS
- `test_crop_skipped_when_target_box_invalid`: PASS
- `test_crop_with_non_square_input_dimensions`: PASS
- `test_experiment_matrix_flags_isolation`: PASS (EXP-0 tanpa augmentasi khusus; EXP-1 hanya adaptive sampling; EXP-2 hanya context crop; EXP-3 kombinasi keduanya)

---

## 9. P1 — Dry-Run OOF dengan Dataset Terstandarisasi

### A. Temuan & Analisis Akar Masalah
Pada Stage 9.4, beberapa artefak scratch menyertakan path ke `cassava-v7-research` dan default orchestrator belum mengikat `cassava-v7-standardized/data.yaml` secara Single Source of Truth (SSOT). Selain itu, alur `--dry-run` mencoba melanjutkan ke Stage 8.2 meskipun prediksi OOF belum dihasilkan.

### B. Implementasi Perbaikan & Eksekusi Nyata
1. **Pembaruan Default Orchestrator**:
   `run_stage_8_profiling.py` secara resmi mendefinisikan default `--data-yaml` ke `cassava-v7-standardized/data.yaml`.
2. **Penyempurnaan Alur Dry-Run**:
   Jika flag `--dry-run` aktif dan prediksi OOF belum tersedia, skrip menghentikan eksekusi secara bersih setelah memverifikasi pembuatan split OOF.
3. **Eksekusi Dry-Run Nyata**:
   ```bash
   python run_stage_8_profiling.py --dry-run --data-yaml cassava-v7-standardized/data.yaml --output-dir-81 stage_9_5_code_remediation/dry_run_standardized
   ```
   **Hasil Eksekusi Nyata**:
   - Exit code: `0`
   - Total file split dihasilkan: **12 file** (`fold_[0-3]_inner_train.txt`, `fold_[0-3]_inner_val.txt`, `fold_[0-3]_outer_holdout.txt`)
   - Total baris path citra: **35.920 baris**
   - Path mengarah ke `cassava-v7-standardized`: **35.920 baris (100.0%)**
   - Path mengarah ke `cassava-v7-research`: **0 baris (0.00% leakage)**
   - Kebocoran grup antar-fold: **0.00% across 5.546 atomic groups** (terverifikasi 100% ZERO_LEAKAGE pada `fold_group_leakage_check.csv`).
   - Seluruh hasil audit terekam pada [`stage_9_5_code_remediation/dry_run_standardized_audit.json`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/dry_run_standardized_audit.json).

---

## 10. P2 — Validasi Formula Mixture Sampling & Konsistensi Manifest

### A. Formula Matematis Kanonis
Sesuai metodologi penelitian, probabilitas mixture sampling didefinisikan sebagai:
$$P(k) = \frac{1 - \lambda}{N} + \lambda \frac{D_k}{\sum_{j=1}^N D_j}$$
dengan parameter resmi $\lambda = 0.70$.

### B. Implementasi & Verifikasi
1. Parameter `--mixture-lambda` diteruskan secara eksplisit dari orchestrator ke perhitungan bobot sampler;
2. Validasi ketat memverifikasi bahwa $0 \le \lambda \le 1$, bobot finite, strictly positive (floor $(1-\lambda)/N > 0$), dan $\sum P = 1.0 \pm 10^{-5}$;
3. Untuk kasus ekstrem di mana seluruh citra bersih ($\sum D = 0$), implementasi menyediakan fallback deterministik matematis ke distribusi uniform $1/N$ tanpa menghasilkan nilai NaN atau crash pembagian nol.

### C. Pembuktian Uji Regresi
- `test_mixture_formula_exact_values`: PASS
- `test_mixture_formula_sum_d_zero_fallback`: PASS
- `test_mixture_lambda_out_of_range_rejected`: PASS
- `test_mixture_probabilities_strictly_positive_and_sum_to_one`: PASS

---

## 11. P2 — Audit Pelestarian Artefak & Keselamatan Eksekusi

1. **Anti-Overwrite Guards**:
   - Seluruh skrip pelatihan dan evaluasi menolak menimpa direktori historis `stage_8_ablation/` (dilindungi `PermissionError`).
   - Eksekusi baru diarahkan ke `stage_10_11_clean_experiments/` atau `stage_9_5_code_remediation/`.
2. **Locked Test Set Protection**:
   - [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py) tetap terkunci di balik guard `--confirm-final-evaluation` dan tidak dijalankan selama Stage 9.5.

---

## 12. Hasil Eksekusi Pengujian Regresi Komprehensif

Seluruh pengujian otomatis dijalankan secara nyata pada environment Python lokal:
```bash
python -m unittest discover -p "test_stage_9_*.py" -v
```

### Ringkasan Eksekusi Pengujian:

| File Test Suite | Fokus Pengujian | Jumlah Test | Hasil | Durasi |
|---|---|---:|:---:|---:|
| [`test_stage_9_3_regression.py`](file:///d:/UNDIP/SKRIPSI/test_stage_9_3_regression.py) | Regression suite historis Stage 9.3 | 16 | 16 PASS | 0.080s |
| [`test_stage_9_4_regression.py`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py) | Regression suite terintegrasi Stage 9.4 | 26 | 26 PASS | 0.140s |
| [`test_stage_9_5_regression.py`](file:///d:/UNDIP/SKRIPSI/test_stage_9_5_regression.py) | Targeted remediation & verification Stage 9.5 | 48 | 48 PASS | 1.295s |
| **TOTAL KESELURUHAN** | **All Combined Regression Suites** | **90** | **90 PASS (0 Failed, 0 Errors)** | **1.515s** |

Log lengkap eksekusi pengujian tersimpan secara permanen pada:  
[`stage_9_5_code_remediation/regression_test_results.txt`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/regression_test_results.txt).

---

## 13. Inventarisasi Artefak dan Deliverables Stage 9.5

Seluruh output Stage 9.5 tersimpan terisolasi di [`stage_9_5_code_remediation/`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/):

| File Artefak | Peran & Deskripsi |
|---|---|
| [`STAGE_9_5_REMEDIATION_REPORT.md`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/STAGE_9_5_REMEDIATION_REPORT.md) | Laporan komprehensif master audit dan remediasi Stage 9.5. |
| [`code_findings.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/code_findings.csv) | Matriks 11 temuan audit dari P0 s.d. P2 beserta status remediasinya. |
| [`code_changes_manifest.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/code_changes_manifest.csv) | Manifest 10 file yang dimodifikasi beserta hash pre-fix dan post-fix. |
| [`regression_test_matrix.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/regression_test_matrix.csv) | Pemetaan lengkap 90 test cases ke kelas, metode, dan status kelulusan. |
| [`regression_test_results.txt`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/regression_test_results.txt) | Log eksekusi mentah dari runner unittest (90 tests OK). |
| [`method_to_code_traceability.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/method_to_code_traceability.csv) | Matriks keterlacakan metodologi penelitian (METH-01 s.d. METH-12) ke implementasi kode. |
| [`pipeline_config_audit.json`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pipeline_config_audit.json) | Audit konfigurasi pipeline machine-readable lengkap dengan seluruh hash SHA-256. |
| [`clean_run_readiness_checklist.md`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/clean_run_readiness_checklist.md) | Checklist kesiapan eksekusi clean OOF dan eksperimen bersih. |
| [`code_diff_summary.md`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/code_diff_summary.md) | Ringkasan diff teknis dan keputusan desain untuk seluruh file yang dimodifikasi. |
| [`pre_fix_backup/`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pre_fix_backup/) | Salinan file pre-fix sebelum modifikasi Stage 9.5 dilakukan. |
| [`pre_fix_inventory.json`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pre_fix_inventory.json) | Inventaris hash SHA-256 pre-fix. |
| [`dry_run_standardized/`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/dry_run_standardized/) | Direktori output dry-run OOF (12 file split terstandarisasi). |
| [`dry_run_standardized_audit.json`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/dry_run_standardized_audit.json) | Laporan audit dry-run membuktikan 100% path terstandarisasi & 0% kebocoran grup. |
| [`test_stage_9_5_regression.py`](file:///d:/UNDIP/SKRIPSI/test_stage_9_5_regression.py) | Skrip pengujian regresi Stage 9.5 (48 unit & integration tests). |

---

## 14. Keputusan Gerbang Akhir (Final Gate Decision)

Berdasarkan seluruh hasil investigasi kode, perbaikan terarah, pengujian regresi otomatis, verifikasi dataset tanpa modifikasi, dan eksekusi dry-run empiris:

1. **P0 Selesai**: Checkpoint reuse pada Stage 8.1 terikat 20 field konfigurasi identitas run kanonis; mismatch sekecil apa pun memicu penolakan runtime tanpa silent reuse.
2. **P1 Selesai**: Pengecekan skip Stage 8.4 memvalidasi seluruh fingerprint konfigurasi; evaluator Stage 7 telah sepenuhnya disatukan ke SSOT `research_utils/detection_metrics.py`; asumsi dimensi tetap 640 telah dihilangkan; standardisasi dataset lolos verifikasi paritas dan geometri 100%; integrasi crop adaptif terbukti pada dataloader nyata; dry-run OOF membuktikan 100% path terstandarisasi tanpa kebocoran grup.
3. **P2 Selesai**: Formula mixture sampling ($\lambda = 0.70$) dan manifest divalidasi ketat dengan penanganan fallback deterministik matematis; proteksi penimpaan direktori historis aktif; test set tetap terkunci.
4. **90 Regression Tests PASS**: Seluruh 90 pengujian lulus 100% tanpa kegagalan maupun error.

Status Keputusan Resmi:

### **`READY_FOR_CLEAN_OOF`**

---

## 15. Rekomendasi Langkah Selanjutnya

1. **Jalankan Clean 4-Fold OOF Probe Training (Stage 8.1)**:
   - Gunakan dataset terstandarisasi `cassava-v7-standardized/data.yaml`.
   - Gunakan output direktori bersih, misalnya `stage_10_clean_oof_probe/`.
   - Mode seleksi checkpoint: `nested_inner_val`.
2. **Bangun Manifest Kesulitan Pasca-OOF (Stage 8.2)**:
   - Eksekusi pembuatan manifest kesulitan dari cache prediksi OOF bersih.
3. **Jalankan Clean Ablation Experiments EXP-0 s.d. EXP-3 (Stage 8.4)**:
   - Output diarahkan ke `stage_10_11_clean_experiments/`.
   - Validasi metrik kanonis via SSOT `extract_canonical_metrics()`.
4. **Seleksi Model Berdasarkan Validation Set (Stage 8.5)**:
   - Evaluasi murni pada validation set terstandarisasi (2.567 citra).
5. **Evaluasi Akhir pada Locked Test Set (Stage 8.6)**:
   - Hanya dijalankan satu kali pada model terbaik terpilih setelah konfirmasi eksplisit dari pengguna.

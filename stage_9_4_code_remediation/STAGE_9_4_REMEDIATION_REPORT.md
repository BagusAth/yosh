# STAGE 9.4 — COMPREHENSIVE AUDIT REMEDIATION, PIPELINE HARDENING, AND REGRESSION VALIDATION REPORT

**Penelitian**: Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong Berbasis YOLO11n dengan Error-Driven Adaptive Optimization  
**Repository**: [yosh](https://github.com/BagusAth/yosh)  
**Workspace**: `D:\UNDIP\SKRIPSI`  
**Auditor**: Senior ML Research Engineer, Computer Vision Engineer, and Independent Code Auditor  
**Tanggal Audit**: 17 Oktober 2026  
**Status Akhir**: `READY_FOR_CLEAN_OOF`  

---

## 1. Eksekutif Ringkasan

Tahap **Stage 9.4** bertujuan untuk menyelesaikan seluruh temuan audit kritis dan celah validasi yang teridentifikasi pasca-Stage 9.3 pada kode aktual repository. Sesuai batasan eksekusi yang ketat:
1. **Tidak ada pelatihan penuh (zero full training)** yang dijalankan (tidak ada 4-fold 12-epoch training, tidak ada 50-epoch ablation).
2. **Tidak ada evaluasi pada locked test set** yang dijalankan selama Stage 9.4.
3. **Seluruh checkpoint, laporan historis Stage 7–9.3, dan data asli tetap utuh** tanpa penimpaan.
4. Seluruh perbaikan dibuktikan melalui unit test, parser checks, static analysis, dan regression suite komprehensif 26-test yang **100% lulus (PASS)**.

Berdasarkan hasil pembuktian empiris, repository ini secara resmi memperoleh status **`READY_FOR_CLEAN_OOF`**.

---

## 2. Temuan Audit Kritis & Solusi Remediasi

### A. Temuan Kritis 1: Kebocoran Holdout pada Pemilihan Checkpoint OOF Profiling
- **File Terdampak**: [`stage_8_1_group_oof_profiling.py`](file:///d:/UNDIP/SKRIPSI/stage_8_1_group_oof_profiling.py)
- **Akar Masalah**: Pada implementasi awal, satu fold $k$ dari 4 fold digunakan sebagai split `val:` saat melatih probe model YOLO11n. Model checkpoint `best.pt` kemudian dipilih berdasarkan metrik validasi pada fold $k$ tersebut. Setelah itu, `best.pt` yang sama digunakan untuk memprediksi citra fold $k$ sebagai estimasi Out-of-Fold. Prosedur ini menyebabkan label fold holdout ikut memengaruhi pemilihan checkpoint model (holdout leakage / data snooping).
- **Remediasi yang Diimplementasikan**:
  - Mengubah prosedur partisi menjadi **Nested Group-Aware Holdout Isolation** (`nested_inner_val`).
  - Outer Holdout fold $k$ (1/4 data latih = 2.245 citra) **100% diisolasi** dan sama sekali tidak dimasukkan ke dalam `fold_k_data.yaml` baik pada `train:` maupun `val:`.
  - Tiga fold sisanya dibagi secara group-aware menjadi Inner-Train (2 fold = 4.490 citra) dan Inner-Validation (1 fold = 2.245 citra).
  - Ultralytics mengevaluasi dan memilih `best.pt` murni berdasarkan Inner-Validation tanpa pernah melihat Outer Holdout $k$.
  - Checkpoint `best.pt` tersebut kemudian dievaluasi secara eksklusif pada Outer Holdout $k$ (@conf=0.001).
  - Seluruh pemetaan grup disimpan pada [`nested_fold_assignments.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/scratch/test_8_1/nested_fold_assignments.csv).
- **Hasil Pengujian**: Terverifikasi melalui [`TestNestedOOFGroupIsolation`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py) bahwa irisan grup antara Outer Holdout, Inner-Val, dan Inner-Train adalah **0 grup (0.00% leakage)**.

### B. Temuan Kritis 2: Ketidakkonsistenan Sumber Dataset & Path Evaluator (SSOT)
- **File Terdampak**: [`stage_8_5_model_selection_and_validation.py`](file:///d:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py), [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py), [`stage_8_1_group_oof_profiling.py`](file:///d:/UNDIP/SKRIPSI/stage_8_1_group_oof_profiling.py), [`stage_8_2_build_difficulty_manifest.py`](file:///d:/UNDIP/SKRIPSI/stage_8_2_build_difficulty_manifest.py)
- **Akar Masalah**: Terdapat path hard-coded literal seperti `d:/UNDIP/SKRIPSI/cassava-v7-research/valid/images` dan `test/images` pada skrip evaluasi, sehingga perubahan argumen `--data-yaml` ke dataset terstandarisasi diabaikan pada evaluasi dekomposisi error dan penghitungan recall per ukuran.
- **Remediasi yang Diimplementasikan**:
  - Menjadikan `data.yaml` sebagai **Single Source of Truth (SSOT)**.
  - Mengimplementasikan fungsi resolver dinamis: `resolve_val_paths()`, `resolve_test_paths()`, dan `resolve_train_paths()`. Path citra dan label diturunkan secara langsung dari entri YAML.
  - Memastikan urutan kelas konsisten: `["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]`.
- **Hasil Pengujian**: Terverifikasi melalui [`TestEvaluatorDataSourceSSOT`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py) bahwa seluruh resolver path menurunkan direktori fisik yang valid sesuai konfigurasi YAML tanpa mengakses path hard-coded.

### C. Temuan Prioritas Tinggi 1: Penyeragaman Evaluator & Algoritma Matching
- **File Terdampak**: [`research_utils/detection_metrics.py`](file:///d:/UNDIP/SKRIPSI/research_utils/detection_metrics.py) (Modul Baru), [`stage_8_2_build_difficulty_manifest.py`](file:///d:/UNDIP/SKRIPSI/stage_8_2_build_difficulty_manifest.py), [`stage_8_5_model_selection_and_validation.py`](file:///d:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py), [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py)
- **Akar Masalah**: Stage 8.2 mengurutkan kandidat pasangan GT-prediksi secara global menurun berdasarkan IoU, sedangkan Stage 8.5 dan 8.6 mencocokkan prediksi secara sekuensial per objek GT.
- **Remediasi yang Diimplementasikan**:
  - Membangun fungsi kanonis bersama `greedy_bipartite_matching()` pada [`research_utils/detection_metrics.py`](file:///d:/UNDIP/SKRIPSI/research_utils/detection_metrics.py).
  - Algoritma menghitung seluruh pasangan kandidat dengan $\text{IoU} \ge 0.10$, mengurutkan kandidat secara global descending berdasarkan IoU, dan melakukan penugasan serakah 1-to-1 (zero prediction reuse, zero GT reuse).
  - Menambahkan penghitungan metrik strata ukuran objek (Small $< 32^2$, Medium $32^2 - 96^2$, Large $\ge 96^2$) pada Stage 8.5 dan Stage 8.6 secara konsisten.
- **Hasil Pengujian**: Terverifikasi melalui [`TestGlobalOneToOneMatching`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py) pada kasus sintetis 1 prediksi untuk 2 GT dan multiple prediksi untuk 1 GT.

### D. Temuan Prioritas Tinggi 2: Hierarki Seleksi Model Terkunci
- **File Terdampak**: [`stage_8_5_model_selection_and_validation.py`](file:///d:/UNDIP/SKRIPSI/stage_8_5_model_selection_and_validation.py), [`research_utils/detection_metrics.py`](file:///d:/UNDIP/SKRIPSI/research_utils/detection_metrics.py)
- **Akar Masalah**: Urutan tie-break pada kode lama mengevaluasi `mAP@50` sebelum `recall_small`, dan terdapat celah di mana model dengan mAP@50-95 lebih rendah dapat menang jika memiliki selisih dalam margin $0.5\%$.
- **Remediasi yang Diimplementasikan**:
  - Mengunci hierarki keputusan seleksi model resmi melalui `select_best_model()`:
    1. **Kriteria Utama**: Validasi mAP@50-95 tertinggi.
    2. **Tie-Break 1**: Recall Small ($< 32^2$) tertinggi (hanya jika $|\Delta \text{mAP@50-95}| \le 10^{-5}$).
    3. **Tie-Break 2**: Validasi mAP@50 tertinggi.
    4. **Tie-Break 3**: Total False Negative (`fn_missed`) terendah.
  - Menghapus seluruh margin arbitrer $0.5\%$. Menggunakan floating-point presisi penuh.
  - Memastikan seleksi model murni 100% bergantung pada Validation Set (tanpa data snooping ke test set).
- **Hasil Pengujian**: Terverifikasi melalui [`TestModelSelectionHierarchy`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py) bahwa kandidat dengan mAP@50-95 lebih tinggi selalu menang, dan tie-break berjalan tepat sesuai hierarki.

### E. Temuan Prioritas Tinggi 3: Eliminasi Silent Uniform Fallback pada Manifest & Sampler
- **File Terdampak**: [`stage_8_3_adaptive_pipeline.py`](file:///d:/UNDIP/SKRIPSI/stage_8_3_adaptive_pipeline.py)
- **Akar Masalah**: Pada `get_sampling_weights()`, citra yang tidak ditemukan di manifest kesulitan secara otomatis diberikan bobot uniform $1/N$ secara diam-diam tanpa memperingatkan pengguna.
- **Remediasi yang Diimplementasikan**:
  - Menambahkan metode `validate_manifest_coverage()` yang memicu `ValueError` langsung jika terdapat citra yang hilang dalam manifest kesulitan.
  - Memvalidasi formula mixture sampling: $P(k) = (1 - \lambda)/N + \lambda (D_k / \sum D)$ dengan $\lambda = 0.70$ eksplisit, menjamin bobot finite, strictly positive (floor $(1-\lambda)/N > 0$), dan $\sum P = 1.0 \pm 10^{-5}$.
  - Memastikan patch crop menggunakan ukuran `imgsz` secara dinamis dari parameter `hyp` dan label dictionary dilindungi menggunakan `deepcopy`.
- **Hasil Pengujian**: Terverifikasi melalui [`TestManifestFailFastAndCoverage`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py) dan [`TestSamplingWeightsAndMixtureFormula`](file:///d:/UNDIP/SKRIPSI/test_stage_9_4_regression.py).

### F. Temuan Sedang: Proteksi Artefak Historis & Metadata Checksum
- **File Terdampak**: [`stage_8_4_run_ablation_experiments.py`](file:///d:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py), [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py)
- **Akar Masalah**: Script Stage 8.4 secara default menulis ke `stage_8_ablation/` yang berisiko menimpa data historis, dan flag `--skip-if-done` dapat menggunakan kembali bobot lama tanpa verifikasi kecocokan konfigurasi.
- **Remediasi yang Diimplementasikan**:
  - Mengalihkan default `--output-dir` Stage 8.4 ke direktori baru `stage_10_11_clean_experiments/`.
  - Menambahkan proteksi `PermissionError` yang memblokir penulisan ke `stage_8_ablation/`.
  - Menyimpan metadata sidik jari checksum (`script_sha256`, `data_yaml_sha256`, `epochs`, `batch`, `imgsz`, `seed`) pada `fold_metadata.json` dan `metrics_summary.json` agar checkpoint hanya dapat di-skip jika konfigurasi benar-benar identik.
  - Menambahkan guard `--confirm-final-evaluation` pada Stage 8.6 untuk mencegah evaluasi test set yang tidak disengaja.

---

## 3. Hasil Validasi Regression Test Aktual

Pengujian otomatis dijalankan secara langsung pada environment Python lokal (`Python 3.13.7`, `PyTorch 2.6.0+cu124`, `CUDA:0`).

### A. Eksekusi `test_stage_9_4_regression.py` (Master Regression Suite)
- **Total Test Cases**: 26 unit & integration tests
- **Passed**: 26
- **Failed**: 0
- **Errors**: 0
- **Durasi Eksekusi**: 0.140 detik
- **Status Outcome**: **`PASS (100% SUCCEEDED)`**

Ringkasan Matriks Pengujian:

| Test Suite | Kasus Uji | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| `TestDifficultyFormulaAndBounds` | `test_image_difficulty_aggregation` | $D_\text{image} = 0.5 \max + 0.5 \text{mean}$ | $0.70$ objek, $0.65$ FP, $0.0$ bersih | **PASS** |
| `TestDifficultyFormulaAndBounds` | `test_matched_correct_class_formula` | $D_i$ sesuai formula bobot | $0.4000 \in [0, 1]$ | **PASS** |
| `TestDifficultyFormulaAndBounds` | `test_matched_wrong_class_penalty_and_clipping` | $+0.35$ penalty, clipping $1.0$ | Tepat ter-clip pada $1.0000$ | **PASS** |
| `TestDifficultyFormulaAndBounds` | `test_unmatched_fn_formula` | $D_i = 0.80 + 0.20 S_i$ | Small $1.0$, Med $0.9$, Large $0.8$ | **PASS** |
| `TestGlobalOneToOneMatching` | `test_global_descending_iou_sorting` | Pengurutan kandidat IoU global | Sorting global terverifikasi | **PASS** |
| `TestGlobalOneToOneMatching` | `test_multiple_predictions_for_single_gt` | 1 GT maksimal 1 pred (zero GT reuse) | Top pred cocok; pred 2 jadi FP | **PASS** |
| `TestGlobalOneToOneMatching` | `test_single_prediction_between_two_gt` | 1 pred maksimal 1 GT (zero pred reuse) | Pred terdekat cocok; GT 2 jadi FN | **PASS** |
| `TestManifestFailFastAndCoverage` | `test_dataset_coverage_fail_fast_on_missing_images` | Fail-fast bila citra hilang di manifest | `ValueError` dengan pesan jelas | **PASS** |
| `TestManifestFailFastAndCoverage` | `test_duplicate_image_id_rejection` | Reject duplikasi ID pada manifest | `ValueError` duplikasi terpicu | **PASS** |
| `TestManifestFailFastAndCoverage` | `test_invalid_d_image_bounds_rejection` | Reject $D_\text{image} \notin [0, 1]$ | `ValueError` bounds terpicu | **PASS** |
| `TestSamplingWeightsAndMixtureFormula` | `test_mixture_weights_formula_and_normalization` | $\sum P = 1.0 \pm 10^{-5}$, floor $> 0$ | $\sum P = 1.000000$, min $\ge 0.075$ | **PASS** |
| `TestTargetedContextCropGeometry` | `test_crop_expansion_and_centering` | Crop $2\times$ bbox terpusat pada target | Box tetap terpusat di $(0.5, 0.5)$ | **PASS** |
| `TestTargetedContextCropGeometry` | `test_original_label_not_mutated` | Label asli tidak termutasi in-place | Koordinat instances identik | **PASS** |
| `TestExperimentComponentToggles` | `test_experiment_spec_toggles` | EXP-0 s.d. EXP-3 toggle akurat | 4 varian konfigurasi valid | **PASS** |
| `TestEvaluatorDataSourceSSOT` | `test_resolve_paths_from_yaml` | Dynamic resolution dari `data.yaml` | Resolusi val & test path valid | **PASS** |
| `TestModelSelectionHierarchy` | `test_primary_criterion_map50_95_strictly_wins` | mAP@50-95 lebih tinggi menang mutlak | Pemenang mAP tertinggi terpilih | **PASS** |
| `TestModelSelectionHierarchy` | `test_tie_break_1_recall_small` | Tie mAP dipecahkan oleh Recall Small | Recall Small tertinggi menang | **PASS** |
| `TestModelSelectionHierarchy` | `test_tie_break_2_map50` | Tie lanjut dipecahkan oleh mAP@50 | mAP@50 tertinggi menang | **PASS** |
| `TestModelSelectionHierarchy` | `test_tie_break_3_lowest_fn` | Tie lanjut dipecahkan oleh min FN | FN terendah menang | **PASS** |
| `TestFullPrecisionMetricSerialization` | `test_canonical_metric_extraction` | Pemisahan kanonis `ap50` vs `maps` | Metrik kanonis terekstraksi presisi | **PASS** |
| `TestIsolatedPathsAndAntiOverwrite` | `test_historic_stage_8_ablation_overwrite_blocked` | Blokir tulis ke `stage_8_ablation` | `PermissionError` terpicu aman | **PASS** |
| `TestStandardizedFiveColumnLabels` | `test_reject_invalid_class_id` | Tolak class ID di luar rentang $[0, 4]$ | `ValueError` terpicu | **PASS** |
| `TestStandardizedFiveColumnLabels` | `test_reject_polygon_in_standardized_label` | Tolak poligon pada dataset deteksi | `ValueError` kolom $> 5$ terpicu | **PASS** |
| `TestStandardizedFiveColumnLabels` | `test_valid_five_column_detection` | Validasi tepat 5 kolom kanonis | 5 kolom berhasil divalidasi | **PASS** |
| `TestNonSquareDynamicDimensionParser` | `test_dynamic_coordinate_scaling_1920x1080` | Penskalaan dimensi non-persegi | Area piksel kuadrat akurat | **PASS** |
| `TestNestedOOFGroupIsolation` | `test_nested_fold_zero_leakage_and_isolation` | Zero holdout leakage pada nested fold | 0% kebocoran holdout ke YAML | **PASS** |

### B. Backward Compatibility Test (`test_stage_9_3_regression.py`)
- **Total Test Cases**: 16 unit tests
- **Passed**: 16
- **Status**: **`OK (100% PASS)`** (Durasi: 2.29 detik)

---

## 4. Daftar Artefak yang Dihasilkan pada Stage 9.4

Seluruh artefak hasil audit disimpan di direktori [`stage_9_4_code_remediation/`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation):

1. [`STAGE_9_4_REMEDIATION_REPORT.md`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/STAGE_9_4_REMEDIATION_REPORT.md) — Laporan komprehensif remediasi audit (dokumen ini).
2. [`code_findings.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/code_findings.csv) — Tabel inventarisasi temuan audit lengkap dengan tingkat keparahan, akar masalah, solusi, dan status perbaikan.
3. [`code_changes_manifest.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/code_changes_manifest.csv) — Daftar lengkap file yang dimodifikasi beserta SHA-256 sebelum dan sesudah perbaikan.
4. [`regression_test_matrix.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/regression_test_matrix.csv) — Matriks 26 kasus uji regresi beserta expected result, actual result, dan status PASS.
5. [`regression_test_results.txt`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/regression_test_results.txt) — Log eksekusi pengujian otomatis sesungguhnya.
6. [`method_to_code_traceability.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/method_to_code_traceability.csv) — Pemetaan formal rumus matematika penelitian ke fungsi kode dan unit test pembuktikannya.
7. [`pipeline_config_audit.json`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/pipeline_config_audit.json) — Konfigurasi efektif, metadata environment, dan status konsistensi path.
8. [`clean_run_readiness_checklist.md`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/clean_run_readiness_checklist.md) — Daftar syarat kesiapan sebelum menjalankan clean OOF profiling dan clean ablation training.
9. [`code_diff_summary.md`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/code_diff_summary.md) — Ringkasan perubahan teknis per file.
10. [`pre_fix_backup/`](file:///d:/UNDIP/SKRIPSI/stage_9_4_code_remediation/pre_fix_backup) — Salinan persis file sebelum modifikasi Stage 9.4 dimulai.

---

## 5. Konfirmasi Kepatuhan Aturan Kerja

1. **Konfirmasi Zero Full Training**:
   - Selama Stage 9.4, tidak ada proses training YOLO11n 50 epoch per model ataupun 4-fold 12-epoch OOF training yang dijalankan.
   - Hanya unit test berbobot ringan, verifikasi partisi data via `--dry-run`, dan smoke test dataloader satu batch yang dieksekusi.
2. **Konfirmasi Zero Test Set Evaluation**:
   - Skrip [`stage_8_6_final_test_benchmark.py`](file:///d:/UNDIP/SKRIPSI/stage_8_6_final_test_benchmark.py) tidak pernah dijalankan pada locked test set.
   - Ditambahkan guard `--confirm-final-evaluation` untuk mengunci test set dari eksekusi tanpa konfirmasi.
3. **Konfirmasi Pelestarian Artefak Historis**:
   - Checkpoint lama, `results.csv`, `metrics_summary.json`, manifest lama, dan laporan Stage 7–9.3 dipertahankan tanpa perubahan angka atau penimpaan.

---

## 6. Perintah Eksekusi Tahap Berikutnya

Setelah Stage 9.4 selesai dan disetujui, tahapan penelitian dapat dilanjutkan menggunakan perintah berikut:

### 1. Menjalankan Smoke Test Dataloader Adaptif (Non-Training)
```bash
python stage_8_3_adaptive_pipeline.py
```

### 2. Menjalankan Dry-Run Profiling OOF Nested (Memverifikasi Partisi Tanpa Training)
```bash
python run_stage_8_profiling.py --data-yaml cassava-v7-standardized/data.yaml --output-dir-81 stage_10_1_clean_oof --output-dir-82 stage_10_2_clean_difficulty --dry-run
```

### 3. Menjalankan Clean 4-Fold OOF Profiling (Ketika Lingkungan & GPU Siap)
```bash
python run_stage_8_profiling.py --data-yaml cassava-v7-standardized/data.yaml --output-dir-81 stage_10_1_clean_oof --output-dir-82 stage_10_2_clean_difficulty --checkpoint-selection-mode nested_inner_val --epochs 12 --batch 16
```

### 4. Menjalankan Clean Ablation Experiments (EXP-0 s.d. EXP-3)
```bash
python stage_8_4_run_ablation_experiments.py --data-yaml cassava-v7-standardized/data.yaml --manifest-path stage_10_2_clean_difficulty/train_image_difficulty_manifest.csv --output-dir stage_10_11_clean_experiments --epochs 50 --batch 16
```

---

## 7. Keputusan Akhir Gerbang Kelulusan

Berdasarkan inventarisasi temuan dan hasil eksekusi pengujian aktual:
- Seluruh 10 temuan audit kode telah **DIREMEDIASI**.
- Nested group-aware isolation menjamin **0.00% kebocoran holdout** pada seleksi model probe.
- Single source of truth (SSOT) telah mengeliminasi seluruh path literal hard-coded.
- Modul bersama `research_utils/detection_metrics.py` telah menyeragamkan IoU, greedy matching, formula kesulitan, dan aturan seleksi model.
- 26/26 unit dan integration tests pada `test_stage_9_4_regression.py` memperoleh status **PASS**.

**KEPUTUSAN EKSPLISIT: `READY_FOR_CLEAN_OOF`**

*Repository telah memenuhi standar integritas metodologi penelitian dan layak untuk melanjutkan ke proses 4-fold Out-of-Fold profiling bersih pada dataset terstandarisasi.*

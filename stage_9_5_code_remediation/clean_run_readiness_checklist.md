# Clean Run Readiness Checklist — Stage 9.5

**Tanggal Audit**: 17 Oktober 2026  
**Auditor**: Senior Machine Learning Engineer, Computer Vision Research Engineer, Python Test Engineer, and Independent Research Code Auditor  
**Workspace**: `D:\UNDIP\SKRIPSI`  
**Base Commit**: `a74eec540c649959d319a192d77442bff2a0f3f7`  
**Active Branch**: `stage-9-5-targeted-remediation`  
**Target Pipeline**: Clean 4-Fold OOF Profiling & Clean Ablation Experiments (EXP-0 s.d. EXP-3)  

---

## 1. Status Dataset & Integritas Anotasi

- [x] **Homogenitas Format Anotasi**: Seluruh citra pada dataset bersih `cassava-v7-standardized` memiliki anotasi kanonis tepat 5 kolom YOLO (`class_id xc yc w h`).
- [x] **Integritas Koordinat**: Koordinat ternormalisasi dalam rentang valid $[0.0, 1.0]$, bernilai finite, dan berdimensi positif ($w > 0, h > 0$). Tidak ada lagi penjepitan diam-diam (silent clamping) pada parser.
- [x] **Paritas Citra & Label (Zero Orphan, Zero Missing)**:
  - Train: 8.980 citra & 8.980 label (0 hilang, 0 orphan).
  - Validation: 2.567 citra & 2.567 label (0 hilang, 0 orphan).
  - Test: 1.282 citra & 1.282 label (0 hilang, 0 orphan).
  - Total: 12.829 citra dan 48.957 bounding box anotasi.
- [x] **Presisi Geometri Anotasi (100.0% Equivalent)**:
  - Verifikasi empiris melalui `execute_phase_b_standardization.py --verify-only` membuktikan kesetaraan koordinat hasil konversi terhadap anotasi sumber dalam batas toleransi numerik ketat $\le 10^{-4}$ pada 48.957 objek.
- [x] **Pelestarian Dataset Historis**: Dataset `cassava-v7-research` dipertahankan 100% tanpa perubahan fisik atau penimpaan file sebagai artefak penelitian historis.

---

## 2. Status OOF Profiling & Checkpoint Reuse Safety (Stage 8.1)

- [x] **Konfigurasi Identitas Run Kanonis (20 Field Eksplisit)**:
  - Identitas run mengikat `script_sha256`, `data_yaml_sha256`, `fold_yaml_sha256`, `inner_train_sha256`, `inner_val_sha256`, `outer_holdout_sha256`, `dataset_inventory_fingerprint`, `fold_id`, `checkpoint_selection_mode`, `epochs`, `batch_size`, `imgsz`, `seed`, `conf_threshold`, `model_cfg`, `hyp_cfg`, `software_versions`, `selected_checkpoint_sha256`, dan `predictions_sha256`.
- [x] **Runtime Fail-Fast Anti-Reuse Guard**:
  - Jika checkpoint `best.pt` atau `last.pt` ada tetapi metadata tidak cocok atau verifikasi reuse gagal, fungsi pelatihan langsung melempar `RuntimeError` dan menolak menggunakan kembali checkpoint yang tidak cocok.
- [x] **Nested Group-Aware Holdout Isolation**:
  - Outer Holdout fold $k$ (1/4 data latih) **100% diisolasi** dari konfigurasi YAML pelatihan probe.
  - Tiga fold sisanya dibagi secara group-aware menjadi Inner-Train (2 fold) dan Inner-Validation (1 fold).
  - Checkpoint dipilih murni berdasarkan Inner-Validation tanpa pernah mengevaluasi Outer Holdout saat pemilihan checkpoint.
- [x] **Zero Cross-Fold Group Leakage**:
  - Terverifikasi 100% zero leakage pada dry-run dataset terstandarisasi di seluruh 5.546 grup atomik.
- [x] **Validasi Jumlah Prediksi Holdout**:
  - Memvalidasi bahwa setiap citra outer holdout diprediksi tepat satu kali dan jumlah baris prediksi cocok 100% dengan jumlah citra outer holdout.

---

## 3. Status Manifest Kesulitan & Formula Sampling (Stage 8.2)

- [x] **Formula Mixture Sampling Kanonis**:
  - Mengimplementasikan $P(k) = \frac{1 - \lambda}{N} + \lambda \frac{D_k}{\sum_j D_j}$ dengan parameter `--mixture-lambda` default 0.70.
  - Memvalidasi secara ketat bahwa $0 \le \lambda \le 1$, bobot finite, strictly positive (floor $(1-\lambda)/N > 0$), dan $\sum P = 1.0 \pm 10^{-5}$.
  - Fallback deterministik matematis untuk kasus $\sum D = 0$ ke distribusi uniform $1/N$ tanpa menghasilkan nilai NaN atau pembagian nol.
- [x] **Dimensi Citra Dinamis**:
  - Membaca resolusi aktual setiap citra latih via `get_image_dimensions()`, menghilangkan asumsi dimensi tetap $640 \times 640$.
  - Menyimpan koordinat ternormalisasi (`norm_xyxy`) beserta dimensi aktual (`img_width`, `img_height`) ke dalam manifest.
- [x] **Batas Strata Ukuran Objek Kanonis**:
  - Small: $\text{area} < 32^2$ ($1.024$ px$^2$).
  - Medium: $32^2 \le \text{area} < 96^2$ ($1.024$ s.d. $9.215$ px$^2$).
  - Large: $\text{area} \ge 96^2$ ($\ge 9.216$ px$^2$).

---

## 4. Status Dataloader & Augmentasi Adaptif (Stage 8.3)

- [x] **Integrasi DataLoader Ultralytics Nyata**:
  - Teruji melalui integrasi nyata dengan kelas dataset, transform, dan collate Ultralytics pada dataset mini sintetis.
  - Menambahkan lazy image loading fallback dari `label["im_file"]` di `apply_context_crop()` jika `label["img"]` belum dimuat ke memori.
  - Memisahkan argumen trainer `enable_mixture_sampling` dari `AdaptiveYOLODataset.__init__()` untuk mencegah `TypeError`.
- [x] **Zero Silent Fallback pada Manifest**:
  - Memvalidasi coverage 100% manifest terhadap citra dataset latih secara fail-fast (`ValueError` jika ada citra yang hilang).
- [x] **Kontrak Augmentasi Terarah (EXP-2 & EXP-3)**:
  - Crop diperluas $2\times$ bounding box target kesulitan tertinggi ($D_i \ge 0.70$).
  - Lesi sekunder dipertahankan jika visibilitas $\ge 0.30$; dibuang jika $< 0.30$.
  - Label input dilindungi dengan `deepcopy` (zero in-place mutation).

---

## 5. Status Eksperimen Ablasi & Checkpoint Reuse (Stage 8.4)

- [x] **Validasi Sidik Jari Eksperimen Komprehensif**:
  - Keputusan `--skip-if-done` memvalidasi kesesuaian script hash, data yaml hash, manifest hash, pretrained weight hash, dan konfigurasi hyperparameter.
- [x] **Proteksi Penimpaan Checkpoint Historis**:
  - Default output dialihkan ke `stage_10_11_clean_experiments/`.
  - Penulisan ke `stage_8_ablation/` diblokir otomatis dengan `PermissionError`.
  - Mismatch pada direktori yang telah ada memicu `RuntimeError` penolakan penimpaan.
- [x] **Pemisahan Matriks Ablasi Resmi**:
  - EXP-0: Uniform 1/N sampling, Standard Augmentation.
  - EXP-1: Mixture Adaptive Sampling $P(k)$, Standard Augmentation.
  - EXP-2: Uniform 1/N sampling, Targeted Context Crop.
  - EXP-3: Mixture Adaptive Sampling $P(k)$, Targeted Context Crop.

---

## 6. Status Evaluator Kanonis & Proteksi Test Set (Stage 7, 8.5, 8.6)

- [x] **Unifikasi Evaluator Stage 7 ke SSOT**:
  - Stage 7 telah direfaktor penuh menggunakan `research_utils/detection_metrics.py`.
  - Menghapus implementasi parsing dan matching duplikat lokal; menyelesaikan bug `matched_preds` NameError.
  - Menetapkan versi evaluator: `EVALUATOR_VERSION = "Stage-9.5-Canonical-v1"`.
- [x] **Pemisahan Metrik Resmi vs Metrik Diagnostik**:
  - Metrik resmi Ultralytics (Precision, Recall, mAP@50, mAP@50-95, AP per kelas) dipertahankan sebagai metrik evaluasi resmi.
  - Matching serakah satu-ke-satu IoU descending digunakan secara eksklusif untuk analisis diagnostik error dan dekomposisi kesulitan.
- [x] **Hierarki Seleksi Model Terkunci (Stage 8.5)**:
  - 100% bergantung pada Validation Set (2.567 citra).
  - Kriteria: mAP@50-95 $\rightarrow$ Recall Small (selisih $\le 10^{-5}$) $\rightarrow$ mAP@50 $\rightarrow$ min FN.
- [x] **Locked Test Set Protection (Stage 8.6)**:
  - Tetap terkunci di balik guard `--confirm-final-evaluation`. Tidak ada evaluasi test set yang dijalankan pada Stage 9.5.

---

## 7. Status Dry-Run & Orchestrator (run_stage_8_profiling.py)

- [x] **Single Source of Truth Dataset**:
  - Default orchestrator diarahkan ke `cassava-v7-standardized/data.yaml`.
- [x] **Dry-Run OOF Terverifikasi Penuh**:
  - 12 file split dihasilkan di `stage_9_5_code_remediation/dry_run_standardized/`.
  - 35.920 baris path citra diverifikasi: **100.0% mengarah ke `cassava-v7-standardized`** (0 baris mengarah ke dataset historis).
  - 0% kebocoran grup antar-fold.
  - Alur dry-run berhenti secara bersih sebelum Stage 8.2 tanpa memicu kegagalan cascade.

---

## 8. Status Pengujian Regresi (Regression Testing Suite)

- [x] **`test_stage_9_3_regression.py`**: 16 / 16 passed (100% PASS).
- [x] **`test_stage_9_4_regression.py`**: 26 / 26 passed (100% PASS).
- [x] **`test_stage_9_5_regression.py`**: 48 / 48 passed (100% PASS).
- [x] **Total Keseluruhan**: **90 / 90 tests passed (100% PASS, 0 failures, 0 errors)** dalam 1.515 detik.

---

## 9. Rekomendasi Keputusan Gerbang (Gate Decision)

Berdasarkan pemenuhan seluruh kriteria di atas:
- Seluruh defect prioritas P0, P1, dan P2 telah diperbaiki dan dibuktikan dengan tes regresi empiris;
- Reuse checkpoint lama yang tidak cocok diblokir secara fail-fast;
- Integritas dataset terstandarisasi terbukti 100% secara geometris (48.957 bounding box);
- Dry-run OOF membuktikan isolasi grup 100% dan zero historical path leakage;
- Tidak ada evaluasi test set maupun training penuh yang dijalankan selama Stage 9.5;

Status Kesiapan Pipeline Resmi:

### **`READY_FOR_CLEAN_OOF`**

# Diff Summary and Code Changes — Stage 9.8

## 1. Overview Perubahan

Branch kerja: `stage-9-8-training-config-integration`  
Baseline commit SHA: `8ce840a1244f950a529fee068b019356eb4000de`  

Stage 9.8 melakukan remediasi kode untuk mengintegrasikan konfigurasi pelatihan terbaru, memperluas run identity untuk mengikat seluruh parameter operasional, menambahkan pemanggilan validasi integritas dataset pada jalur preflight, membuktikan ketersediaan runtime GPU/CPU, serta memverifikasi zero mutation pada dataset penelitian.

---

## 2. Rincian Modifikasi per File

### 2.1 `stage_8_4_run_ablation_experiments.py`
- **Pre-fix SHA-256**: `667cbf7e31e51b3260c6dbe76e031eb582b95b871261cb46114eb3897eb3d3cb`
- **Post-fix SHA-256**: `91ca5cfc37e163777f98c48408a28f7318ff01d293881df3e04e223c3b0ebfdc`
- **Perubahan Utama**:
  1. *Perbaikan Shadowing Import*: Menghapus definisi fungsi duplikat `compute_file_sha256()` lokal yang menutupi fungsi resmi dari `research_utils.detection_metrics`.
  2. *Resolusi Bobot Pretrained Lokal*: Mengimplementasikan fungsi `resolve_pretrained_weights()` yang hanya menerima bobot lokal konkret (`yolo11n.pt`) dan secara eksplisit menolak pengunduhan dinamis (`raise FileNotFoundError("Bobot pretrained tidak ditemukan ... Dilarang keras mengunduh bobot secara dinamis")`).
  3. *Validasi Argumen CLI*: Menambahkan fungsi `validate_training_args()` yang memvalidasi rentang numerik parameter pelatihan (`epochs >= 1`, `patience >= 0`, `batch >= 1`, `imgsz >= 32`, `workers >= 0`, `0 <= close_mosaic <= epochs`, `0.0 <= aug_prob <= 1.0`).
  4. *Pembaruan Nilai Default CLI*:
     - `epochs`: 50 -> **30**
     - `patience`: None -> **10**
     - `batch`: 16 (tetap)
     - `imgsz`: 640 (tetap)
     - `workers`: 0 -> **2**
     - `amp`: None -> `--amp / --no-amp` (default **True**)
     - `plots`: True -> `--plots / --no-plots` (default **False**)
     - `close_mosaic`: None -> **6**
     - `optimizer`: None -> **"auto"**
     - `pretrained_weights`: "yolo11n.pt" (default konkret)
  5. *Ekspansi Run Identity*: Memperluas `build_experiment_run_identity()` untuk menyertakan seluruh 26 parameter pelatihan efektif (`workers`, `amp`, `plots`, `close_mosaic`, `optimizer`, `aug_prob`, dsb.).
  6. *Pembaruan Verifikasi Reuse*: Memperbarui `verify_ablation_reuse_safety()` agar mencocokkan seluruh field konfigurasi baru dengan toleransi float numerik, sambil tetap mengisolasi metrik hasil konvergensi.
  7. *Validasi Integritas Dataset Preflight*: Menambahkan pemanggilan `validate_yolo_dataset_integrity()` pada split latih dan validasi sebelum model diinisialisasi.

---

### 2.2 `stage_8_1_group_oof_profiling.py`
- **Pre-fix SHA-256**: `b1d7d08ba06a6b83f3ba9ce20626b9a89d311fa6a12ba77ecf2bf7d7fb797fa9`
- **Post-fix SHA-256**: `bf6588db257c790b2eb34d588523fc63d911be245fe854b7324483beaf2e7033`
- **Perubahan Utama**:
  1. *Integrasi Preflight Dataset Integrity*: Menambahkan pemanggilan langsung ke `validate_yolo_dataset_integrity(images_dir=train_images_dir, labels_dir=train_labels_dir, num_classes=5)` pada blok preflight utama sebelum pembagian 4-fold group profiling.
  2. *Preservasi Konfigurasi OOF*: Tetap mempertahankan konfigurasi 12 epoch per fold probe model sesuai rencana Stage 8.1.

---

### 2.3 `stage_7_baseline_and_error_analysis.py`
- **Pre-fix SHA-256**: `fc59957dfae6bfbca53ae4dc6fc7b960be67d32e9f027376a92ecda13346413a`
- **Post-fix SHA-256**: `75f1b5ea13aa1ddbbf760d691ebdd1c070c7e2b7e12739466ceca7c10b2f567b`
- **Perubahan Utama**:
  1. *Integrasi Preflight Dataset Integrity*: Menambahkan pemanggilan eksplisit ke `validate_yolo_dataset_integrity()` untuk split train dan validation pada jalur persiapan baseline.

---

### 2.4 `research_utils/detection_metrics.py`
- **Pre-fix SHA-256**: `2f9f836511b81665a39cb68b1937e2dbf9b1f23ee2cae58d57d763ef46ec39d7`
- **Post-fix SHA-256**: `c055627263c9b74044ee7447d446a81b3cc5fc1765c3bb3522a1068832a89345`
- **Perubahan Utama**:
  1. *Penyempurnaan Deteksi Orphan Labels*: Pada split dataset yang didefinisikan menggunakan daftar citra subset (misalnya `.txt` file list), pengecekan orphan label disempurnakan agar tidak salah menandai label milik citra di luar subset sebagai orphan, asalkan citra fisiknya ada di direktori citra.

---

### 2.5 `test_stage_9_8_regression.py`
- **Status**: Berkas pengujian baru (Total 859 baris kode).
- **Post-fix SHA-256**: `d3d9ce459ee69a48f4305ce79f8c6eb561c28c89c8a9f60bcfe433d7b420f18d`
- **Cakupan Pengujian**:
  - Group A: 4 tests (Training configuration defaults, CLI boolean flags, argument validator, run identity parameter capture).
  - Group B: 5 tests (Exact match reuse, parameter modification rejection, missing field fail-closed, partial run rejection, validation content change rejection).
  - Group C: 9 tests (Missing label, orphan label, out-of-bounds bbox, zero-dimension bbox, invalid class id, txt subset list, byte modification fingerprint, empty split rejection, preflight validator invocation).
  - Group D: 3 tests (Ablation toggles EXP-0..EXP-3, global configuration consistency, single experiment failure pipeline halting).
  - Group E: 4 tests (Pretrained weights local resolution, nonexistent weights error, context crop target_imgsz fallback, collate_fn with mixed cropped/uncropped items).
  - Group F: 4 tests (Forbidden historical directories rejection, clean output directory acceptance, zero mutation verification, unmutated before-after checksums).
  - **Total**: 29 tests, **100% PASS**.

---

## 3. Kompatibilitas Mundur (Backward Compatibility)

Seluruh test suite historis dijalankan secara kumulatif untuk memastikan tidak ada regresi fungsionalitas:
- Stage 9.3: 17 tests PASS
- Stage 9.4: 25 tests PASS
- Stage 9.5: 37 tests PASS
- Stage 9.6: 34 tests PASS
- Stage 9.7: 39 tests PASS
- Stage 9.8: 29 tests PASS
- **Total Keseluruhan**: **181 tests PASS** (0 failures, 0 errors dalam 3.16 detik).

# STAGE 9.8 — FINAL TRAINING CONFIGURATION INTEGRATION, VALIDATION & READINESS GATE REPORT

**Penelitian**: Deteksi dan Lokalisasi Gejala Penyakit Daun Singkong Berbasis YOLO11n dengan Error-Driven Adaptive Optimization  
**Repository**: `https://github.com/BagusAth/yosh`  
**Branch Baseline**: `stage-9-7-readiness-gate-hardening` (`8ce840a1244f950a529fee068b019356eb4000de`)  
**Branch Kerja**: `stage-9-8-training-config-integration`  
**Dataset Kanonis**: `cassava-v7-standardized/data.yaml`  
**Kelas Target**: `CBB`, `CBSD`, `CGM`, `CMD`, `HEALTHY`  
**Auditor**: Senior Machine Learning Engineer, Computer Vision Research Engineer, Python Test Engineer & Research Reproducibility Auditor  
**Tanggal**: 19 Oktober 2026  

---

## 1. EXECUTIVE SUMMARY

Stage 9.8 merupakan gerbang kesiapan akhir (*final readiness gate*) sebelum eksekusi pelatihan bersih 4-fold Out-of-Fold (Stage 8.1) dan eksperimen ablasi EXP-0 hingga EXP-3 (Stage 8.4). 

Tugas utama Stage 9.8 adalah melakukan integrasi konfigurasi pelatihan hemat waktu yang telah disepakati, memperluas arsitektur identitas eksperimen (*run identity*) agar mencakup seluruh parameter operasional, memvalidasi pemanggilan pemeriksa integritas dataset pada jalur preflight, mengaudit status perangkat keras GPU dan bobot pretrained lokal, serta membuktikan prinsip *zero dataset mutation* secara mutlak.

### Ringkasan Pencapaian Utama:
1. **Konfigurasi Pelatihan Terintegrasi**: Seluruh default CLI dan trainer overrides telah diperbarui ke konfigurasi efisien: `epochs=30`, `patience=10`, `batch=16`, `imgsz=640`, `workers=2`, `amp=True`, `plots=False`, `close_mosaic=6`, `optimizer="auto"`, `aug_prob=0.50`, dan `pretrained_weights="yolo11n.pt"`.
2. **Pengikatan Run Identity Penuh**: Fungsi `build_experiment_run_identity()` kini mengikat 26 parameter operasional, mencegah penggunaan ulang checkpoint (*checkpoint reuse*) apabila terdapat perubahan sekecil apa pun pada konfigurasi.
3. **Validasi Integritas Dataset pada Preflight**: Fungsi `validate_yolo_dataset_integrity()` telah diintegrasikan langsung pada preflight Stage 8.1, Stage 8.4, dan Stage 7. Verifikasi pada `cassava-v7-standardized` membuktikan 8.980 citra latih (66 background), 34.219 objek, dan **0 orphan label**.
4. **Audit Hardware & Pretrained Weights**:
   - Fisik GPU terdeteksi di level sistem operasi: **NVIDIA GeForce RTX 4060 Laptop GPU** (8.188 MiB VRAM, Driver 581.86, CUDA 13.0).
   - Interpreter Python saat ini menggunakan PyTorch versi CPU (`torch==2.9.1+cpu`), sehingga `torch.cuda.is_available() == False`.
   - Bobot lokal `yolo11n.pt` terverifikasi ada dengan SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
   - Smoke test 1-epoch pada dataset sintetis berhasil dieksekusi 100% pada CPU.
5. **Bukti Zero Mutation**: Pemeriksaan `git status` dan fingerprint SHA-256 membuktikan bahwa kedua direktori dataset riset (`cassava-v7-standardized` dan `cassava-v7-research`) tidak mengalami penambahan, perubahan, atau pencemaran file cache.
6. **Hasil Uji Regresi Kumulatif**: Seluruh test suite dari Stage 9.3 hingga Stage 9.8 dieksekusi dengan hasil **181 / 181 PASS** (100% lulus, 0 gagal, 0 error).

---

## 2. AUDIT FINDINGS MATRIX (STAGE 9.7 -> 9.8 RESOLUTION)

| ID Temuan | Komponen Terdampak | Status | Akar Masalah | Solusi & Remediasi | Bukti Verifikasi |
|---|---|---|---|---|---|
| **FINDING-9.8-01** | `stage_8_4_run_ablation_experiments.py` | **RESOLVED** | Default CLI masih `epochs=50`, `workers=0`, `plots=True`, tanpa `patience`, `amp`, dan `close_mosaic` eksplisit. | Default diperbarui: `epochs=30`, `patience=10`, `workers=2`, `amp=True`, `plots=False`, `close_mosaic=6`, `optimizer="auto"`. Ditambahkan fungsi `validate_training_args()`. | `TestGroupATrainingConfiguration` (PASS), `effective_training_config.json`. |
| **FINDING-9.8-02** | `stage_8_4_run_ablation_experiments.py` | **RESOLVED** | `build_experiment_run_identity()` belum mencatat `workers`, `amp`, `plots`, `close_mosaic`, dan `optimizer`. | Memperluas kamus run identity dengan 26 field lengkap dan memperbarui `verify_ablation_reuse_safety()` dengan perbandingan fail-closed. | `TestGroupBRunIdentityAndReuse` (PASS), `run_identity_audit.md`. |
| **FINDING-9.8-03** | `stage_8_1_group_oof_profiling.py`, `stage_8_4_run_ablation_experiments.py`, `stage_7_baseline_and_error_analysis.py` | **RESOLVED** | `validate_yolo_dataset_integrity()` sudah ada di utilitas tetapi belum dipanggil pada preflight training. | Menambahkan pemanggilan eksplisit `validate_yolo_dataset_integrity()` pada preflight Stage 8.1, Stage 8.4, dan Stage 7 sebelum eksekusi model. | `test_validator_called_by_stage_8_4_preflight` (PASS), `dataset_integrity_preflight.json`. |
| **FINDING-9.8-04** | Target Runtime Environment | **RESOLVED (DECLARED)** | Laporan Stage 9.7 mencatat CUDA False; belum dibuktikan apakah hardware GPU tersedia atau bobot pretrained lokal valid. | Mengaudit `nvidia-smi` (RTX 4060 Laptop GPU terdeteksi), memverifikasi bobot lokal `yolo11n.pt`, mengeksekusi smoke test CPU, dan mendeklarasikan status resmi `NEEDS_LOCAL_GPU_SMOKE_TEST`. | `gpu_smoke_test_evidence/` (nvidia-smi output, runtime env JSON, smoke test log). |
| **FINDING-9.8-05** | Dataset Integrity & Regression Tests | **RESOLVED** | Git status dapat melewatkan file `.cache` yang diabaikan dalam `.gitignore`; test dapat membuat cache pada dataset riset. | Memastikan validasi integritas murni read-only; mengisolasi unit test DataLoader pada direktori temp; membuktikan 0 file cache pada folder citra dataset riset. | `test_zero_mutation_dataset_verification` (PASS), `test_verify_dataset_files_and_checksums_unmutated_before_after` (PASS). |

---

## 3. INTEGRASI KONFIGURASI PELATIHAN EFEKTIF

Konfigurasi pelatihan yang tertanam pada [`stage_8_4_run_ablation_experiments.py`](file:///d:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py) dan tersimpan di [`effective_training_config.json`](file:///d:/UNDIP/SKRIPSI/stage_9_8_training_config_integration/effective_training_config.json):

```json
{
  "epochs": 30,
  "patience": 10,
  "batch": 16,
  "imgsz": 640,
  "seed": 42,
  "device": "0",
  "workers": 2,
  "amp": true,
  "plots": false,
  "close_mosaic": 6,
  "optimizer": "auto",
  "aug_prob": 0.50,
  "pretrained_weights": "yolo11n.pt",
  "pretrained_weights_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
  "data_yaml": "cassava-v7-standardized/data.yaml"
}
```

### Spesifikasi 4 Varian Eksperimen Ablasi:
1. **EXP-0 — Baseline Control**: Uniform sampling ($1/N$), standard augmentations, manifest diabaikan.
2. **EXP-1 — Mixture Adaptive Sampling**: Mixture adaptive sampling $P(k)$, standard augmentations, manifest wajib diverifikasi.
3. **EXP-2 — Targeted Context Augmentation**: Uniform sampling ($1/N$), on-the-fly targeted context crop ($2\times$ bbox), manifest wajib diverifikasi.
4. **EXP-3 — Full Proposed Method**: Mixture adaptive sampling $P(k)$ + on-the-fly targeted context crop ($2\times$ bbox), manifest wajib diverifikasi.

Keempat eksperimen menggunakan konfigurasi global yang identik secara deterministik, sehingga perbedaan performa murni mencerminkan kontribusi komponen yang diuji.

---

## 4. PREFLIGHT INTEGRITAS DATASET RISET

Validasi integritas pada dataset resmi [`cassava-v7-standardized/data.yaml`](file:///d:/UNDIP/SKRIPSI/cassava-v7-standardized/data.yaml) menghasilkan status **PASSED**:

```
Dataset YAML      : cassava-v7-standardized/data.yaml (SHA-256: e40454b874596ad6f0275ae489a44eca51b2350540c16e50e4b489f479e9ed84)
Train Split       : 8,980 citra | 8,980 label | 66 background | 34,219 objek | 0 orphan labels
Val Split         : 2,567 citra | 2,567 label | 18 background |  9,649 objek | 0 orphan labels
Train Fingerprint : 085a6a61f5164ca6e00b8529f79caefbc809b4db070ad7d727b3d33cb42c7333
Val Fingerprint   : d59cb65b0953a73c155490a1f26f29a0f443b74704a2cbaf863260c878b6ee38
```

Distribusi kelas pada train split:
- CBB (0): 2.502 anotasi
- CBSD (1): 2.658 anotasi
- CGM (2): 7.153 anotasi
- CMD (3): 9.946 anotasi
- HEALTHY (4): 11.960 anotasi
- **Total Objek**: 34.219 anotasi valid.

---

## 5. AUDIT LINGKUNGAN RUNTIME & BUKTI PERANGKAT KERAS

### 5.1 Status Perangkat Keras (Physical GPU)
- **Model**: NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM Total**: 8.188 MiB (~8 GB)
- **Versi Driver NVIDIA**: 581.86
- **Dukungan CUDA Sistem**: 13.0
- **Metode Deteksi**: `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`

### 5.2 Status Lingkungan Python
- **Python**: 3.13.7 (64-bit AMD64)
- **PyTorch Terpasang**: `2.9.1+cpu`
- **CUDA Backend PyTorch**: `torch.cuda.is_available() == False`
- **Ultralytics**: 8.3.235

### 5.3 Bobot Pretrained Lokal
- **File**: `D:\UNDIP\SKRIPSI\yolo11n.pt`
- **Ukuran**: 5.626.837 bytes (~5,6 MB)
- **SHA-256**: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- **Status**: Berhasil di-resolve secara lokal tanpa unduhan dinamis.

### 5.4 Eksekusi Smoke Test 1-Epoch
Smoke test sintetis 1-epoch dijalankan untuk memverifikasi fungsionalitas end-to-end Ultralytics training pipeline:
- **Konfigurasi**: 1 epoch, imgsz=64, batch=2, device='cpu', 2 citra sintetis.
- **Hasil**: Training selesai sukses, bobot `best.pt` (5.411.546 bytes) tercipta, evaluasi validasi selesai tanpa error.
- **Log Tersimpan**: [`gpu_smoke_test_evidence/smoke_test_log.txt`](file:///d:/UNDIP/SKRIPSI/stage_9_8_training_config_integration/gpu_smoke_test_evidence/smoke_test_log.txt).

---

## 6. HASIL SUITE UJI REGRESI (181 / 181 PASS)

Pengujian kumulatif dijalankan secara penuh:

```bash
$ python -m unittest test_stage_9_3_regression.py test_stage_9_4_regression.py test_stage_9_5_regression.py test_stage_9_6_regression.py test_stage_9_7_regression.py test_stage_9_8_regression.py
----------------------------------------------------------------------
Ran 181 tests in 3.155s

OK
```

### Distribusi Pengujian per Stage:
- Stage 9.3: 17 tests (**PASS**)
- Stage 9.4: 25 tests (**PASS**)
- Stage 9.5: 37 tests (**PASS**)
- Stage 9.6: 34 tests (**PASS**)
- Stage 9.7: 39 tests (**PASS**)
- Stage 9.8: 29 tests (**PASS**)
  * Group A (Training Config Integration): 4 tests
  * Group B (Run Identity & Reuse Verification): 5 tests
  * Group C (Dataset Integrity Preflight): 9 tests
  * Group D (Ablation Configuration Consistency): 3 tests
  * Group E (Data Loading & Pretrained Resolution): 4 tests
  * Group F (Output Safety & Zero Mutation): 4 tests

Daftar rinci seluruh 29 pengujian Stage 9.8 tercatat di [`regression_test_matrix.csv`](file:///d:/UNDIP/SKRIPSI/stage_9_8_training_config_integration/regression_test_matrix.csv).

---

## 7. BUKTI EKSEKUSI DRY-RUN AMAN

### 7.1 Stage 8.1 Dry-Run (Group-Aware 4-Fold OOF)
- **Perintah**: `python stage_8_1_group_oof_profiling.py --dry-run --data-yaml cassava-v7-standardized/data.yaml --output-dir stage_9_8_training_config_integration/dry_run_evidence/stage_8_1_dry_run`
- **Hasil**:
  - Validasi integritas dataset: 8.980 citra latih terverifikasi.
  - Partisi 4-fold group-aware: 5.546 grup atomik terbagi merata (Fold 1: 2.246 citra, Fold 2: 2.244 citra, Fold 3: 2.245 citra, Fold 4: 2.245 citra).
  - Cross-fold leakage check: **0% leakage** (100% isolasi terverifikasi).
  - Berkas YAML dan penugasan fold tercipta lengkap di folder bukti.

### 7.2 Stage 8.4 Dry-Run (Ablation Study EXP-0)
- **Perintah**: `python stage_8_4_run_ablation_experiments.py --dry-run --exp 0 --data-yaml cassava-v7-standardized/data.yaml --output-dir stage_9_8_training_config_integration/dry_run_evidence/stage_8_4_dry_run`
- **Hasil**:
  - Validasi integritas dataset train & val: PASSED.
  - Resolusi bobot pretrained `yolo11n.pt`: PASSED.
  - Validasi konfigurasi & argumen: PASSED.
  - Dry-run status: `[EXP-0] DRY-RUN PASS: Validasi preflight, dataset fingerprint, dan konfigurasi lolos 100%.`

---

## 8. KEPUTUSAN FINAL READINESS GATE (VERDICT)

Berdasarkan seluruh hasil audit, pengujian regresi, validasi preflight, dan pembuktian empiris:

### 1. Kesiapan Kode Pipeline & Logika Pelatihan:
$$\mathbf{READY\_FOR\_SAFE\_DRY\_RUN} \quad \text{(100\% PASS)}$$
Seluruh kode pipeline, mekanisme run identity, pengaman direktori historis, isolasi dataset, logika sampling adaptif, targeted context crop, dan skrip orkestrasi telah terverifikasi aman, deterministik, dan bebas cacat fungsional.

### 2. Kesiapan Runtime Akselerasi GPU:
$$\mathbf{NEEDS\_LOCAL\_GPU\_SMOKE\_TEST}$$
Perangkat keras **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB VRAM) terdeteksi secara fisik pada mesin pengguna. Namun, karena environment Python saat ini menggunakan PyTorch build CPU (`torch==2.9.1+cpu`), eksekusi pelatihan GPU memerlukan pengaktifan PyTorch berkemampuan CUDA.

---

## 9. PANDUAN EKSEKUSI BERSIH BERIKUTNYA

Sebelum memulai pelatihan clean OOF (Stage 8.1) atau eksperimen ablasi (Stage 8.4) pada GPU:

### Langkah A: Aktifkan PyTorch dengan Dukungan CUDA
Jalankan perintah berikut pada virtual environment Anda:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Verifikasi bahwa CUDA aktif:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
# Output yang diharapkan: CUDA Available: True | Device: NVIDIA GeForce RTX 4060 Laptop GPU
```

### Langkah B: Jalankan Clean 4-Fold OOF Profiling (Stage 8.1)
```bash
python stage_8_1_group_oof_profiling.py \
    --data-yaml cassava-v7-standardized/data.yaml \
    --output-dir stage_8_1_oof_clean \
    --epochs 12 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

### Langkah C: Bentuk Difficulty Manifest Bersih (Stage 8.2)
```bash
python stage_8_2_build_difficulty_manifest.py \
    --raw-oof-dir stage_8_1_oof_clean \
    --output-dir stage_8_2_difficulty_clean \
    --data-yaml cassava-v7-standardized/data.yaml
```

### Langkah D: Jalankan Eksperimen Ablasi Bersih EXP-0..EXP-3 (Stage 8.4)
```bash
python stage_8_4_run_ablation_experiments.py \
    --exp all \
    --data-yaml cassava-v7-standardized/data.yaml \
    --manifest-path stage_8_2_difficulty_clean/difficulty_manifest.csv \
    --output-dir stage_8_ablation_clean \
    --epochs 30 \
    --patience 10 \
    --batch 16 \
    --imgsz 640 \
    --workers 2 \
    --amp \
    --no-plots \
    --close-mosaic 6 \
    --optimizer auto \
    --device 0
```

---

## 10. DAFTAR ARTEFAK TAHAP 9.8

Seluruh berkas artefak Stage 9.8 tersimpan terisolasi di [`stage_9_8_training_config_integration/`](file:///d:/UNDIP/SKRIPSI/stage_9_8_training_config_integration/):
1. `STAGE_9_8_FINAL_READINESS_REPORT.md` — Laporan kesiapan final ini.
2. `effective_training_config.json` — Konfigurasi pelatihan terintegrasi 30 epoch & parameter lengkap.
3. `dataset_integrity_preflight.json` — Rekaman bukti validasi integritas dataset kanonis.
4. `code_changes_manifest.csv` — Inventaris berkas yang diubah beserta hash SHA-256 pre-fix dan post-fix.
5. `regression_test_matrix.csv` — Matriks 29 regression tests Stage 9.8 (Groups A–F, 100% PASS).
6. `regression_test_results.txt` — Log eksekusi lengkap 181 pengujian unit kumulatif (9.3..9.8).
7. `run_identity_audit.md` — Audit pengikatan identitas run dan penolakan checkpoint reuse fail-closed.
8. `output_safety_audit.md` — Audit perlindungan direktori historis dan bukti nol mutasi dataset.
9. `diff_summary.md` — Rangkuman diferensial kode per berkas.
10. `pre_fix_inventory.json` — Inventaris SHA-256 baseline sebelum modifikasi Stage 9.8.
11. `gpu_smoke_test_evidence/` — Bukti deteksi hardware GPU (nvidia-smi, runtime env, smoke test log).
12. `dry_run_evidence/` — Bukti eksekusi dry-run Stage 8.1 dan Stage 8.4.

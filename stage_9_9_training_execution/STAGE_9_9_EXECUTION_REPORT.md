# STAGE 9.9 — LAPORAN EKSEKUSI PELATIHAN GPU BERSIH, PROVENANCE OOF, & VALIDASI ABLASI

Tanggal Eksekusi: 20 Oktober 2026  
Status Stage: **SUKSES PENUH (COMPLETED & AUDITED)**  
Branch Kerja: `stage-9-9-clean-gpu-training-execution`  
Baseline Commit: `4058024d44c0e3390b6c89278f49221bc181abfe` (HEAD Stage 9.8)  
Evaluator & Auditor: Senior Machine Learning Engineer, Computer Vision Research Engineer, & Research Reproducibility Auditor

---

## 1. Ringkasan Eksekutif & Verifikasi Lingkungan

Tujuan utama dari Stage 9.9 adalah menjalankan eksekusi pelatihan sungguhan (*real clean GPU training*) pada perangkat keras lokal untuk memvalidasi pipeline penelitian **YOLO11n dengan Error-Driven Adaptive Optimization** secara end-to-end tanpa mock, tanpa bypass, dan tanpa manipulasi data.

Seluruh target teknis telah diselesaikan secara tuntas dan diaudit secara independen:
1. **Lingkungan CUDA Terverifikasi**: Virtual environment `.venv-cuda` dengan Python 3.13.7, PyTorch 2.9.1+cu130, dan Ultralytics 8.3.235 berjalan pada NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM).
2. **Zero Mutation Source Dataset**: Dibangun *isolated runtime dataset mirror* (`cassava-v7-runtime-mirror`) menggunakan NTFS hardlinks untuk citra dan salinan bersih untuk label. Seluruh file cache Ultralytics (`labels.cache`) terisolasi di dalam mirror, dan dataset sumber kanonis `cassava-v7-standardized` terverifikasi 0% termutasi.
3. **Stage 8.1 4-Fold Group-Aware OOF Profiling Selesai**: Melatih 4 fold (12 epoch/fold) pada 8.980 citra latih dengan isolasi holdout group 100% (0% group leakage). Menghasilkan `oof_predictions_raw.csv` (1.458.979 prediksi mentah; SHA-256: `0e1a89053a4dd1ffc4078b5e5365a777cbef74a991d222072c9eaebf2fbc02b2`).
4. **Stage 8.2 Difficulty Manifest Selesai & Sanity Gate PASS**: Menghasilkan `train_image_difficulty_manifest.csv` dan sidecar provenance JSON yang mencocokkan SHA-256 raw OOF secara eksak. Sanity gate lulus penuh (**PASS**, Hard Ratio: 10,61%, Dynamic Range: 0,6106, ESS Ratio: 0,9130).
5. **Stage 8.4 Ablation Experiments Selesai Penuh**: 4 eksperimen ablasi (EXP-0, EXP-1, EXP-2, EXP-3) dilatih secara berurutan masing-masing 30 epoch menggunakan konfigurasi terdaftar identik pada GPU RTX 4060.
6. **Stage 8.5 Model Selection Selesai**: Berdasarkan kriteria hierarki terkunci pada Validation Set (2.567 citra), model **EXP-2 (Targeted Context Augmentation)** terpilih sebagai model terbaik dengan **mAP@50-95 = 27,26%** dan **mAP@50 = 37,27%** (+1,44% peningkatan mAP@50 di atas baseline).
7. **Batasan Metodologis Terjaga Ketat**: Split test (1.282 citra) terkunci rapat; Stage 8.6 tidak disentuh; branch `main` tidak disentuh.

---

## 2. Profil Lingkungan Komputasi & Perangkat Keras

Data runtime environment direkam dalam [`runtime_environment.json`](file:///D:/UNDIP/SKRIPSI/stage_9_9_training_execution/runtime_environment.json):

| Komponen | Spesifikasi / Nilai | Verifikasi |
| :--- | :--- | :---: |
| **Sistem Operasi** | Windows 11 Home Single Language (Build 26100) | Valid |
| **GPU Hardware** | NVIDIA GeForce RTX 4060 Laptop GPU | Terdeteksi |
| **Total VRAM** | 8,188 MiB (8.00 GiB) | Valid |
| **Compute Capability** | SM 8.9 (Ada Lovelace) | Valid |
| **Python Runtime** | Python 3.13.7 (64-bit) | Terverifikasi |
| **PyTorch Version** | 2.9.1+cu130 (CUDA 13.0) | Terverifikasi |
| **Ultralytics Version** | 8.3.235 | Terverifikasi |
| **Smoke Test CUDA** | 1-epoch synthetic model training pada `cuda:0` | Lolos (100%) |

---

## 3. Audit Remediasi Kode & Penanganan Error Runtime

Selama eksekusi Stage 9.9, dilakukan 4 perbaikan teknis minimum yang krusial untuk kestabilan pipeline tanpa mengubah logika ilmiah:

1. **Parameter Pretrained Weights Eksplisit pada Stage 8.1**  
   - *File*: [`stage_8_1_group_oof_profiling.py`](file:///D:/UNDIP/SKRIPSI/stage_8_1_group_oof_profiling.py)
   - *Masalah*: Script sebelumnya mengasumsikan nama model string tanpa path absolut, berisiko mengunduh `yolo11n.pt` otomatis.
   - *Perbaikan*: Ditambahkan argumen CLI `--pretrained-weights`, path absolut bobot lokal di-resolve, dan diperiksa keberadaannya sebelum training dimulai.

2. **Fleksibilitas Keyword Matching Bipartit Kanonis**  
   - *File*: [`research_utils/detection_metrics.py`](file:///D:/UNDIP/SKRIPSI/research_utils/detection_metrics.py#L86-L120)
   - *Masalah*: `greedy_bipartite_matching` mendefinisikan parameter pertama sebagai `gt_objects`, sedangkan `stage_8_2` dan `stage_8_5` memanggil dengan keyword `gt_boxes=gt_boxes`, menimbulkan `TypeError`.
   - *Perbaikan*: Ditambahkan parameter `gt_boxes: Any = None` dan fallback backward-compatible sehingga mendukung baik pemanggilan posisional maupun keyword `gt_boxes=`.

3. **Inkompatibilitas Parameter DataLoader pada Stage 8.3**  
   - *File*: [`stage_8_3_adaptive_pipeline.py`](file:///D:/UNDIP/SKRIPSI/stage_8_3_adaptive_pipeline.py#L520-L532)
   - *Masalah*: `build_dataloader()` dari Ultralytics versi 8.3.235 tidak menerima keyword argument `device=self.device`, menyebabkan kegagalan `TypeError`.
   - *Perbaikan*: Dihapus kwarg `device=self.device` pada panggilan `build_dataloader()`, diselaraskan dengan implementasi upstream `DetectionTrainer`.

4. **Pembatasan Worker Multiprocessing Evaluasi Validasi Stage 8.4**  
   - *File*: [`stage_8_4_run_ablation_experiments.py`](file:///D:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py#L827-L835)
   - *Masalah*: Pada Windows, pemanggilan `eval_model.val()` tanpa parameter `workers` menggunakan default Ultralytics (8 worker), memicu kehabisan memory mapping shared file Windows (`ERROR_COMMITMENT_LIMIT 1455`) setelah sesi training panjang.
   - *Perbaikan*: Parameter `workers=workers` diteruskan secara eksplisit ke `eval_model.val()`, memastikan penggunaan 2 worker secara konsisten dan bebas crash.

5. **Verifikasi Regresi Komprehensif**  
   - *Test Suite*: Seluruh 31 unit test pada [`test_stage_9_8_regression.py`](file:///D:/UNDIP/SKRIPSI/test_stage_9_8_regression.py) lulus 100% (`Ran 31 tests in 2.699s, OK`).

---

## 4. Audit Integritas Dataset & Isolasi Runtime Mirror

Untuk menjamin bahwa dataset penelitian historis tidak tersentuh dan tidak mengalami penambahan file cache, dibuat cermin runtime pada `cassava-v7-runtime-mirror`:

| Item Audit | Dataset Sumber (`cassava-v7-standardized`) | Dataset Cermin (`cassava-v7-runtime-mirror`) | Status Audit |
| :--- | :--- | :--- | :---: |
| **Path data.yaml** | `cassava-v7-standardized/data.yaml` | `cassava-v7-runtime-mirror/data.yaml` | Terisolasi |
| **SHA-256 data.yaml** | `e40454b874596ad6f0275ae489a44eca51b2350540c16e50e4b489f479e9ed84` | `019ade73a896521b80ef48e38d0e725db4eed98c684907068f8a17318f845a30` | Valid (path adjusted) |
| **Jumlah Citra Train** | 8.980 citra | 8.980 citra (NTFS hardlinks) | Identik 100% |
| **Jumlah Label Train** | 8.980 file (34.219 objek) | 8.980 file (34.219 objek) | Identik 100% |
| **Jumlah Citra Val** | 2.567 citra | 2.567 citra (NTFS hardlinks) | Identik 100% |
| **Jumlah Citra Test** | 1.282 citra (Terkunci) | Tidak dimuat ke mirror | Terkunci & Aman |
| **Lokasi Cache Training** | Tidak ada penulisan cache baru | `cassava-v7-runtime-mirror/train/labels.cache` | **Zero Mutation** |

---

## 5. Hasil Stage 8.1 — Clean 4-Fold Group-Aware OOF Profiling

Output disimpan pada folder `stage_8_1_oof_clean_run_20261020/`:

- **Konfigurasi**: 4 Fold, 12 Epoch/fold, batch 16, imgsz 640, device 0, seed 42.
- **Cakupan Citra Evaluasi**: 8.980 citra unik (100,00% dari split latih).
- **Audit Kebocoran Group (Leakage Check)**: 0 citra bocor (0,00% group leakage). Setiap grup isolasi tanaman dijamin 100% berada pada sisi latih atau sisi evaluasi, tidak pernah keduanya sekaligus.
- **Waktu Eksekusi Total**: 1 jam 32 menit (Fold 1: 22,8m, Fold 2: 22,8m, Fold 3: 22,9m, Fold 4: 23,0m).
- **File Output Prediksi Mentah**:
  - File: [`stage_8_1_oof_clean_run_20261020/oof_predictions_raw.csv`](file:///D:/UNDIP/SKRIPSI/stage_8_1_oof_clean_run_20261020/oof_predictions_raw.csv)
  - Ukuran: 144.345.745 bytes (1,458,979 baris prediksi; conf >= 0.001)
  - SHA-256: `0e1a89053a4dd1ffc4078b5e5365a777cbef74a991d222072c9eaebf2fbc02b2`

---

## 6. Hasil Stage 8.2 — Difficulty Manifest & Sanity Gate Audit

Output disimpan pada folder `stage_8_2_difficulty_clean_20261020/`:

- **Input Prediksi**: `stage_8_1_oof_clean_run_20261020/oof_predictions_raw.csv`
- **Total Objek Ground Truth**: 34.219 objek pada 8.980 citra.
- **Taksonomi Error OOF Terhitung**:
  - True Positive (TP): 26.703 objek (78,04%)
  - Classification Confusion: 4.985 objek (14,57%)
  - Localization Error: 1.184 objek (3,46%)
  - False Negative Missed: 881 objek (2,57%)
  - Localization and Class Error: 466 objek (1,36%)

### Hasil Sanity Gate (Ambang Batas Metodologis)

| Metrik Sanity Gate | Nilai Terhitung | Ambang Batas / Kriteria | Status |
| :--- | :---: | :---: | :---: |
| **Hard Image Ratio ($D_\text{image} \ge 0.80$)** | **10,61%** (953 citra) | $\le 60\%$ PASS, $60-80\%$ REVIEW, $>80\%$ FAIL | **PASS** |
| **Dynamic Range ($P90 - P10$)** | **0,6106** | $\ge 0,30$ | **PASS** |
| **Effective Sample Size (ESS)** | **8.198,8** / 8.980 | Rasio $[0,65, 0,95]$ | **PASS** |
| **ESS Ratio** | **0,9130** | Rentang target valid | **PASS** |
| **Wrong Class Clipping Rate** | **9,28%** (506/5.451) | Finite, terkontrol | **PASS** |
| **Status Sanity Gate Akhir** | **PASS** | Seluruh kriteria terpenuhi | **PASS** |

Sidecar provenance [`train_image_difficulty_manifest.provenance.json`](file:///D:/UNDIP/SKRIPSI/stage_8_2_difficulty_clean_20261020/train_image_difficulty_manifest.provenance.json) mencatat hash eksak:
- `manifest_csv_sha256`: `4eb899772df8cb253ce3459e18e715202fb7f0b8f4e9c13c25777c9c753a6876`
- `raw_oof_sha256`: `0e1a89053a4dd1ffc4078b5e5365a777cbef74a991d222072c9eaebf2fbc02b2` (Cocok 100% dengan file output Stage 8.1)
- `dataset_inventory_fingerprint`: `af7dfbc43071fe6af6457d74823b7a46d5424074bb2de8216c55b71215d7c872`

---

## 7. Hasil Stage 8.4 — Ablation Study Experiments (EXP-0 s.d. EXP-3)

Seluruh 4 eksperimen ablasi dilatih pada GPU dengan konfigurasi terdaftar identik:
- `epochs=30`, `patience=10`, `batch=16`, `imgsz=640`, `seed=42`, `workers=2`, `amp=True`, `plots=False`, `close_mosaic=6`, `optimizer='auto'`, `aug_prob=0.50`, `device='0'`.

### Tabel Komparasi Hasil Validasi (Validation Set: 2.567 citra)

| ID | Eksperimen | Sampling | Augmentasi | Precision | Recall | mAP@50 | mAP@50-95 | Δ mAP@50 | Δ mAP@50-95 | Durasi |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | Baseline YOLO11n | Uniform ($1/N$) | Standard YOLO11n | 40,37% | 41,19% | **35,83%** | **26,20%** | Ref (0,0%) | Ref (0,0%) | 42,13m |
| **EXP-1** | Mixture Sampling | Mixture $P(k)$ | Standard YOLO11n | 39,41% | 42,37% | **34,53%** | **25,23%** | -1,30% | -0,97% | 47,91m |
| **EXP-2** | Targeted Context Aug | Uniform ($1/N$) | On-the-Fly Context Crop | **40,87%** | **42,67%** | **37,27%** | **27,26%** | **+1,44%** | **+1,06%** | 48,02m |
| **EXP-3** | Full Proposed System | Mixture $P(k)$ | On-the-Fly Context Crop | 39,40% | 42,64% | **34,98%** | **25,55%** | -0,85% | -0,65% | 43,16m |

### Performa Per-Kelas AP@50 (Validation Set)

| ID | Konfigurasi | CBB | CBSD | CGM | CMD | HEALTHY | Mean AP@50 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | Baseline YOLO11n | 33,33% | 39,18% | 38,48% | 32,14% | 36,02% | 35,83% |
| **EXP-1** | Mixture Sampling | 31,24% | 39,10% | 36,41% | 30,58% | 35,30% | 34,53% |
| **EXP-2** | **Targeted Context Aug** | **35,65%** | **42,08%** | **39,89%** | **32,29%** | **36,46%** | **37,27%** |
| **EXP-3** | Full Proposed System | 32,45% | 37,69% | 38,14% | 31,94% | 34,65% | 34,98% |

### Performa Per-Kelas AP@50-95 (Validation Set)

| ID | Konfigurasi | CBB | CBSD | CGM | CMD | HEALTHY | Mean AP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-0** | Baseline YOLO11n | 23,91% | 29,86% | 27,06% | 20,65% | 29,49% | 26,20% |
| **EXP-1** | Mixture Sampling | 22,29% | 29,71% | 25,53% | 19,64% | 28,96% | 25,23% |
| **EXP-2** | **Targeted Context Aug** | **25,24%** | **32,57%** | **27,49%** | **20,81%** | **30,20%** | **27,26%** |
| **EXP-3** | Full Proposed System | 23,30% | 28,55% | 26,78% | 20,55% | 28,56% | 25,55% |

---

## 8. Hasil Stage 8.5 — Seleksi Model Berbasis Validation Set

Berdasarkan aturan hierarki seleksi model terkunci (*locked decision hierarchy*):
1. **Kriteria Utama**: Validasi mAP@50-95 tertinggi.
2. **Tie-Break 1**: Recall_Small tertinggi (hanya jika tie numerik selisih $\le 10^{-5}$).
3. **Tie-Break 2**: Validasi mAP@50 tertinggi.
4. **Tie-Break 3**: Total Missed False Negative (FN) terendah.

### Evaluasi Seleksi:
- **Pemenang Mutlak**: **EXP-2 (Targeted Context Augmentation)**
- **Rasional Keputusan**: Kandidat EXP-2 mengungguli seluruh model pada metrik primer mAP@50-95 sebesar **27,26%** (selisih +1,06% dari baseline EXP-0 sebesar 26,20%), sehingga terpilih langsung pada Kriteria Utama tanpa memerlukan tie-break.
- **Karakteristik Error Model Pemenang (EXP-2)**:
  - Error Lokalisasi (*Localization Error*): **682** (terendah dari seluruh 4 model; EXP-0 = 738, EXP-1 = 786, EXP-3 = 766).
  - Kebingungan Klasifikasi (*Classification Confusion*): **263** (terendah dari seluruh 4 model; EXP-0 = 276, EXP-1 = 277, EXP-3 = 280).
  - Peningkatan konsisten di seluruh 5 kelas daun singkong, terutama CBSD (+2,90% AP@50, +2,71% AP@50-95) dan CBB (+2,32% AP@50, +1,33% AP@50-95).
- **Lokasi Checkpoint Juara**:
  - Path: `stage_8_ablation_clean_20261020/exp_2_targeted_aug/weights/best.pt`
  - SHA-256: `ceeb6d8a709d7274070a2734185294a2fa364020a597e742c382f7634f19b08f`

---

## 9. Register Artefak & Bukti Provenance

| Artefak / File | Lokasi Relatif | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Pretrained Weights** | `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| **Dataset Asli** | `cassava-v7-standardized/data.yaml` | `e40454b874596ad6f0275ae489a44eca51b2350540c16e50e4b489f479e9ed84` |
| **Runtime data.yaml** | `cassava-v7-runtime-mirror/data.yaml` | `019ade73a896521b80ef48e38d0e725db4eed98c684907068f8a17318f845a30` |
| **Raw OOF Predictions** | `stage_8_1_oof_clean_run_20261020/oof_predictions_raw.csv` | `0e1a89053a4dd1ffc4078b5e5365a777cbef74a991d222072c9eaebf2fbc02b2` |
| **Difficulty Manifest** | `stage_8_2_difficulty_clean_20261020/train_image_difficulty_manifest.csv` | `4eb899772df8cb253ce3459e18e715202fb7f0b8f4e9c13c25777c9c753a6876` |
| **Best Model EXP-0** | `stage_8_ablation_clean_20261020/exp_0_baseline/weights/best.pt` | `eeb5e7c772dbee094677de1b3b3d5c3bfd0fc1824aa0d18588321247bbb94f07` |
| **Best Model EXP-1** | `stage_8_ablation_clean_20261020/exp_1_adaptive_sampling/weights/best.pt` | `5e724c367074cd18ca5b89b8c7c8a83eb50e67050139810a26b7e3ebbefb2192` |
| **Best Model EXP-2 (Juara)** | `stage_8_ablation_clean_20261020/exp_2_targeted_aug/weights/best.pt` | `ceeb6d8a709d7274070a2734185294a2fa364020a597e742c382f7634f19b08f` |
| **Best Model EXP-3** | `stage_8_ablation_clean_20261020/exp_3_full_proposed/weights/best.pt` | `1c828ca367469611b8bbf68625ae7ee6d30ca74122d9b2e77e84d47348981446` |
| **Ablation Comparison CSV** | `stage_9_9_training_execution/ablation_metrics_comparison.csv` | Terverifikasi |
| **Run Manifest JSON** | `stage_9_9_training_execution/training_run_manifest.json` | Terverifikasi |

---

## 10. Kesimpulan & Status Kesiapan Tahap Selanjutnya

1. Seluruh proses pelatihan fisik GPU untuk **Stage 8.1 (OOF Profiling)**, **Stage 8.2 (Difficulty Manifest)**, dan **Stage 8.4 (Ablation Experiments EXP-0 s.d. EXP-3)** telah terlaksana 100% pada hardware fisik RTX 4060 dengan bukti log nyata dan checksum artefak yang dapat direproduksi.
2. Evaluasi seleksi model pada **Stage 8.5** telah menetapkan **EXP-2 (Targeted Context Augmentation)** sebagai kandidat champion dengan performa tertinggi pada Validation Set.
3. Dataset locked test set (1.282 citra) tetap perawan dan tidak tersentuh (Stage 8.6 belum dieksekusi).
4. Pipeline kini berada pada status **SIAP SEPENUHNYA (TRAINING COMPLETED & VALIDATED)** untuk evaluasi akhir (Stage 8.6) pada locked test set kapan pun pengguna memberikan otorisasi resmi.

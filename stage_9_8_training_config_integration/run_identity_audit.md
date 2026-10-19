# Run Identity and Checkpoint Reuse Audit — Stage 9.8

## 1. Executive Summary

Audit ini memverifikasi arsitektur identitas eksperimen (*run identity*) dan mekanisme verifikasi penggunaan ulang checkpoint (*checkpoint reuse verification*) pada pipeline penelitian YOLO11n dengan Error-Driven Adaptive Optimization (Stage 8.4).

Sebelum Stage 9.8, fungsi `build_experiment_run_identity()` belum mencakup seluruh parameter operasional training (seperti `workers`, `amp`, `plots`, `close_mosaic`, dan `optimizer`). Hal ini membuka celah keamanan reprodusibilitas di mana dua eksekusi dengan hyperparameter berbeda berpotensi dianggap identik dan mengeksekusi *checkpoint reuse* yang salah.

Pada Stage 9.8, **seluruh parameter training efektif telah diikat secara deterministik ke dalam identitas eksperimen**, dilengkapi mekanisme *fail-closed* yang menolak checkpoint reuse apabila terjadi ketidakcocokan parameter sekecil apa pun.

---

## 2. Parameter yang Terikat dalam Run Identity

Fungsi [`build_experiment_run_identity()`](file:///d:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py) menghasilkan kamus identitas run yang mencakup 26 atribut kritis:

| Kategori | Parameter | Tipe | Rationale Pengikatan |
|---|---|---|---|
| **Eksperimen** | `experiment_id` | int (0..3) | Membedakan konfigurasi ablasi EXP-0 hingga EXP-3 |
| | `experiment_name` | str | Nama resmi varian eksperimen |
| | `enable_mixture_sampling` | bool | Saklar mekanisme adaptive mixture sampling |
| | `enable_targeted_aug` | bool | Saklar targeted context crop augmentation |
| **Training Hyperparameters** | `epochs` | int (30) | Batas atas epoch pelatihan |
| | `patience` | int (10) | Ambang batas early stopping |
| | `batch` / `batch_size` | int (16) | Ukuran batch pelatihan |
| | `imgsz` | int (640) | Resolusi spasial citra masukan |
| | `seed` | int (42) | Random seed PyTorch, NumPy, dan Python |
| | `device` | str ("0" / "cpu") | Target akselerator komputasi |
| | `workers` | int (2) | Jumlah worker multiprocessing DataLoader |
| | `amp` | bool (True) | Status FP16 Automatic Mixed Precision |
| | `plots` | bool (False) | Status penyimpanan training plots |
| | `close_mosaic` | int (6) | Nonaktifkan mosaic augmentation pada 6 epoch akhir |
| | `optimizer` | str ("auto") | Algoritma optimasi (AdamW) |
| | `aug_prob` | float (0.50) | Probabilitas aplikasi augmentasi per bounding box |
| **Bobot Pretrained** | `pretrained_weights_path` | str | Path bobot awal (`yolo11n.pt`) |
| | `pretrained_weights_sha256` | str (hex) | Hash SHA-256 bobot awal (`0ebbc8...`) |
| **Integritas Kode & Data** | `script_sha256` | str (hex) | Hash SHA-256 skrip eksekutor ablasi |
| | `data_yaml_sha256` | str (hex) | Hash SHA-256 berkas konfigurasi dataset |
| | `manifest_sha256` | str (hex) | Hash SHA-256 manifest kesulitan (jika berlaku) |
| | `raw_oof_sha256` | str (hex) | Hash SHA-256 prediksi OOF mentah |
| | `clean_oof_run_id` | str | ID run OOF bersih yang memproduksi manifest |
| | `dataset_inventory_fingerprint` | str (hex) | Hash gabungan seluruh citra & label validasi |
| | `total_train_images` | int (8980) | Jumlah citra latih terdaftar |
| | `total_val_images` | int (2567) | Jumlah citra validasi terdaftar |
| **Lingkungan Runtime** | `python_version` | str | Versi interpreter Python |
| | `torch_version` | str | Versi library PyTorch |
| | `cuda_available` | bool | Ketersediaan backend CUDA runtime |
| | `ultralytics_version` | str | Versi package Ultralytics |

---

## 3. Mekanisme Verifikasi Checkpoint Reuse (Fail-Closed)

Fungsi [`verify_ablation_reuse_safety()`](file:///d:/UNDIP/SKRIPSI/stage_8_4_run_ablation_experiments.py) mengimplementasikan verifikasi berbasis *fail-closed*:

1. **Keberadaan Berkas Metadata & Bobot**:
   - Berkas `run_metadata.json` wajib ada. Jika tidak ada, fungsi mengembalikan `(False, "run_metadata.json not found")`.
   - Berkas `best.pt` wajib ada. Jika tidak ada, fungsi mengembalikan `(False, "Checkpoint best.pt does not exist")`.
2. **Pemeriksaan Kelengkapan Field**:
   - Seluruh field dalam `expected_identity` diperiksa keberadaannya pada metadata yang tersimpan.
   - Jika terdapat field yang hilang atau `None`, reuse langsung **DITOLAK** (`(False, "Missing field in run_metadata.json: ...")`).
3. **Pencocokan Nilai Presisi Tinggi**:
   - Nilai float (seperti `aug_prob`) dicocokkan dengan toleransi presisi numerik `math.isclose(a, b, rel_tol=1e-5)`.
   - Nilai integer, string, boolean, dan hash dicocokkan secara strik (`==`).
   - Jika satu field saja tidak cocok, reuse langsung **DITOLAK** dan melaporkan field mana yang mismatch.
4. **Isolasi Metrik Hasil Pelatihan**:
   - Metrik hasil pelatihan (`completed_epochs`, `best_epoch`, `stopped_early`, `map50`, `precision`, `recall`) disimpan terpisah dalam metadata dan **tidak dicampurkan** ke dalam kunci pencocokan konfigurasi. Ini memastikan bahwa perbedaan perilaku konvergensi model yang sah tidak menimbulkan kekeliruan pencocokan.

---

## 4. Pembuktian Uji Regresi (Group B)

Seluruh mekanisme run identity telah diuji secara menyeluruh melalui test suite [`test_stage_9_8_regression.py`](file:///d:/UNDIP/SKRIPSI/test_stage_9_8_regression.py):

| Test Method | Skenario Uji | Hasil |
|---|---|---|
| `test_exact_match_allows_reuse` | Seluruh parameter identitas cocok 100% | **PASS** (Reuse diizinkan) |
| `test_parameter_change_rejects_reuse` | 13 parameter diubah satu per satu (epochs, patience, batch, imgsz, seed, device, workers, amp, plots, close_mosaic, optimizer, aug_prob, pretrained_weights) | **PASS** (13/13 penolakan terverifikasi) |
| `test_missing_identity_field_causes_fail_closed` | Field kritis dihapus dari metadata tersimpan | **PASS** (Penolakan fail-closed) |
| `test_partial_run_rejected_for_reuse` | Eksperimen parsial (summary tanpa best.pt atau metadata) | **PASS** (Penolakan reuse dan penolakan output safety) |
| `test_validation_content_change_rejects_reuse` | Modifikasi 1 byte konten dataset validasi | **PASS** (Fingerprint mismatch menolak reuse) |

---

## 5. Kesimpulan Audit

Arsitektur identitas eksperimen Stage 9.8 telah terbukti secara formal maupun empiris memenuhi standar penelitian *reproducible deep learning*:
- Tidak ada konfigurasi tersembunyi (*zero hidden hyperparameters*).
- Tidak ada risiko penggunaan ulang bobot yang tidak sesuai (*zero stale reuse*).
- Sifat *fail-closed* memastikan integritas riset tetap terjaga tanpa asumsi implisit.

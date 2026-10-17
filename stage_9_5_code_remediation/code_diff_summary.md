# Ringkasan Perubahan Kode (Code Diff Summary) — Stage 9.5

Dokumen ini menyajikan ringkasan perbedaan teknis (code diff summary), alasan perubahan, dan pertimbangan desain untuk seluruh modul yang dimodifikasi selama **Stage 9.5 — Targeted Remediation, Artifact Integrity, and Independent Pipeline Verification**.

Seluruh file asli sebelum perubahan telah dicadangkan secara aman pada direktori:
[`stage_9_5_code_remediation/pre_fix_backup/`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pre_fix_backup/)  
dan inventaris hash SHA-256 pre-fix terdokumentasi pada:
[`stage_9_5_code_remediation/pre_fix_inventory.json`](file:///d:/UNDIP/SKRIPSI/stage_9_5_code_remediation/pre_fix_inventory.json).

---

## 1. `stage_8_1_group_oof_profiling.py` (P0: Checkpoint Reuse Safety & Run Identity)
- **Akar Masalah**:
  - Pemeriksaan `meta_match` pada Stage 9.4 hanya membandingkan sebagian parameter dan mengabaikan `script_sha256`, `data_yaml_sha256`, serta hash daftar citra fold.
  - Jika metadata tidak cocok, kode hanya memberikan peringatan teks tanpa menghentikan eksekusi atau mengisolasi checkpoint lama. Kondisi berikutnya seperti `if not eval_weight_path.exists()` masih berpotensi mengevaluasi checkpoint lama (`best.pt`) yang tidak cocok.
  - Jumlah citra fold di-hardcode ke 8.980 daripada dihitung dari inventaris fisik aktual.
- **Perubahan Utama**:
  1. **Konfigurasi Identitas Run Kanonis (20 Field Eksplisit)**: Mengimplementasikan `build_fold_run_identity()` yang mengikat:
     - `script_sha256`, `data_yaml_sha256`, `fold_yaml_sha256`;
     - `inner_train_sha256`, `inner_val_sha256`, `outer_holdout_sha256`;
     - `dataset_inventory_fingerprint` (jumlah citra, label, dan data YAML SHA256);
     - `fold_id`, `checkpoint_selection_mode`, `epochs`, `batch_size`, `imgsz`, `seed`, `conf_threshold`;
     - `model_cfg`, `hyp_cfg`, `software_versions` (Python, PyTorch, Ultralytics);
     - `selected_checkpoint_sha256`, `predictions_sha256`.
  2. **Verifikasi Reuse Berlapis (`verify_fold_reuse_safety()`)**:
     - Membandingkan seluruh field identitas run secara ketat.
     - Memvalidasi keberadaan fisik checkpoint dan file prediksi.
     - Memvalidasi hash aktual checkpoint dan prediksi terhadap metadata tersimpan.
     - Memvalidasi bahwa jumlah baris prediksi pada outer holdout tepat sama dengan jumlah citra outer holdout.
  3. **Runtime Fail-Fast Anti-Reuse Guard**:
     - Jika checkpoint lama ada tetapi metadata tidak cocok atau verifikasi reuse gagal, fungsi pelatihan langsung melempar `RuntimeError` penolakan, menginstruksikan pengguna untuk menentukan output directory bersih atau menghapus artefak tidak valid secara eksplisit.
  4. **Integritas Split Path**:
     - Menghitung `total_train_images` secara dinamis dari file daftar citra (bukan konstanta hardcoded).
     - Menambahkan validasi integritas path: memastikan file citra fisik benar-benar ada di disk dan tidak ada path yang mengarah ke `cassava-v7-research` saat menggunakan dataset terstandarisasi.
  5. **Dukungan Dua Mode Seleksi Checkpoint**:
     - `nested_inner_val`: Checkpoint dipilih Ultralytics murni berdasarkan Inner-Validation.
     - `fixed_epoch_last`: Checkpoint `last.pt` dipilih tanpa outer holdout.

---

## 2. `stage_8_4_run_ablation_experiments.py` (P1: Fingerprint & Skip-If-Done Integrity)
- **Akar Masalah**:
  - `script_sha256`, `data_yaml_sha256`, dan `manifest_sha256` dihitung tetapi diabaikan dalam keputusan `--skip-if-done`.
  - Checkpoint lama berpotensi digunakan kembali meskipun skrip atau manifest kesulitan berubah.
- **Perubahan Utama**:
  1. **Struktur Identitas Run Kanonis (`build_experiment_run_identity()`)**:
     - Menyatukan `exp_id`, `exp_name`, `epochs`, `batch`, `imgsz`, `seed`, `device`;
     - `script_sha256`, `data_yaml_sha256`, `manifest_sha256`;
     - `pretrained_weight_sha256` (jika file bobot ada di disk);
     - `train_flags` (adaptive sampling flag, context crop flag);
     - `software_versions` (Python, PyTorch, Ultralytics).
  2. **Pengecekan Reuse Komprehensif (`verify_ablation_reuse_safety()`)**:
     - Memeriksa kesesuaian seluruh field identitas run terhadap `run_metadata.json` / `metrics_summary.json`.
     - Memverifikasi keberadaan dan SHA-256 aktual `best.pt`.
     - Jika salah satu fingerprint tidak cocok, reuse ditolak dengan alasan spesifik.
  3. **Anti-Overwrite Protection**:
     - Jika checkpoint lama ada di direktori tujuan tetapi verifikasi identitas run gagal, proses melempar `RuntimeError` dan menolak menimpa atau menggunakan kembali artefak yang tidak cocok tanpa izin pengguna.

---

## 3. `stage_7_baseline_and_error_analysis.py` (P1: Evaluator Unification & Versioning)
- **Akar Masalah**:
  - Menggunakan parser label dan algoritma matching tersendiri yang berbeda dari `research_utils/detection_metrics.py`.
  - Terdapat bug `NameError: free variable 'matched_preds' referenced before assignment`.
  - Tidak memiliki nomor versi evaluator resmi untuk membedakan hasil lama dan baru.
- **Perubahan Utama**:
  1. **Penyatuan Penuh ke SSOT**:
     - Mengimpor dan menggunakan fungsi kanonis dari `research_utils.detection_metrics`: `calculate_iou_xyxy`, `classify_object_size`, `parse_yolo_labels`, dan `greedy_bipartite_matching`.
     - Menggunakan dimensi citra dinamis melalui `get_image_dimensions()`.
  2. **Perbaikan Bug Matching**:
     - Menghapus referensi variabel bebas `matched_preds` yang tidak terdefinisi dan menyelaraskan struktur data matching dengan hasil kembalian `greedy_bipartite_matching`.
  3. **Identitas Versi Evaluator**:
     - Menetapkan `EVALUATOR_VERSION = "Stage-9.5-Canonical-v1"`.
     - Mendokumentasikan secara eksplisit dalam metadata bahwa evaluasi diagnostik error bersifat komplementer dan tidak menggantikan metrik resmi Ultralytics.

---

## 4. `research_utils/detection_metrics.py` (P1: Dynamic Dimensions & Strata Boundary)
- **Akar Masalah**:
  - `parse_yolo_labels()` memiliki default parameter $640 \times 640$.
  - Batas klasifikasi objek Medium sebelumnya menggunakan `<= 96*96`, sehingga area tepat $96^2$ (9.216 piksel) terklasifikasi sebagai Medium, padahal spesifikasi penelitian mendefinisikan Large sebagai area $\ge 96^2$.
- **Perubahan Utama**:
  1. **Deteksi Dimensi Citra Dinamis**:
     - Menambahkan fungsi `get_image_dimensions(img_path)` yang membaca resolusi fisik citra via Pillow (`Image.open().size`).
     - Jika pembacaan gagal, melempar exception atau menggunakan fallback dimensi yang tercatat secara transparan.
  2. **Koreksi Batas Klasifikasi Strata Ukuran Objek**:
     - Small: $\text{area} < 32^2$ ($1.024$ px$^2$).
     - Medium: $32^2 \le \text{area} < 96^2$ ($1.024$ s.d. $9.215$ px$^2$).
     - Large: $\text{area} \ge 96^2$ ($\ge 9.216$ px$^2$).
  3. **Eksplisit Kontrak Satuan Koordinat**:
     - Normalisasi: `[x_center, y_center, width, height]` dalam $[0.0, 1.0]$.
     - Piksel: `[x1, y1, x2, y2]` dalam rentang $[0, W] \times [0, H]$.
     - Luas: piksel kuadrat ($\text{width} \times \text{height}$).

---

## 5. `stage_8_2_build_difficulty_manifest.py` (P1: Dynamic Dimension Manifest)
- **Akar Masalah**:
  - `parse_yolo_labels` dipanggil dengan dimensi default $640 \times 640$ tanpa membaca resolusi citra aktual.
- **Perubahan Utama**:
  1. **Resolusi Dimensi Aktual**:
     - Menggunakan `get_image_dimensions(img_path)` untuk membaca lebar dan tinggi sebenarnya dari setiap citra latih.
  2. **Pengayaan Kolom Manifest**:
     - Menyimpan `norm_xyxy`, `img_width`, dan `img_height` ke dalam `object_difficulty_manifest.csv` dan `image_difficulty_manifest.csv`.
  3. **Penanganan Kasus Ekstrem Matematis**:
     - Memvalidasi formula mixture sampling dengan verifikasi $\sum D > 0$. Jika $\sum D = 0$, melakukan fallback deterministik matematis ke distribusi uniform $1/N$ tanpa menghasilkan nilai NaN atau pembagian dengan nol.

---

## 6. `stage_8_3_adaptive_pipeline.py` (P1: Lazy Image Loading & DataLoader Integration)
- **Akar Masalah**:
  - Pada pipeline training nyata Ultralytics, `get_image_and_label()` hanya memuat label ke dalam memori secara malas (lazy loading); `label["img"]` bernilai `None` sampai `load_image()` dipanggil. Akibatnya, `apply_context_crop()` dilewati (silent skip) jika dipanggil sebelum gambar di-cache.
  - `AdaptiveYOLODataset` menerima argumen trainer `enable_mixture_sampling` yang menyebabkan `TypeError` jika diteruskan ke `BaseDataset.__init__()`.
  - Format string `target_small_box` yang dipisahkan koma dalam CSV rentan terpecah kolom jika tidak dikutip dengan tanda kutip ganda.
- **Perubahan Utama**:
  1. **Lazy Image Loading Fallback**:
     - Di dalam `apply_context_crop()`, jika `label.get("img") is None` tetapi `label.get("im_file")` ada, gambar dimuat secara aman dari disk melalui `cv2.imread(label["im_file"])`.
  2. **Pembersihan Parameter Inisialisasi**:
     - Mengekstrak dan menghapus (`pop`) argumen khusus `enable_mixture_sampling` sebelum memanggil `super().__init__()`.
  3. **Robust Box Parsing**:
     - Menangani baik koordinat piksel maupun koordinat normalisasi dengan parsing tipe yang aman dan penanganan tanda kurung/kutip CSV.

---

## 7. `stage_8_5_model_selection_and_validation.py` & `stage_8_6_final_test_benchmark.py`
- **Akar Masalah**:
  - Pemanggilan `parse_yolo_labels` masih berpotensi memakai dimensi default 640 jika citra memiliki rasio non-standar.
- **Perubahan Utama**:
  - Menyelaraskan pembacaan ground truth dengan resolusi citra aktual melalui `get_image_dimensions()`.
  - Mempertahankan guard konfirmasi `--confirm-final-evaluation` pada Stage 8.6 untuk mencegah unconfirmed test snooping.

---

## 8. `execute_phase_b_standardization.py` (P1: Non-Destructive Verify-Only & Geometric Equivalence)
- **Akar Masalah**:
  - Penjepitan koordinat poligon dilakukan secara diam-diam tanpa mencatat peringatan jika titik berada di luar batas valid.
  - Mode verifikasi sebelumnya hanya membandingkan jumlah baris anotasi tanpa menguji kesetaraan geometri bounding box.
- **Perubahan Utama**:
  1. **Penolakan Koordinat di Luar Rentang**:
     - Menghapus silent clamping; koordinat di luar rentang valid $[0.0, 1.0]$ dilaporkan sebagai invalid.
  2. **Pengujian Geometri Ketat (Toleransi $\le 10^{-4}$)**:
     - Membandingkan koordinat hasil konversi `(xc, yc, w, h)` terhadap hasil transformasi poligon referensi dengan toleransi numerik $10^{-4}$.
  3. **Mode `--verify-only` Menyeluruh**:
     - Memeriksa paritas pasangan citra-label, mendeteksi citra/label orphan, memeriksa kesamaan ukuran file/checksum, dan memastikan zero data loss tanpa menulis atau memodifikasi file pada disk.

---

## 9. `run_stage_8_profiling.py` (P1: Standardized Dataset Single Source of Truth)
- **Akar Masalah**:
  - Argumen default `--data-yaml` mengarah ke `cassava-v7-research/data.yaml` dan alur `--dry-run` mencoba melanjutkan ke pembuatan manifest Stage 8.2 meskipun prediksi OOF belum dihasilkan.
- **Perubahan Utama**:
  1. **Pembaruan Default YAML**: Mengubah default `--data-yaml` ke `cassava-v7-standardized/data.yaml`.
  2. **Penyempurnaan Alur Dry-Run**: Menghentikan eksekusi secara bersih setelah pembuatan split OOF jika OOF prediction cache belum tersedia, mencegah kegagalan cascade pada Stage 8.2.

---

## 10. `test_stage_9_5_regression.py` (Regression Suite Baru)
- **Penambahan**:
  - 48 test cases baru yang menguji seluruh jalur kegagalan:
    - Penolakan reuse saat metadata hilang, checkpoint hilang, script hash berubah, YAML berubah, inner train/val berubah, outer holdout berubah, atau checkpoint mismatch.
    - Penolakan ablasi skip saat script, yaml, manifest, bobot awal, atau hyperparameter berubah.
    - Unifikasi evaluator Stage 7 vs Stage 8.
    - Dimensi dinamis pada citra non-square (1920x1080).
    - Verifikasi geometri standardisasi dan deteksi label orphan.
    - Integrasi augmentasi crop dengan dataloader Ultralytics pada mini dataset sintetis.
    - Validasi formula mixture sampling dan penanganan fallback matematis $\sum D = 0$.

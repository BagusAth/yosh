# Ringkasan Perubahan Kode (Code Diff Summary) — Stage 9.4

Laporan ini menyajikan ringkasan diff teknis untuk setiap file yang dimodifikasi selama audit remediasi Stage 9.4. Seluruh salinan asli sebelum perubahan telah tersimpan aman di `stage_9_4_code_remediation/pre_fix_backup/`.

---

## 1. `research_utils/detection_metrics.py` (MODUL BARU)
- **Tujuan**: Membangun Single Source of Truth (SSOT) evaluasi untuk seluruh tahapan penelitian.
- **Fitur Utama**:
  - `calculate_iou_xyxy`: Perhitungan IoU format `[x1, y1, x2, y2]`.
  - `classify_object_size`: Partisi skala Small ($< 32^2$), Medium ($32^2 - 96^2$), dan Large ($> 96^2$).
  - `parse_yolo_labels`: Parser koordinat adaptif dengan penskalaan dinamis `img_width` dan `img_height`.
  - `greedy_bipartite_matching`: Pencocokan serakah satu-ke-satu dengan pengurutan IoU kandidat menurun secara global (zero prediction reuse, zero GT reuse).
  - `calculate_object_difficulty`: Formula kanonis $D_i$ (+0.35 confusion penalty, FN = $0.80 + 0.20 S_i$).
  - `calculate_image_difficulty`: Formula kanonis $D_\text{image} = 0.5 \max(D_i) + 0.5 \text{mean}(D_i)$ atau $\max(C_\text{FP})$.
  - `calculate_mixture_weights`: Formula sampling campuran $P(k) = (1 - \lambda)/N + \lambda (D_k / \sum D)$ dengan $\lambda = 0.70$.
  - `select_best_model`: Hierarki seleksi model terkunci (Primary: mAP@50-95 $\rightarrow$ Tie-1: Recall Small $\rightarrow$ Tie-2: mAP@50 $\rightarrow$ Tie-3: min FN).
  - `extract_canonical_metrics`: Pemisahan kanonis `box.ap50` (AP@50) dan `box.maps` (AP@50-95) dengan presisi penuh.

---

## 2. `stage_8_1_group_oof_profiling.py`
- **Akar Masalah**: Fold holdout $k$ dimasukkan ke `val:` dalam `fold_k_data.yaml`. Checkpoint `best.pt` dipilih menggunakan metrik fold holdout tersebut sebelum inferensi holdout dilakukan, menyebabkan kebocoran label fold holdout ke pemilihan model probe.
- **Perubahan Utama**:
  - Mengubah fungsi `generate_fold_configs` menjadi `generate_nested_fold_configs`.
  - Mengimplementasikan mode `nested_inner_val`: Outer Holdout fold $k$ (1/4 data latih) **100% diisolasi** dari konfigurasi YAML pelatihan probe.
  - Tiga fold sisanya dibagi secara group-aware menjadi Inner-Train (2 fold) dan Inner-Validation (1 fold).
  - Inner-Validation digunakan oleh Ultralytics untuk memilih `best.pt`.
  - Checkpoint tersebut digunakan secara eksklusif untuk memprediksi Outer Holdout fold $k$.
  - Menambahkan argumen CLI `--data-yaml`, `--dataset-dir`, dan `--checkpoint-selection-mode`.
  - Menambahkan verifikasi metadata checksum pada flag `--skip-training-if-done` untuk mencegah penggunaan artefak yang tidak cocok secara diam-diam.
  - Menyimpan manifest `nested_fold_assignments.csv`.

---

## 3. `stage_8_2_build_difficulty_manifest.py`
- **Akar Masalah**: Evaluator mendefinisikan algoritma matching lokal yang tidak terpadu dan mengasumsikan resolusi tetap $640 \times 640$.
- **Perubahan Utama**:
  - Mengimpor `greedy_bipartite_matching`, `calculate_object_difficulty`, `calculate_image_difficulty`, dan `calculate_mixture_weights` dari `research_utils.detection_metrics`.
  - Menghubungkan direktori data latih secara dinamis melalui argumen `--data-yaml` (SSOT).
  - Menggunakan parser dimensi dinamis untuk membaca ground truth.
  - Memverifikasi jumlah citra latih ($8.980$ citra) dan konsistensi ID prediksi secara fail-fast.

---

## 4. `stage_8_3_adaptive_pipeline.py`
- **Akar Masalah**: Jika terdapat citra latih yang tidak tercakup dalam manifest kesulitan, `get_sampling_weights` melakukan fallback diam-diam ke probabilitas uniform $1/N$ tanpa memperingatkan pengguna. Resolusi output crop juga di-hardcode ke 640.
- **Perubahan Utama**:
  - Menambahkan validasi `validate_manifest_coverage()` yang memicu `ValueError` langsung jika ada citra yang hilang dalam manifest (Zero Silent Fallback).
  - Menambahkan validasi integritas manifest pada `load_difficulty_manifest()` (mendeteksi ID duplikat atau nilai di luar batas).
  - Memperbarui `apply_context_crop()` untuk membaca target `imgsz` secara dinamis dari konfigurasi `hyp`.
  - Melindungi kamus label input (`deepcopy`) agar tidak mengalami mutasi in-place saat proses transformasi augmentasi.
  - Menambahkan fallback aman pada pembacaan atribut `hyp` untuk dataset tiruan (mock datasets).

---

## 5. `stage_8_4_run_ablation_experiments.py`
- **Akar Masalah**: Script default menulis ke direktori historis `stage_8_ablation/` yang berisiko menimpa hasil lama. Pengambilan metrik validation belum menggunakan helper kanonis.
- **Perubahan Utama**:
  - Mengalihkan default `--output-dir` ke direktori baru `stage_10_11_clean_experiments/`.
  - Menambahkan proteksi `PermissionError` yang memblokir penulisan ke `stage_8_ablation/` kecuali flag eksplisit `--allow-historic-overwrite` diberikan.
  - Mengintegrasikan `extract_canonical_metrics()` dari `research_utils.detection_metrics` untuk serialisasi metrik AP kanonis.
  - Menambahkan pencatatan metadata eksekusi (checksum kode, data YAML, manifest, seed, batch, epochs) ke dalam `metrics_summary.json` untuk memvalidasi penggunaan flag `--skip-if-done`.

---

## 6. `stage_8_5_model_selection_and_validation.py`
- **Akar Masalah**: Path citra dan label validasi di-hardcode secara literal ke `d:/UNDIP/SKRIPSI/cassava-v7-research/valid/images` mengabaikan argumen `--data-yaml`. Urutan tie-break di kode mendahulukan mAP@50 daripada Recall Small.
- **Perubahan Utama**:
  - Menambahkan fungsi `resolve_val_paths()` untuk menentukan path citra dan label validasi secara dinamis dari `data.yaml`.
  - Menerapkan fungsi `select_best_model()` dari `research_utils.detection_metrics` dengan urutan hierarki terkunci:
    1. Kriteria Utama: validation mAP@50-95 tertinggi.
    2. Tie-break 1: Recall Small tertinggi (hanya jika $|\Delta \text{mAP@50-95}| \le 10^{-5}$).
    3. Tie-break 2: validation mAP@50 tertinggi.
    4. Tie-break 3: Total False Negative (`fn_missed`) terendah.
  - Menghapus margin arbitrer $0.5\%$.
  - Mengintegrasikan `greedy_bipartite_matching()` global untuk dekomposisi error validasi.

---

## 7. `stage_8_6_final_test_benchmark.py`
- **Akar Masalah**: Path test set di-hardcode ke direktori research lama, `recall_small` tidak dilaporkan, dan script dapat dieksekusi tanpa konfirmasi selama masa audit.
- **Perubahan Utama**:
  - Menambahkan fungsi `resolve_test_paths()` untuk menurunkan path test set langsung dari `data.yaml`.
  - Menambahkan safety guard `--confirm-final-evaluation` yang menghentikan eksekusi secara aman jika tidak ada konfirmasi eksplisit, melindungi isolasi locked test set.
  - Menambahkan penghitungan metrik berstrata ukuran (Small, Medium, Large) lengkap dengan jumlah GT, TP, dan recall.
  - Mengintegrasikan `greedy_bipartite_matching()` dan `extract_canonical_metrics()`.

---

## 8. `run_stage_8_profiling.py`
- **Akar Masalah**: Parameter CLI tidak diteruskan secara fleksibel ke Stage 8.1 dan 8.2, dan tidak mendukung dry-run.
- **Perubahan Utama**:
  - Menambahkan parser argumen CLI (`--data-yaml`, `--output-dir-81`, `--output-dir-82`, `--checkpoint-selection-mode`, `--dry-run`).
  - Meneruskan seluruh konfigurasi secara transparan ke subprocess `stage_8_1_group_oof_profiling.py` dan `stage_8_2_build_difficulty_manifest.py`.
  - Menghentikan proses secara langsung jika terdapat tahapan yang menghasilkan returncode non-zero.

---

## 9. `execute_phase_b_standardization.py`
- **Akar Masalah**: Script tidak memiliki argumen CLI, tidak memiliki mode inspeksi non-destruktif, dan parsing baris berpotensi melewatkan baris malformed secara diam-diam.
- **Perubahan Utama**:
  - Menambahkan argumen CLI `--src-root`, `--dst-root`, `--verify-only`, dan `--force-overwrite`.
  - Menambahkan fungsi validasi ketat `validate_standardized_label_file()` yang memastikan setiap baris memiliki tepat 5 kolom (`class_id xc yc w h`).
  - Memverifikasi rentang class ID $[0, 4]$, koordinat ternormalisasi, dan dimensi positif $w > 0, h > 0$.
  - Memastikan poligon memiliki jumlah koordinat genap $\ge 6$.
  - Mencegah penulisan ulang hardlink jika direktori tujuan sudah ada dan terverifikasi.

---

## 10. `test_stage_9_3_regression.py`
- **Akar Masalah**: Hardcoded path `d:/UNDIP/SKRIPSI` menghambat portabilitas workspace.
- **Perubahan Utama**:
  - Mengubah path resolusi menjadi `WORKSPACE_DIR = Path(__file__).resolve().parent`.
  - Memverifikasi kelulusan backward compatibility (16/16 tests passed).

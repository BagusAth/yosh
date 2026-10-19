# RINGKASAN PERUBAHAN KODE (DIFF SUMMARY) — STAGE 9.7

**Baseline Commit**: `6e92427d70a93c45f0d3f5cc0853d65753a48ea7` (HEAD Stage 9.6)  
**Branch Kerja**: `stage-9-7-readiness-gate-hardening`  
**Statistik Perubahan**: 8 file diubah, +616 baris ditambahkan, -122 baris dihapus.  

---

## 1. Tabel Ringkasan Perubahan per File

| File | Status | Tambah | Hapus | Deskripsi Remediasi Utama |
| :--- | :---: | :---: | :---: | :--- |
| `research_utils/detection_metrics.py` | Modified | +383 | -31 | Penutupan celah nested path di `validate_output_directory_safety()`, dukungan `is_root`, validasi `validate_yolo_dataset_integrity()` dengan background images, hashing byte citra di `compute_dataset_inventory_fingerprint()`, dan dukungan layout list di `resolve_data_yaml_split()`. |
| `stage_8_4_run_ablation_experiments.py` | Modified | +254 | -74 | Pemisahan root check dari child check (`is_root=True`), penghapusan false collision EXP-0 -> EXP-1, penambahan schema provenansi v2.0 dengan verifikasi raw OOF SHA, dan dukungan dependency injection (`trainer_cls`, `model_eval_fn`). |
| `stage_8_1_group_oof_profiling.py` | Modified | +46 | -14 | Resolusi canonical train images via `resolve_train_paths()`, perbaikan variabel `images_to_load` di `load_annotations()`, dan validasi safety direktori per-fold (`is_root=False`). |
| `stage_8_2_build_difficulty_manifest.py` | Modified | +38 | -13 | Emisi sidecar JSON berversi (`provenance_schema_version: 2.0`, `evaluator_version: Stage-9.7-Canonical-v2`, `raw_oof_sha256`), dan iterasi dataset latih berbasis SSOT. |
| `stage_8_3_adaptive_pipeline.py` | Modified | +2 | -1 | Fallback `target_imgsz` ke `self.imgsz` di `apply_context_crop()` ketika `hyp.imgsz` tidak didefinisikan, menjamin konsistensi resolusi tensor antar item batch. |
| `stage_7_baseline_and_error_analysis.py` | Modified | +5 | -2 | Standardisasi derivasi label path menggunakan helper kanonis `derive_yolo_label_path()`. |
| `stage_8_5_model_selection_and_validation.py` | Modified | +5 | -2 | Standardisasi derivasi label path menggunakan helper kanonis `derive_yolo_label_path()`. |
| `stage_8_6_final_test_benchmark.py` | Modified | +5 | -2 | Standardisasi derivasi label path menggunakan helper kanonis `derive_yolo_label_path()`. |

---

## 2. Rincian Teknis Perubahan per Modul

### 2.1. `research_utils/detection_metrics.py`
- **Output Safety Normalization**: Mengganti perbandingan leaf folder dengan inspeksi rekursif komponen `parts` dan `p.is_relative_to(resolved_forb)`.
- **Parameter `is_root`**: Menambahkan parameter `is_root=False` pada `validate_output_directory_safety()`. Ketika `is_root=True` dan `allow_reuse=True`, fungsi mengizinkan reuse direktori root tanpa menuntut `expected_identity` tunggal.
- **YOLO Dataset Integrity**: Mengimplementasikan `validate_yolo_dataset_integrity()` yang memvalidasi pasangan citra-label 1:1, menolak file orphan, memvalidasi rentang koordinat `[0.0, 1.0]`, dan memperlakukan file label 0-byte sebagai citra background yang valid.
- **Dataset Content Fingerprint**: Mengimplementasikan full SHA-256 byte hashing untuk seluruh citra dan label, serta mengikutsertakan konten split validasi ketika `data_yaml_path` diberikan.
- **YAML Split Resolver**: Memperluas `resolve_data_yaml_split()` untuk mendukung `split_entry` bertipe `list`, mengekstraksi seluruh direktori dan file list `.txt` secara teragregasi.

### 2.2. `stage_8_4_run_ablation_experiments.py`
- **Orkestrator vs Child Isolation**: `run_ablation_pipeline()` memanggil `validate_output_directory_safety(output_dir, is_root=True)` satu kali di tingkat root. Fungsi `run_single_experiment()` hanya memvalidasi `exp_dir` child (`is_root=False`).
- **Verifikasi Manifest Provenance 2.0**: Memvalidasi versi skema, `clean_oof_run_id` (bebas dari direktori historis), `raw_oof_sha256` cocok dengan file mentah aktual, dan 100% baris manifest memiliki `d_image` dalam [0, 1], `p_sampling > 0`, serta `sum(p_sampling) ~ 1.0`.
- **Isolasi EXP-0**: Memastikan EXP-0 strictly menetapkan `manifest_to_use = None` dan mengabaikan argumen `--manifest-path`.
- **Dependency Injection**: Parameter `trainer_cls` dan `model_eval_fn` diekspos di `run_ablation_pipeline()` dan `run_single_experiment()` untuk memungkinkan mock testing tanpa full training.

### 2.3. `stage_8_1_group_oof_profiling.py`
- Mengganti pembacaan filesystem non-kanonis dengan SSOT `resolve_train_paths()`.
- Memperbaiki bug variabel `actual_count = len(image_files)` menjadi `len(images_to_load)`.
- Menambahkan pemeriksaan keselamatan direktori output per-fold sebelum training atau inferensi.

### 2.4. `stage_8_2_build_difficulty_manifest.py`
- Mengikutsertakan metadata provenansi lengkap ke dalam `train_image_difficulty_manifest.provenance.json`: skema 2.0, hash raw OOF, evaluasi versi kanonis, dan dataset fingerprint.

### 2.5. `stage_8_3_adaptive_pipeline.py`
- Memperbaiki penentuan `target_imgsz` di `apply_context_crop()` agar menghormati `self.imgsz` dari dataset ketika `hyp.imgsz` kosong.

### 2.6. `stage_7`, `stage_8_5`, `stage_8_6`
- Menggantikan logika manual manipulasi string path label dengan `derive_yolo_label_path()`.

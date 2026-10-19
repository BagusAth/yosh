# AUDIT IDENTITAS EKSPERIMEN, PROVENANSI MANIFEST & REUSE SAFETY

**Tahap Riset**: STAGE 9.7 — Pipeline Safety, Provenance & End-to-End Readiness Remediation  
**Status Audit**: PASS (100% Provenance & Checksum Enforced)  
**Komponen Utama**:
- `stage_8_1_group_oof_profiling.py::build_fold_run_identity()`, `verify_fold_reuse_safety()`
- `stage_8_2_build_difficulty_manifest.py::provenance_data`
- `stage_8_4_run_ablation_experiments.py::build_experiment_run_identity()`, `verify_manifest_provenance_and_coverage()`, `verify_ablation_reuse_safety()`

---

## 1. Definisi Skema Identitas Eksperimen Kanonis

Untuk mencegah kontaminasi dan menjamin reprodusibilitas 100%, seluruh tahapan eksperimen mewajibkan pencatatan identitas run (*run identity*) yang mencakup parameter operasional dan checksum kriptografis.

### 1.1. Skema Identitas Stage 8.1 (Per-Fold OOF Run Identity)

Tersimpan dalam `fold_<id>/fold_metadata.json`:

```json
{
  "script_sha256": "64-character SHA-256 dari stage_8_1_group_oof_profiling.py",
  "data_yaml_path": "Path absolut ke data.yaml",
  "data_yaml_sha256": "64-character SHA-256 dari data.yaml",
  "inner_train_txt_sha256": "SHA-256 dari file list citra inner training",
  "inner_val_txt_sha256": "SHA-256 dari file list citra inner validation",
  "outer_holdout_txt_sha256": "SHA-256 dari file list citra outer holdout",
  "dataset_inventory_fingerprint": "64-character deterministic inventory fingerprint",
  "checkpoint_selection_mode": "nested_inner_val",
  "fold_id": 1,
  "epochs": 12,
  "batch_size": 16,
  "imgsz": 640,
  "seed": 42,
  "conf_thresh": 0.001,
  "model_variant": "yolo11n.pt",
  "pretrained_weights_sha256": "0ebbc80d... (SHA-256 bobot awal yolo11n.pt)",
  "checkpoint_sha256": "SHA-256 aktual dari best.pt yang dihasilkan",
  "pred_csv_sha256": "SHA-256 aktual dari predictions.csv",
  "holdout_image_count": 2246,
  "inner_train_count": 4489,
  "inner_val_count": 2245,
  "evaluated_images": 2246,
  "num_predictions": 15420
}
```

### 1.2. Skema Provenansi Stage 8.2 (Difficulty Manifest Provenance v2.0)

Tersimpan dalam `train_image_difficulty_manifest.provenance.json`:

```json
{
  "provenance_schema_version": "2.0",
  "evaluator_version": "Stage-9.7-Canonical-v2",
  "manifest_file": "train_image_difficulty_manifest.csv",
  "manifest_csv_sha256": "SHA-256 dari file manifest CSV",
  "raw_oof_path": "Path absolut ke file oof_predictions_raw.csv",
  "raw_oof_sha256": "64-character SHA-256 dari oof_predictions_raw.csv",
  "clean_oof_run_id": "Path atau ID dari clean Stage 8.1 run",
  "dataset_inventory_fingerprint": "Fingerprint inventaris dataset latih",
  "data_yaml_path": "Path data.yaml kanonis",
  "data_yaml_sha256": "SHA-256 data.yaml",
  "iou_match_thresh": 0.40,
  "confusion_penalty": 0.35,
  "mixture_lambda": 0.70,
  "script_sha256": "SHA-256 dari stage_8_2_build_difficulty_manifest.py",
  "image_count": 8980,
  "unique_image_count": 8980
}
```

### 1.3. Skema Identitas Stage 8.4 (Ablation Experiment Run Identity)

Tersimpan dalam `<exp_name>/metrics_summary.json` di bawah key `run_metadata`:

```json
{
  "exp_id": "EXP-1",
  "spec_name": "exp_1_adaptive_sampling",
  "title": "Mixture Adaptive Sampling",
  "script_sha256": "SHA-256 dari stage_8_4_run_ablation_experiments.py",
  "data_yaml_path": "Path data.yaml",
  "data_yaml_sha256": "SHA-256 data.yaml",
  "manifest_path": "Path manifest jika digunakan (EXP-1, EXP-2, EXP-3; None untuk EXP-0)",
  "manifest_sha256": "SHA-256 manifest CSV jika digunakan",
  "dataset_inventory_fingerprint": "Fingerprint inventaris dataset latih & validasi",
  "pretrained_weights_sha256": "SHA-256 dari yolo11n.pt",
  "epochs": 50,
  "batch": 16,
  "imgsz": 640,
  "seed": 42,
  "device": "0",
  "enable_mixture_sampling": true,
  "enable_targeted_aug": false,
  "best_weight_sha256": "SHA-256 aktual dari best.pt hasil training"
}
```

---

## 2. Mekanisme Verifikasi Integritas & Proteksi Reuse

Fungsi verifikasi (`verify_fold_reuse_safety` dan `verify_ablation_reuse_safety`) menerapkan aturan ketat:
1. **Pemeriksaan Eksistensi Fisik**: Checkpoint `best.pt`, file prediksi, dan file metadata JSON harus ada di disk.
2. **Kecocokan Identitas Konfigurasi**: Nilai `epochs`, `batch`, `imgsz`, `seed`, `device`, toggles strategi sampling dan augmentasi harus identik.
3. **Kecocokan Kode & Lingkungan**: Checksum `script_sha256`, `data_yaml_sha256`, `dataset_inventory_fingerprint`, dan `pretrained_weights_sha256` wajib sama persis.
4. **Kecocokan Bobot Model**: SHA-256 aktual dari checkpoint `best.pt` di disk dihitung ulang secara mandiri dan dicocokkan dengan `best_weight_sha256` di metadata.

Jika satu karakter pun menyimpang, reuse otomatis dibatalkan, training ulang dijalankan, atau eksekusi dihentikan (FAIL-FAST) jika penimpaan dilarang.

---

## 3. Matriks Pembuktian Pengujian (Group B & Group C)

| ID Tes | Parameter yang Dimodifikasi | Respon Sistem | Status |
| :--- | :--- | :--- | :---: |
| `test_identity_exact_match_passes` | Seluruh parameter cocok | Valid reuse diizinkan | PASS |
| `test_identity_detects_script_modification` | Byte skrip diubah | Ditolak: `script_sha256 mismatch` | PASS |
| `test_identity_detects_data_yaml_modification` | Byte `data.yaml` diubah | Ditolak: `data_yaml_sha256 mismatch` | PASS |
| `test_identity_detects_pretrained_weights_modification` | Checksum bobot awal diubah | Ditolak: `pretrained_weights_sha256 mismatch` | PASS |
| `test_valid_manifest_and_provenance_passes` | Manifest + sidecar valid 100% | Lolos validasi cakupan 100% | PASS |
| `test_manifest_tampered_csv_rejected` | 1 byte di manifest CSV diubah | Ditolak: `Manifest CSV SHA-256 mismatch` | PASS |
| `test_manifest_tampered_raw_oof_rejected` | Isi raw OOF prediksi diubah | Ditolak: `Raw OOF predictions file hash mismatch` | PASS |
| `test_manifest_historical_oof_id_rejected` | Path/ID OOF menunjuk ke folder historis | Ditolak: `Provenance references historical unverified OOF` | PASS |
| `test_manifest_missing_or_extra_coverage_rejected` | Manifest hanya mencakup 3 dari 4 citra | Ditolak: `Missing image_ids in manifest` | PASS |
| `test_manifest_duplicate_image_ids_rejected` | Terdapat baris ID duplikat di CSV | Ditolak: `Duplicate image_id found in manifest` | PASS |
| `test_manifest_invalid_sampling_probs_rejected` | Probabilitas sampling bernilai <= 0 | Ditolak: `non-positive p_sampling found` | PASS |

---

## 4. Kesimpulan

Rantai provenansi eksperimen dari Stage 8.1 (OOF profiling) -> Stage 8.2 (Difficulty manifest) -> Stage 8.4 (Ablation experiments) terikat secara kriptografis menggunakan SHA-256 berversi, mendeteksi setiap bentuk manipulasi artefak, dan mencegah penggunaan data historis yang tidak tervalidasi.

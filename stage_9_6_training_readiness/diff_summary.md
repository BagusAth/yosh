# Stage 9.6 Code Diff & Remediation Summary

## 1. Overview
Stage 9.6 enforces strict pipeline hardening, complete isolation of all new execution outputs from historical runs, cryptographic content fingerprinting, real-path targeted context augmentation verification, single-source-of-truth dataset path resolution, and definitive training readiness verification without executing any actual training or evaluating the locked test set.

---

## 2. File-by-File Detailed Changes

### `research_utils/detection_metrics.py`
- **Output Safety Guard**: Added `validate_output_directory_safety(output_dir, stage_name, allow_reuse=False, expected_identity=None, metadata_filename=None, is_dry_run=False)`:
  - Blocks any attempt to write into forbidden historical directories: `stage_8_1_oof`, `stage_8_2_difficulty`, `stage_8_ablation`, `cassava-v7-stage-7`.
  - Rejects directories containing unverified partial artifacts.
  - Requires 100% parameter and checksum match for `--skip-if-done` reuse.
- **Pretrained Weights SHA-256**: Added `compute_pretrained_weights_sha256(weights_path)` to cryptographically track base model weights (`yolo11n.pt`).
- **Full-Content Dataset Fingerprinting**: Upgraded `compute_dataset_inventory_fingerprint(images_dir, labels_dir, data_yaml_path)` to compute SHA-256 hashes of every individual image and label file, verifying image/label pairing completeness, total objects, class distributions, and `data.yaml` hash.
- **Single Source of Truth Split Resolution**: Added `resolve_data_yaml_split(data_yaml_path, split, allow_empty=True)` returning `DataYamlSplitResult`:
  - Supports 3-tuple unpacking (`images_dir, labels_dir, images = resolve_data_yaml_split(...)`) and dictionary access.
  - Handles relative paths, absolute paths, and `.txt` file image lists.

### `execute_phase_b_standardization.py`
- **Content-Aware SHA-256 Image Verification**: In `verify_phase_b_standardization()`, added per-image SHA-256 comparison (`sha256_file(im) != sha256_file(dst_im)`). Catches any bit/byte modification even when file sizes are identical.

### `stage_8_3_adaptive_pipeline.py`
- **Targeted Context Augmentation Fixes**:
  - Replaced buggy `self.format_class(...)` in `__init__` with imported `Format(...)` from `ultralytics.data.augment`.
  - Passed `segments=np.zeros((0, 1000, 2), dtype=np.float32)` to `Instances` constructor to satisfy `RandomFlip.fliplr`'s slicing contract, eliminating `TypeError: 'NoneType' object is not subscriptable`.
  - Popped `ratio_pad` (`out_label.pop("ratio_pad", None)` and `cropped_label.pop("ratio_pad", None)`) to ensure uniform dictionary keys across cropped and uncropped samples, preventing `IndexError: tuple index out of range` in `collate_fn`.
  - Ensured original label dictionary is protected from in-place mutation using `deepcopy(label)`.
  - Added properties `crop_attempts` and `crops_applied` on `AdaptiveYOLODataset`.

### `stage_8_4_run_ablation_experiments.py`
- **Manifest Provenance & 100% Coverage Verification**: Added `verify_manifest_provenance_and_coverage`:
  - Enforces explicit `--manifest-path` for EXP-1, EXP-2, EXP-3 (default set to `None`; no silent fallback to historical directories).
  - EXP-0 does not require manifest.
  - Verifies existence of sidecar `train_image_difficulty_manifest.provenance.json`.
  - Compares CSV hash, dataset inventory fingerprint, and source `data.yaml` SHA-256.
  - Rejects manifests referencing historical unverified OOF runs.
  - Verifies 100% image coverage against train split with zero duplicates, missing images, extra images, or invalid rows.
- **Safe Dry-Run & Output Safety**:
  - Added `--dry-run` flag to validate preflight, dataset fingerprint, and manifest without launching trainer.
  - Default `--output-dir` updated to `stage_8_ablation_clean`.
  - Added partial output detection (e.g. `metrics_summary.json` without `best.pt` or vice versa).
  - Eliminated `try/except: pass` in dataset fingerprint computation.

### `stage_8_1_group_oof_profiling.py`
- **Clean Default Output & Safety**: Default `--output-dir` updated to `stage_8_1_oof_clean`. Added `validate_output_directory_safety` before writing partition/config files.
- **Single Source of Truth Split Resolution**: Integrated `resolve_data_yaml_split` for train split path resolution.
- **Pretrained Weights Tracking**: Integrated `pretrained_weights_sha256` into `build_fold_run_identity` and `train_and_predict_fold`.
- **Dynamic Image Count**: Added `--expected-train-images` (default 8,980) and dynamic count verification in `load_annotations`.

### `stage_8_2_build_difficulty_manifest.py`
- **Clean Default Output**: Default `--output-dir` updated to `stage_8_2_difficulty_clean` and default `--oof-dir` to `stage_8_1_oof_clean`.
- **Sidecar Provenance Generation**: Generates `train_image_difficulty_manifest.provenance.json` containing CSV SHA-256, dataset fingerprint, `data.yaml` SHA-256, raw OOF SHA-256, clean OOF run ID, matching thresholds, and script hash.
- **Safety Guard**: Integrated `validate_output_directory_safety` before writing manifest files.

### `stage_7_baseline_and_error_analysis.py`
- **Clean Defaults & SSOT**: Default `--data-yaml` updated to standardized dataset, default `--output-dir` to `stage_7_baseline_clean`. Integrated `resolve_data_yaml_split` for Pilar 3/4 validation images.
- **Strict Reuse & Dry-Run**: Added `--skip-if-done` and `--dry-run`. Checkpoint reuse strictly verified against dataset fingerprint, script hash, pretrained weights, and `data.yaml` SHA-256 via `stage_7_run_metadata.json`.

### `stage_8_5_model_selection_and_validation.py` & `stage_8_6_final_test_benchmark.py`
- **Clean Default Ablation Dir**: Updated defaults to `stage_8_ablation_clean`.
- **Single Source of Truth**: Integrated `resolve_data_yaml_split` for `val` and `test` splits.
- **Backward Compatibility**: Added `resolve_val_paths` and `resolve_test_paths` wrappers and flexible `list[Path] | Path` handling to maintain full compatibility with existing test suites.

### `run_stage_8_profiling.py`
- **Clean Defaults**: Default `--output-dir-81` updated to `stage_8_1_oof_clean` and `--output-dir-82` to `stage_8_2_difficulty_clean`.

### `test_stage_9_6_regression.py`
- Comprehensive new regression test suite covering:
  - Output directory safety (empty succeeds, historical rejected, partial rejected, metadata mismatch rejected, valid reuse passes, dry-run does not mutate disk).
  - Manifest provenance & coverage (missing manifest fails, missing sidecar fails, tamper fails, historical OOF fails, fingerprint mismatch fails, duplicate ID fails, incomplete coverage fails, extra images fail, valid passes).
  - Real-path targeted context augmentation on `AdaptiveYOLODataset.__getitem__()` with `augment=True` (verifying `crop_attempts` and `crops_applied`, valid normalized bboxes, 640x640 shape, and mixed batch `collate_fn` success).
  - Content-aware dataset fingerprinting (detects byte change with identical size, detects label modification).
  - Single Source of Truth dataset path resolution (relative, absolute, .txt image list, missing split error).
  - Pretrained weights SHA-256 computation.

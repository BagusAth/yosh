#!/usr/bin/env python3
"""
Stage 8.1 — Group-Aware 4-Fold Out-of-Fold (OOF) Profiling (Hardened)
=====================================================================

Tujuan Ilmiah:
1. Membagi 8.980 citra latih ke dalam 4 fold secara group-aware (isolasi base_group_id 100%).
2. Melatih 4 probe model YOLO11n (12 epoch per fold) dengan Nested Group-Aware Split:
   - Outer Fold k diisolasi 100% sebagai Outer Holdout.
   - Citra/label outer holdout TIDAK PERNAH masuk ke training, validasi, ataupun checkpoint selection.
   - 3 fold sisanya dibagi secara group-aware menjadi Inner-Train (2 fold) dan Inner-Validation (1 fold).
   - Inner-Validation digunakan oleh Ultralytics untuk memilih best.pt (atau last.pt pada fixed-epoch mode).
   - Checkpoint tersebut digunakan HANYA untuk inferensi diagnostik pada Outer Holdout k (@conf=0.001).
3. Mengekspor prediksi OOF mentah untuk seluruh 8.980 citra tanpa data snooping / holdout leakage.
4. Menjalankan safety checks otomatis dan metadata checksum verification untuk mencegah silent reuse.

Output:
- stage_8_1_oof/
  ├── fold_assignments.csv
  ├── fold_class_distribution.csv
  ├── fold_group_leakage_check.csv
  ├── nested_fold_assignments.csv
  ├── data_configs/
  ├── fold_1/weights/best.pt, predictions.csv, fold_metadata.json
  ├── fold_2/weights/best.pt, predictions.csv, fold_metadata.json
  ├── fold_3/weights/best.pt, predictions.csv, fold_metadata.json
  ├── fold_4/weights/best.pt, predictions.csv, fold_metadata.json
  ├── oof_predictions_raw.csv
  ├── oof_evaluated_images.csv
  └── OOF_PROFILING_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
import ultralytics
from ultralytics import YOLO

from research_utils.detection_metrics import (
    compute_file_sha256,
    compute_dataset_inventory_fingerprint,
    compute_pretrained_weights_sha256,
    resolve_data_yaml_split,
    validate_output_directory_safety,
    validate_yolo_dataset_integrity,
)

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 8.1: Group-Aware 4-Fold Out-of-Fold (OOF) Profiling"
    )
    parser.add_argument(
        "--data-yaml",
        default="d:/UNDIP/SKRIPSI/cassava-v7-standardized/data.yaml",
        help="Path ke data.yaml (Single Source of Truth). Jika diisi, direktori train diturunkan otomatis.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path alternatif root dataset (misal cassava-v7-standardized atau cassava-v7-research).",
    )
    parser.add_argument(
        "--research-dir",
        default="d:/UNDIP/SKRIPSI/cassava-v7-research",
        help="Path fallback dataset (backward compatibility)",
    )
    parser.add_argument(
        "--image-index",
        default="d:/UNDIP/SKRIPSI/cassava-v7-stage-5/clean_image_index.csv",
        help="Path ke clean_image_index.csv yang memuat pemetaan clean_group_id",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/UNDIP/SKRIPSI/stage_8_1_oof_clean",
        help="Folder output untuk artefak Stage 8.1 (default: stage_8_1_oof_clean)",
    )
    parser.add_argument(
        "--expected-train-images",
        type=int,
        default=8980,
        help="Jumlah citra latih yang diharapkan (default: 8980 untuk dataset riset resmi)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Jumlah epoch per probe model fold (default: 12)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size pelatihan (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi citra input (default: 640)",
    )
    parser.add_argument(
        "--conf-oof",
        type=float,
        default=0.001,
        help="Threshold confidence inferensi diagnostik OOF (default: 0.001)",
    )
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=["nested_inner_val", "fixed_epoch_last"],
        default="nested_inner_val",
        help="Metode pemilihan checkpoint: 'nested_inner_val' (default, inner val fold selects best.pt) atau 'fixed_epoch_last' (uses last.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reprodusibilitas (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Perangkat GPU CUDA (default: '0')",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="yolo11n.pt",
        help="Path ke file bobot pretrained lokal YOLO11n (default: 'yolo11n.pt')",
    )
    parser.add_argument(
        "--skip-training-if-done",
        action="store_true",
        help="Lewati training fold jika weights, predictions.csv, dan metadata checksum cocok",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Jalankan hanya partisi 4-fold, audit label, dan leakage check tanpa training model",
    )
    return parser.parse_args()


def resolve_train_paths(args) -> tuple[Path, Path, list[Path]]:
    """
    Menentukan path direktori train/images dan train/labels serta daftar citra secara Single Source of Truth (SSOT).
    Prioritas:
    1. --data-yaml jika diberikan.
    2. --dataset-dir jika diberikan.
    3. --research-dir sebagai fallback.
    """
    if args.data_yaml:
        split_info = resolve_data_yaml_split(args.data_yaml, split="train", allow_empty=False)
        images_dir = split_info["images_dir"]
        labels_dir = split_info["labels_dir"]
        train_images = split_info["images"]
    elif args.dataset_dir:
        ds_p = Path(args.dataset_dir).resolve()
        images_dir = ds_p / "train" / "images"
        labels_dir = ds_p / "train" / "labels"
        train_images = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda p: p.name) if images_dir.exists() else []
    else:
        r_p = Path(args.research_dir).resolve()
        images_dir = r_p / "train" / "images"
        labels_dir = r_p / "train" / "labels"
        train_images = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda p: p.name) if images_dir.exists() else []

    if not images_dir.exists():
        raise FileNotFoundError(f"Direktori citra latih tidak ditemukan: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Direktori label latih tidak ditemukan: {labels_dir}")

    return images_dir, labels_dir, train_images


def load_annotations(images_dir: Path, labels_dir: Path, expected_count: int | None = 8980, train_images: list[Path] | None = None):
    """
    Membaca label YOLO langsung dari file .txt untuk menentukan:
    - n_objects
    - class distribution
    - empty label status
    """
    if train_images is not None:
        images_to_load = train_images
    else:
        images_to_load = sorted([images_dir / f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    annotation_data = {}

    empty_count = 0
    pure_healthy_count = 0
    pure_disease_count = 0
    mixed_count = 0

    for img_file in images_to_load:
        fname = img_file.name
        stem = img_file.stem
        lbl_file = labels_dir / f"{stem}.txt" if labels_dir else derive_yolo_label_path(img_file)
        if not lbl_file.exists():
            lbl_file = derive_yolo_label_path(img_file)

        boxes = []
        classes = []

        if lbl_file.exists() and os.path.getsize(lbl_file) > 0:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        classes.append(cls_id)
                        boxes.append([float(x) for x in parts[1:5]])

        n_obj = len(classes)
        has_healthy = (4 in classes)
        has_disease = any(c in [0, 1, 2, 3] for c in classes)

        if n_obj == 0:
            empty_count += 1
        elif has_healthy and has_disease:
            mixed_count += 1
        elif has_healthy:
            pure_healthy_count += 1
        elif has_disease:
            pure_disease_count += 1

        annotation_data[fname] = {
            "filename": fname,
            "image_path": str((images_dir / fname).resolve()),
            "label_path": str(lbl_file.resolve()),
            "n_objects": n_obj,
            "classes": classes,
            "boxes": boxes,
            "is_empty": (n_obj == 0),
            "has_healthy": has_healthy,
            "has_disease": has_disease,
        }

    total_with_healthy = pure_healthy_count + mixed_count
    total_with_disease = pure_disease_count + mixed_count

    actual_count = len(images_to_load)
    print(f"[AUDIT LABEL] Total citra latih aktual: {actual_count:,}")
    if expected_count is not None:
        print(f"              Expected citra latih      : {expected_count:,}")
        if actual_count != expected_count:
            raise ValueError(f"Actual image count ({actual_count}) does not match expected count ({expected_count})!")

    print(f"              Empty label (0 box)          : {empty_count:,}")
    print(f"              Pure HEALTHY bbox only       : {pure_healthy_count:,}")
    print(f"              Pure Disease bbox (CBB..CMD) : {pure_disease_count:,}")
    print(f"              Mixed (HEALTHY + Disease)    : {mixed_count:,}")
    print(f"              Total citra memuat HEALTHY   : {total_with_healthy:,}")
    print(f"              Total citra memuat Disease   : {total_with_disease:,}")

    if (empty_count + pure_healthy_count + pure_disease_count + mixed_count) != actual_count:
        raise ValueError(f"Sum of label categories does not match total images ({actual_count})!")
    print(f"[AUDIT LABEL] Verifikasi source of truth PASSED ({actual_count:,} citra valid).")

    return annotation_data


def load_group_mapping(index_path: Path, image_filenames: set[str]):
    """Memetakan setiap citra ke clean_group_id dari clean_image_index.csv."""
    img_to_group = {}
    with open(index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if fname in image_filenames:
                img_to_group[fname] = row["clean_group_id"]

    missing = image_filenames - set(img_to_group.keys())
    if missing:
        raise RuntimeError(f"FATAL ERROR: {len(missing)} citra tidak memiliki mapping clean_group_id!")

    print(f"[GROUP MAPPING] Berhasil memetakan {len(img_to_group):,} citra ke grup.")
    return img_to_group


def build_group_stratified_folds(
    annotation_data: dict, img_to_group: dict, n_splits: int = 4, seed: int = 42
):
    """
    Membagi kelompok (base_group_id) ke dalam 4 fold secara stratifikasi kelas.
    Menjamin:
    - Tidak ada group leakage (1 group masuk tepat 1 fold).
    - Distribusi kelas dan citra antar-fold seimbang.
    """
    rng = random.Random(seed)

    groups = defaultdict(list)
    for fname, data in annotation_data.items():
        gid = img_to_group[fname]
        groups[gid].append(fname)

    print(f"[GROUP SPLIT] Total grup atomik pada train set: {len(groups):,}")

    groups_by_sig = defaultdict(list)
    for gid, fnames in groups.items():
        cls_counter = Counter()
        for fn in fnames:
            cls_counter.update(annotation_data[fn]["classes"])

        if len(cls_counter) == 0:
            sig = "EMPTY"
        else:
            top_cls_id = cls_counter.most_common(1)[0][0]
            sig = CLASS_NAMES[top_cls_id]

        groups_by_sig[sig].append((gid, fnames, len(fnames)))

    fold_groups = {i: [] for i in range(1, n_splits + 1)}
    fold_image_counts = {i: 0 for i in range(1, n_splits + 1)}

    for sig, group_tuples in sorted(groups_by_sig.items()):
        shuffled = list(group_tuples)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda x: x[2], reverse=True)

        for gid, fnames, n_imgs in shuffled:
            target_fold = min(fold_image_counts, key=fold_image_counts.get)
            fold_groups[target_fold].append(gid)
            fold_image_counts[target_fold] += n_imgs

    image_fold_assignment = {}
    for fold_id, gids in fold_groups.items():
        for gid in gids:
            for fname in groups[gid]:
                image_fold_assignment[fname] = fold_id

    print("\n[VERIFIKASI INTEGRITAS 4-FOLD]")
    if len(image_fold_assignment) != len(annotation_data):
        raise ValueError(f"Total assigned images != actual images ({len(image_fold_assignment)} vs {len(annotation_data)})")

    all_assigned_groups = set()
    for fold_id, gids in fold_groups.items():
        gset = set(gids)
        overlap = all_assigned_groups.intersection(gset)
        if len(overlap) > 0:
            raise RuntimeError(f"Group leakage detected in fold {fold_id}: {overlap}")
        all_assigned_groups.update(gset)
        print(f"  Fold {fold_id}: {len(gids):,} grup, {fold_image_counts[fold_id]:,} citra")

    print("[VERIFIKASI INTEGRITAS] Zero cross-fold leakage terverifikasi (100% PASS).\n")
    return fold_groups, image_fold_assignment, groups


def generate_nested_fold_configs(
    annotation_data: dict,
    image_fold_assignment: dict,
    fold_groups: dict,
    output_dir: Path,
    selection_mode: str = "nested_inner_val",
    n_splits: int = 4,
):
    """
    Menyusun konfigurasi Nested Group-Aware Split:
    Untuk setiap outer fold k (Outer Holdout):
    - Outer holdout fold k diisolasi 100% dari fold_k_data.yaml.
    - Sisa 3 fold dibagi menjadi Inner-Train (2 fold) dan Inner-Validation (1 fold).
    - Inner-Validation = ((k % n_splits) + 1) -> pasti bukan k dan selalu deterministik.
    - Inner-Train = 2 fold sisa lainnya.
    - Checkpoint 'best.pt' dipilih murni berdasarkan Inner-Validation tanpa pernah melihat Outer Holdout k.
    - Outer holdout k disimpan di fold_{k}_outer_holdout.txt untuk inferensi OOF.
    """
    config_dir = output_dir / "data_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_paths = {}
    nested_specs = {}

    for fold_id in range(1, n_splits + 1):
        outer_holdout_fold = fold_id
        remaining_folds = [f for f in range(1, n_splits + 1) if f != outer_holdout_fold]

        if selection_mode == "nested_inner_val":
            # Deterministic round-robin inner-val fold among remaining
            inner_val_fold = (fold_id % n_splits) + 1
            if inner_val_fold == outer_holdout_fold:
                # Safety fallback (though mod arithmetic guarantees != fold_id for n_splits >= 2)
                inner_val_fold = remaining_folds[0]
            inner_train_folds = [f for f in remaining_folds if f != inner_val_fold]
        else:
            # fixed_epoch_last: train on all 3 remaining folds, no validation
            inner_val_fold = None
            inner_train_folds = remaining_folds

        # Filter image paths
        outer_holdout_images = []
        inner_val_images = []
        inner_train_images = []

        for fname, assigned_fold in image_fold_assignment.items():
            img_path = annotation_data[fname]["image_path"].replace("\\", "/")
            if assigned_fold == outer_holdout_fold:
                outer_holdout_images.append(img_path)
            elif selection_mode == "nested_inner_val" and assigned_fold == inner_val_fold:
                inner_val_images.append(img_path)
            elif assigned_fold in inner_train_folds:
                inner_train_images.append(img_path)

        # File paths
        train_txt = config_dir / f"fold_{fold_id}_inner_train.txt"
        val_txt = config_dir / f"fold_{fold_id}_inner_val.txt"
        holdout_txt = config_dir / f"fold_{fold_id}_outer_holdout.txt"

        train_txt.write_text("\n".join(inner_train_images) + "\n", encoding="utf-8")
        holdout_txt.write_text("\n".join(outer_holdout_images) + "\n", encoding="utf-8")

        if selection_mode == "nested_inner_val":
            val_txt.write_text("\n".join(inner_val_images) + "\n", encoding="utf-8")
            yaml_val_entry = val_txt.resolve().as_posix()
        else:
            # If no inner validation, train on all 3 folds and use last.pt
            yaml_val_entry = train_txt.resolve().as_posix()

        # Group-isolation verification for this specific fold
        holdout_groups = set(fold_groups[outer_holdout_fold])
        inner_train_groups = set()
        for f_sub in inner_train_folds:
            inner_train_groups.update(fold_groups[f_sub])
        
        leak_train_holdout = holdout_groups.intersection(inner_train_groups)
        if len(leak_train_holdout) > 0:
            raise RuntimeError(f"FATAL: Group leakage between inner-train and outer-holdout in fold {fold_id}!")

        if selection_mode == "nested_inner_val":
            inner_val_groups = set(fold_groups[inner_val_fold])
            leak_val_holdout = holdout_groups.intersection(inner_val_groups)
            if len(leak_val_holdout) > 0:
                raise RuntimeError(f"FATAL: Group leakage between inner-val and outer-holdout in fold {fold_id}!")
            leak_train_val = inner_train_groups.intersection(inner_val_groups)
            if len(leak_train_val) > 0:
                raise RuntimeError(f"FATAL: Group leakage between inner-train and inner-val in fold {fold_id}!")

        yaml_content = f"""# Ultralytics YOLO OOF Configuration — Fold {fold_id} (Nested Isolated)
# Outer Holdout: Fold {outer_holdout_fold} (EXCLUDED FROM TRAINING & CHECKPOINT SELECTION)
# Inner Train Folds: {inner_train_folds}
# Inner Val Fold: {inner_val_fold if inner_val_fold else 'NONE (fixed_epoch_last)'}

train: {train_txt.resolve().as_posix()}
val: {yaml_val_entry}

nc: 5
names: {CLASS_NAMES}
"""
        yaml_file = config_dir / f"fold_{fold_id}_data.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        yaml_paths[fold_id] = yaml_file
        nested_specs[fold_id] = {
            "outer_holdout_fold": outer_holdout_fold,
            "outer_holdout_count": len(outer_holdout_images),
            "inner_train_folds": inner_train_folds,
            "inner_train_count": len(inner_train_images),
            "inner_val_fold": inner_val_fold,
            "inner_val_count": len(inner_val_images),
            "holdout_txt": str(holdout_txt.resolve()),
            "train_txt": str(train_txt.resolve()),
            "val_txt": str(val_txt.resolve()) if (selection_mode == "nested_inner_val" and inner_val_fold) else "",
            "yaml_file": str(yaml_file.resolve()),
        }

    print(f"[DATA CONFIG] 4 file YAML konfigurasi fold nested dibuat di {config_dir}")
    return yaml_paths, nested_specs


def build_fold_run_identity(
    script_path: str | Path,
    yaml_path: str | Path,
    inner_train_txt: str | Path,
    inner_val_txt: str | Path,
    outer_holdout_txt: str | Path,
    dataset_inventory_fingerprint: str,
    checkpoint_selection_mode: str,
    fold_id: int,
    epochs: int,
    batch_size: int,
    imgsz: int,
    seed: int,
    conf_thresh: float,
    model_variant: str = "yolo11n.pt",
    device: str = "",
    pretrained_weights_sha256: str = "",
    checkpoint_eval_sha256: str = "",
    predictions_sha256: str = "",
    holdout_image_count: int = 0,
    inner_train_count: int = 0,
    inner_val_count: int = 0,
) -> dict[str, Any]:
    """Membangun objek identitas konfigurasi fold run secara eksplisit (20+ parameter)."""
    p_script = Path(script_path)
    p_yaml = Path(yaml_path)
    p_train = Path(inner_train_txt)
    p_val = Path(inner_val_txt)
    p_holdout = Path(outer_holdout_txt)

    return {
        "fold_id": fold_id,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "seed": seed,
        "conf_oof": conf_thresh,
        "model_name": model_variant,
        "device": str(device),
        "outer_holdout_count": holdout_image_count,
        "inner_train_count": inner_train_count,
        "inner_val_count": inner_val_count,
        "script_sha256": compute_file_sha256(p_script) if p_script.exists() else str(script_path),
        "fold_yaml_sha256": compute_file_sha256(p_yaml) if p_yaml.exists() else str(yaml_path),
        "inner_train_sha256": compute_file_sha256(p_train) if p_train.exists() else str(inner_train_txt),
        "inner_val_sha256": compute_file_sha256(p_val) if p_val.exists() else str(inner_val_txt),
        "outer_holdout_sha256": compute_file_sha256(p_holdout) if p_holdout.exists() else str(outer_holdout_txt),
        "dataset_inventory_fingerprint": dataset_inventory_fingerprint,
        "pretrained_weights_sha256": pretrained_weights_sha256,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "ultralytics_version": getattr(ultralytics, "__version__", ""),
        "checkpoint_sha256": checkpoint_eval_sha256,
        "pred_csv_sha256": predictions_sha256,
        "evaluated_images": holdout_image_count,
    }


def verify_fold_reuse_safety(
    meta_path: Path,
    expected_identity: dict[str, Any],
    checkpoint_path: Path,
    predictions_path: Path,
    holdout_image_count: int,
) -> tuple[bool, str]:
    """
    Memverifikasi apakah artefak fold yang sudah ada aman untuk digunakan kembali (--skip-training-if-done).
    Memeriksa seluruh field identitas konfigurasi, hash skrip, YAML, daftar split,
    serta hash aktual checkpoint dan file prediksi.
    """
    if not meta_path.exists():
        return False, "Metadata JSON not found"
    if not checkpoint_path.exists():
        return False, "Checkpoint file does not exist"
    if not predictions_path.exists():
        return False, "Predictions CSV file does not exist"

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
    except Exception as e:
        return False, f"Failed to parse metadata JSON: {e}"

    for k, exp_val in expected_identity.items():
        if k in ("checkpoint_sha256", "pred_csv_sha256", "evaluated_images", "timestamp"):
            continue
        saved_val = saved_meta.get(k)
        if saved_val != exp_val:
            return False, f"{k} mismatch: saved={saved_val} vs exp={exp_val}"

    actual_ckpt_sha = compute_file_sha256(checkpoint_path)
    saved_ckpt_sha = saved_meta.get("checkpoint_sha256", "")
    if not saved_ckpt_sha or actual_ckpt_sha != saved_ckpt_sha:
        return False, f"Checkpoint hash mismatch: saved={saved_ckpt_sha[:8]} vs actual={actual_ckpt_sha[:8]}"

    actual_pred_sha = compute_file_sha256(predictions_path)
    saved_pred_sha = saved_meta.get("pred_csv_sha256", "")
    if not saved_pred_sha or actual_pred_sha != saved_pred_sha:
        return False, f"Predictions CSV hash mismatch: saved={saved_pred_sha[:8]} vs actual={actual_pred_sha[:8]}"

    if saved_meta.get("evaluated_images") != holdout_image_count:
        return False, f"Evaluated images count mismatch: saved={saved_meta.get('evaluated_images')} vs exp={holdout_image_count}"

    return True, "Artifacts and metadata verified 100% matching"


def train_and_predict_fold(
    fold_id: int,
    yaml_path: Path,
    holdout_filenames: list[str],
    nested_spec: dict,
    annotation_data: dict,
    output_dir: Path,
    args,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
):
    """
    Melatih probe model YOLO11n pada inner-train (12 epoch), memilih checkpoint berdasarkan
    inner-validation (atau last.pt), lalu menjalankan inferensi diagnostik pada outer holdout fold (@conf=0.001).
    Menjamin validasi checksum identitas run secara lengkap sebelum mengizinkan reuse checkpoint.
    """
    fold_dir = output_dir / f"fold_{fold_id}"
    validate_output_directory_safety(
        output_dir=fold_dir,
        stage_name=f"8.1_fold_{fold_id}",
        allow_reuse=args.skip_training_if_done,
        is_dry_run=args.dry_run,
        is_root=False,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred_csv_path = fold_dir / "predictions.csv"
    weights_dir = fold_dir / "weights"
    metadata_path = fold_dir / "fold_metadata.json"

    best_weight_path = weights_dir / "best.pt"
    last_weight_path = weights_dir / "last.pt"
    eval_weight_path = best_weight_path if args.checkpoint_selection_mode == "nested_inner_val" else last_weight_path

    pw_arg = getattr(args, "pretrained_weights", "yolo11n.pt")
    pretrained_weights_path = Path(pw_arg).resolve()
    if not pretrained_weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights '{pretrained_weights_path}' not found! Dilarang keras mengunduh bobot secara dinamis.")
    pretrained_sha = compute_pretrained_weights_sha256(pretrained_weights_path)

    dataset_inv_sha = compute_dataset_inventory_fingerprint(
        images_dir, labels_dir, data_yaml_path=args.data_yaml
    ) if (images_dir and labels_dir) else ""

    expected_identity = build_fold_run_identity(
        script_path=__file__,
        yaml_path=yaml_path,
        inner_train_txt=nested_spec.get("train_txt", ""),
        inner_val_txt=nested_spec.get("val_txt", "") if nested_spec.get("val_txt") else "",
        outer_holdout_txt=nested_spec.get("holdout_txt", ""),
        dataset_inventory_fingerprint=dataset_inv_sha,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
        fold_id=fold_id,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        conf_thresh=args.conf_oof,
        model_variant="yolo11n.pt",
        device=str(args.device),
        pretrained_weights_sha256=pretrained_sha,
        holdout_image_count=nested_spec["outer_holdout_count"],
        inner_train_count=nested_spec["inner_train_count"],
        inner_val_count=nested_spec["inner_val_count"],
    )

    # Strict Checksum and Artifact Verification for --skip-training-if-done
    if args.skip_training_if_done and pred_csv_path.exists() and eval_weight_path.exists():
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=metadata_path,
            expected_identity=expected_identity,
            checkpoint_path=eval_weight_path,
            predictions_path=pred_csv_path,
            holdout_image_count=len(holdout_filenames),
        )
        if can_reuse:
            print(f"[FOLD {fold_id}] Melewati pelatihan dan inferensi (artefak dan metadata terverifikasi cocok 100%).")
            return pred_csv_path
        else:
            print(f"[FOLD {fold_id}] WARNING: Artefak lama tidak lolos verifikasi reuse ({reason})!")

    # Guard against accidental reuse of unverified/mismatched checkpoint
    if eval_weight_path.exists():
        raise RuntimeError(
            f"[FOLD {fold_id}] SAFETY VIOLATION: Checkpoint '{eval_weight_path}' sudah ada di folder tujuan tetapi gagal verifikasi identitas run / checksum!\n"
            "Untuk melindungi integritas riset, dilarang menjalankan inferensi diam-diam menggunakan bobot yang tidak cocok ataupun menimpa bobot lama.\n"
            "Gunakan direktori --output-dir baru yang bersih untuk run ini."
        )

    # 1. Train Model
    print("=" * 80)
    print(f"MEMULAI PROBE TRAINING (NESTED ISOLATION): FOLD {fold_id} / 4")
    print(f"Outer Holdout  : Fold {nested_spec['outer_holdout_fold']} ({nested_spec['outer_holdout_count']:,} citra - TERISOLASI 100%)")
    print(f"Inner Train    : Folds {nested_spec['inner_train_folds']} ({nested_spec['inner_train_count']:,} citra)")
    print(f"Inner Val      : Fold {nested_spec['inner_val_fold']} ({nested_spec['inner_val_count']:,} citra - CHECKPOINT SELECTION)")
    print(f"Selection Mode : {args.checkpoint_selection_mode}")
    print(f"Train Epochs   : {args.epochs}")
    print(f"Batch Size     : {args.batch}")
    print(f"YAML Path      : {yaml_path}")
    print("=" * 80)

    t_start = time.time()
    model = YOLO(str(pretrained_weights_path))

    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        workers=0,
        project=str(fold_dir),
        name="train_run",
        save=True,
        exist_ok=True,
        verbose=False,
        val=True,
        plots=False,
    )

    t_train = time.time() - t_start
    print(f"[FOLD {fold_id}] Pelatihan selesai dalam {t_train / 60:.2f} menit.")

    weights_dir.mkdir(parents=True, exist_ok=True)
    trained_best = fold_dir / "train_run" / "weights" / "best.pt"
    trained_last = fold_dir / "train_run" / "weights" / "last.pt"

    if trained_best.exists():
        shutil.copy2(trained_best, best_weight_path)
    if trained_last.exists():
        shutil.copy2(trained_last, last_weight_path)

    if not eval_weight_path.exists():
        raise FileNotFoundError(f"Model checkpoint yang dibutuhkan ({eval_weight_path}) tidak ditemukan di {fold_dir}")

    checkpoint_sha256 = compute_file_sha256(eval_weight_path)

    del model, train_results
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Diagnostic OOF Prediction pada Outer Holdout (@conf=0.001)
    print(f"[FOLD {fold_id}] Menjalankan inferensi diagnostik OOF @ conf = {args.conf_oof} pada OUTER HOLDOUT...")
    eval_model = YOLO(str(eval_weight_path))

    holdout_paths = [annotation_data[fn]["image_path"] for fn in holdout_filenames]

    t_infer_start = time.time()
    predictions = []
    evaluated_count = 0
    chunk_size = 32

    for c_idx in range(0, len(holdout_paths), chunk_size):
        chunk = holdout_paths[c_idx : c_idx + chunk_size]
        results = eval_model.predict(
            source=chunk,
            conf=args.conf_oof,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            stream=False,
        )

        for res in results:
            evaluated_count += 1
            img_fname = Path(res.path).name
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for b_idx in range(len(boxes)):
                    cls_id = int(boxes.cls[b_idx].item())
                    conf_val = float(boxes.conf[b_idx].item())
                    xyxy = boxes.xyxy[b_idx].tolist()
                    predictions.append({
                        "image_id": img_fname,
                        "fold_id": fold_id,
                        "prediction_id": b_idx,
                        "class_id": cls_id,
                        "confidence": f"{conf_val:.6f}",
                        "x1": f"{xyxy[0]:.2f}",
                        "y1": f"{xyxy[1]:.2f}",
                        "x2": f"{xyxy[2]:.2f}",
                        "y2": f"{xyxy[3]:.2f}",
                    })

    del eval_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Verify holdout prediction integrity
    if evaluated_count != len(holdout_filenames):
        raise ValueError(
            f"[FOLD {fold_id}] FAIL: Jumlah citra dievaluasi ({evaluated_count}) "
            f"tidak sama dengan jumlah holdout ({len(holdout_filenames)})!"
        )

    holdout_set = set(holdout_filenames)
    foreign_preds = [p["image_id"] for p in predictions if p["image_id"] not in holdout_set]
    if foreign_preds:
        raise ValueError(
            f"[FOLD {fold_id}] FAIL: Ditemukan {len(foreign_preds)} prediksi untuk citra dari luar fold holdout!"
        )

    t_infer = time.time() - t_infer_start
    print(f"[FOLD {fold_id}] Inferensi selesai: {evaluated_count:,} citra diproses, "
          f"{len(predictions):,} bounding box terdeteksi ({t_infer / 60:.2f} menit).")

    # 3. Simpan predictions.csv
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "fold_id", "prediction_id", "class_id",
            "confidence", "x1", "y1", "x2", "y2"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    pred_csv_sha256 = compute_file_sha256(pred_csv_path)

    # 4. Simpan fold_metadata.json
    fold_metadata = dict(expected_identity)
    fold_metadata["checkpoint_sha256"] = checkpoint_sha256
    fold_metadata["pred_csv_sha256"] = pred_csv_sha256
    fold_metadata["checkpoint_used"] = str(eval_weight_path.name)
    fold_metadata["evaluated_images"] = evaluated_count
    fold_metadata["num_predictions"] = len(predictions)
    fold_metadata["timestamp"] = datetime.now().isoformat()

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(fold_metadata, f, indent=2)

    print(f"[FOLD {fold_id}] Prediksi dan metadata tersimpan di {fold_dir}")
    return pred_csv_path


def main():
    args = parse_args()
    print("=" * 80)
    print("STAGE 8.1 — GROUP-AWARE 4-FOLD OUT-OF-FOLD (OOF) PROFILING (HARDENED)")
    print("=" * 80)
    print(f"Waktu Mulai         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data YAML           : {args.data_yaml}")
    print(f"Dataset Dir         : {args.dataset_dir or args.research_dir}")
    print(f"Output Dir          : {args.output_dir}")
    print(f"Selection Mode      : {args.checkpoint_selection_mode}")
    print(f"Probe Epochs        : {args.epochs}")
    print(f"Conf Threshold      : {args.conf_oof}")
    print(f"Random Seed         : {args.seed}")
    print(f"CUDA Device         : {args.device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    images_dir, labels_dir, train_images = resolve_train_paths(args)
    output_dir = Path(args.output_dir).resolve()
    # P0 Safety Guard: Check output directory safety before writing partition files or configs
    validate_output_directory_safety(
        output_dir=output_dir,
        stage_name="8.1",
        allow_reuse=args.skip_training_if_done,
        is_dry_run=args.dry_run,
        is_root=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Integrity Preflight Validation (SSOT)
    integrity_result = validate_yolo_dataset_integrity(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_files=train_images,
        data_yaml_path=args.data_yaml if args.data_yaml else None,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        allow_empty_labels=True,
    )
    if integrity_result.get("status") != "VALID":
        raise ValueError(f"[STAGE 8.1] Dataset integrity validation failed: {integrity_result}")
    print(f"[STAGE 8.1] Dataset integrity verified: {integrity_result['total_images']:,} images "
          f"({integrity_result['background_images']} background), {integrity_result['total_objects']:,} objects.")

    # 2. Audit Label & Ground Truth dari file label .txt
    annotation_data = load_annotations(images_dir, labels_dir, expected_count=args.expected_train_images, train_images=train_images)

    # 2. Pemetaan Group
    img_to_group = load_group_mapping(Path(args.image_index).resolve(), set(annotation_data.keys()))

    # 3. Pembagian 4-Fold Group-Aware
    fold_groups, image_fold_assignment, groups = build_group_stratified_folds(
        annotation_data, img_to_group, n_splits=4, seed=args.seed
    )

    # Simpan fold_assignments.csv
    fold_assign_path = output_dir / "fold_assignments.csv"
    with open(fold_assign_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "base_group_id", "fold_id", "split"])
        for fname, fold_id in sorted(image_fold_assignment.items()):
            writer.writerow([fname, img_to_group[fname], fold_id, "train"])
    print(f"[OUTPUT] Tersimpan: {fold_assign_path}")

    # Simpan fold_class_distribution.csv
    fold_class_dist_path = output_dir / "fold_class_distribution.csv"
    with open(fold_class_dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "num_images", "num_groups", "EMPTY", "CBB", "CBSD", "CGM", "CMD", "HEALTHY"])
        for f_id in range(1, 5):
            c_counts = Counter()
            for fn, f_curr in image_fold_assignment.items():
                if f_curr == f_id:
                    classes = annotation_data[fn]["classes"]
                    if not classes:
                        c_counts["EMPTY"] += 1
                    else:
                        for c in classes:
                            c_counts[CLASS_NAMES[c]] += 1
            writer.writerow([
                f_id,
                sum(1 for f_curr in image_fold_assignment.values() if f_curr == f_id),
                len(fold_groups[f_id]),
                c_counts["EMPTY"],
                c_counts["CBB"],
                c_counts["CBSD"],
                c_counts["CGM"],
                c_counts["CMD"],
                c_counts["HEALTHY"],
            ])
    print(f"[OUTPUT] Tersimpan: {fold_class_dist_path}")

    # Simpan fold_group_leakage_check.csv
    leakage_check_path = output_dir / "fold_group_leakage_check.csv"
    with open(leakage_check_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_a", "fold_b", "shared_groups", "leakage_status"])
        for i in range(1, 5):
            for j in range(i + 1, 5):
                inter = set(fold_groups[i]).intersection(set(fold_groups[j]))
                writer.writerow([f"fold_{i}", f"fold_{j}", len(inter), "LEAKAGE_DETECTED" if inter else "ZERO_LEAKAGE"])
    print(f"[OUTPUT] Tersimpan: {leakage_check_path}")

    # 4. Buat File Konfigurasi Pelatihan Fold Nested
    yaml_paths, nested_specs = generate_nested_fold_configs(
        annotation_data,
        image_fold_assignment,
        fold_groups,
        output_dir,
        selection_mode=args.checkpoint_selection_mode,
        n_splits=4,
    )

    # Simpan nested_fold_assignments.csv
    nested_assign_path = output_dir / "nested_fold_assignments.csv"
    with open(nested_assign_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "outer_holdout_fold", "outer_holdout_images", "inner_train_folds", "inner_train_images", "inner_val_fold", "inner_val_images"])
        for f_id in range(1, 5):
            spec = nested_specs[f_id]
            writer.writerow([
                f_id,
                spec["outer_holdout_fold"],
                spec["outer_holdout_count"],
                "+".join(str(x) for x in spec["inner_train_folds"]),
                spec["inner_train_count"],
                spec["inner_val_fold"] if spec["inner_val_fold"] else "NONE",
                spec["inner_val_count"],
            ])
    print(f"[OUTPUT] Tersimpan: {nested_assign_path}")

    # Path integrity verification across generated splits
    is_standardized = bool(args.data_yaml and any(k in str(args.data_yaml).lower() for k in ("standardized", "runtime-mirror", "mirror")))
    for f_id in range(1, 5):
        spec = nested_specs[f_id]
        for check_key in ["train_txt", "val_txt", "holdout_txt"]:
            check_path = spec.get(check_key, "")
            if check_path and Path(check_path).exists():
                lines = Path(check_path).read_text(encoding="utf-8").splitlines()
                if is_standardized:
                    research_leaks = [l for l in lines if "cassava-v7-research" in l]
                    if research_leaks:
                        raise RuntimeError(
                            f"FATAL INTEGRITY VIOLATION: Split file {check_path} (Fold {f_id}) "
                            f"contains {len(research_leaks)} paths pointing to historical 'cassava-v7-research'!\n"
                            f"Contoh: {research_leaks[:3]}"
                        )

    if args.dry_run:
        print("\n" + "=" * 80)
        print("[DRY-RUN] Verifikasi partisi, nested isolation, dan label audit SELESAI (100% SUKSES).")
        print("          Outer holdout 100% terisolasi dari training dan validation YAML.")
        print(f"          Path dataset terstandarisasi diverifikasi: {is_standardized} (Zero research path leakage).")
        print("=" * 80)
        return

    # 5. Pelatihan dan Inferensi OOF untuk Ke-4 Fold
    fold_pred_files = {}
    for fold_id in range(1, 5):
        holdout_imgs = [fn for fn, f_curr in image_fold_assignment.items() if f_curr == fold_id]
        pred_file = train_and_predict_fold(
            fold_id=fold_id,
            yaml_path=yaml_paths[fold_id],
            holdout_filenames=holdout_imgs,
            nested_spec=nested_specs[fold_id],
            annotation_data=annotation_data,
            output_dir=output_dir,
            args=args,
            images_dir=images_dir,
            labels_dir=labels_dir,
        )
        fold_pred_files[fold_id] = pred_file

    # 6. Agregasi Raw OOF Predictions
    print("\n[AGREGASI PREDIKSI OOF]")
    raw_oof_path = output_dir / "oof_predictions_raw.csv"
    evaluated_img_path = output_dir / "oof_evaluated_images.csv"

    all_raw_predictions = []
    images_with_preds = defaultdict(int)

    for fold_id, p_file in fold_pred_files.items():
        with open(p_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_raw_predictions.append(row)
                images_with_preds[row["image_id"]] += 1

    with open(raw_oof_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "fold_id", "prediction_id", "class_id",
            "confidence", "x1", "y1", "x2", "y2"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_raw_predictions)
    print(f"[OUTPUT] Tersimpan: {raw_oof_path} ({len(all_raw_predictions):,} prediksi mentah)")

    with open(evaluated_img_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "fold_id", "num_predictions", "n_gt_objects", "is_empty"])
        for fname, fold_id in sorted(image_fold_assignment.items()):
            writer.writerow([
                fname,
                fold_id,
                images_with_preds.get(fname, 0),
                annotation_data[fname]["n_objects"],
                int(annotation_data[fname]["is_empty"]),
            ])
    print(f"[OUTPUT] Tersimpan: {evaluated_img_path}")

    # 7. SAFETY CHECKS PASCA-OOF
    print("\n" + "=" * 80)
    print("SAFETY CHECKS PASCA-OOF PROFILING")
    print("=" * 80)

    actual_image_count = len(annotation_data)
    expected_benchmark_count = 8980

    # Check 1: Unique images assigned equals dataset inventory
    if len(image_fold_assignment) != actual_image_count:
        raise ValueError(
            f"FAIL: Assigned images count ({len(image_fold_assignment):,}) does not match "
            f"actual dataset inventory ({actual_image_count:,})!"
        )
    if args.data_yaml and "standardized" in str(args.data_yaml).lower():
        if actual_image_count != expected_benchmark_count:
            raise ValueError(
                f"FAIL: Standardized cassava dataset expected {expected_benchmark_count} images, "
                f"got {actual_image_count}!"
            )
    print(f"[SAFETY CHECK 1] Total citra ter-assign: {len(image_fold_assignment):,} (Actual: {actual_image_count:,}, Expected Benchmark: {expected_benchmark_count:,}) (PASS)")

    # Check 2: Images accounted in evaluated_images
    with open(evaluated_img_path, "r", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    if len(eval_rows) != actual_image_count:
        raise ValueError(f"FAIL: Expected {actual_image_count} evaluated images, got {len(eval_rows)}")
    print(f"[SAFETY CHECK 2] Total citra dievaluasi OOF: {len(eval_rows):,} (PASS)")

    # Check 3: fold_prediction == fold_assignment
    mismatch_count = 0
    for row in all_raw_predictions:
        fname = row["image_id"]
        fold_pred = int(row["fold_id"])
        fold_assigned = image_fold_assignment[fname]
        if fold_pred != fold_assigned:
            mismatch_count += 1
    if mismatch_count != 0:
        raise ValueError(f"FAIL: Ditemukan {mismatch_count} ketidaksesuaian fold_prediction vs fold_assignment!")
    print("[SAFETY CHECK 3] Verifikasi fold_prediction == fold_assignment: 100% Cocok (PASS)")

    # Check 4: Cross-fold group leakage
    leakage_count = 0
    for i in range(1, 5):
        for j in range(i + 1, 5):
            inter = set(fold_groups[i]).intersection(set(fold_groups[j]))
            leakage_count += len(inter)
    if leakage_count != 0:
        raise ValueError(f"FAIL: Ditemukan {leakage_count} group leakage!")
    print("[SAFETY CHECK 4] Cross-Fold Group Leakage: 0.00% (PASS)")

    # Check 5: Nested holdout isolation verification
    for f_id in range(1, 5):
        spec = nested_specs[f_id]
        h_groups = set(fold_groups[spec["outer_holdout_fold"]])
        train_groups = set()
        for sf in spec["inner_train_folds"]:
            train_groups.update(fold_groups[sf])
        if h_groups.intersection(train_groups):
            raise RuntimeError(f"FAIL: Outer holdout leaked into inner training in fold {f_id}!")
        if spec["inner_val_fold"]:
            v_groups = set(fold_groups[spec["inner_val_fold"]])
            if h_groups.intersection(v_groups):
                raise RuntimeError(f"FAIL: Outer holdout leaked into inner validation in fold {f_id}!")
    print("[SAFETY CHECK 5] Nested Holdout Isolation (Zero Snooping): 100% PASS")

    print("=" * 80)
    print("SELURUH SAFETY CHECKS TAHAP 8.1: PASS 100%")
    print("=" * 80)

    # 8. Tulis OOF_PROFILING_REPORT.md
    report_path = output_dir / "OOF_PROFILING_REPORT.md"
    report_content = f"""# Laporan Hasil Stage 8.1 — Group-Aware 4-Fold OOF Profiling (Hardened)
Tanggal Eksekusi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Ringkasan Eksekusi
- **Dataset Latih**: 8.980 citra ({len(groups):,} grup atomik)
- **Skema Partisi**: Nested Group-Aware Stratified 4-Fold Split
- **Isolasi Grup**: 0.00% Cross-Fold Group Leakage (100% Clean)
- **Metode Seleksi Checkpoint**: {args.checkpoint_selection_mode} (Inner-Validation Isolate)
- **Model Probe**: YOLO11n ({args.epochs} epoch per fold)
- **Threshold Inferensi Diagnostik**: conf = {args.conf_oof}

## 2. Struktur Nested Fold
| Outer Fold | Outer Holdout (Eval Only) | Inner Train Folds | Inner Val Fold (Checkpoint Selection) |
| :---: | :---: | :---: | :---: |
| Fold 1 | Fold 1 ({nested_specs[1]['outer_holdout_count']} citra) | Folds {nested_specs[1]['inner_train_folds']} ({nested_specs[1]['inner_train_count']} citra) | Fold {nested_specs[1]['inner_val_fold']} ({nested_specs[1]['inner_val_count']} citra) |
| Fold 2 | Fold 2 ({nested_specs[2]['outer_holdout_count']} citra) | Folds {nested_specs[2]['inner_train_folds']} ({nested_specs[2]['inner_train_count']} citra) | Fold {nested_specs[2]['inner_val_fold']} ({nested_specs[2]['inner_val_count']} citra) |
| Fold 3 | Fold 3 ({nested_specs[3]['outer_holdout_count']} citra) | Folds {nested_specs[3]['inner_train_folds']} ({nested_specs[3]['inner_train_count']} citra) | Fold {nested_specs[3]['inner_val_fold']} ({nested_specs[3]['inner_val_count']} citra) |
| Fold 4 | Fold 4 ({nested_specs[4]['outer_holdout_count']} citra) | Folds {nested_specs[4]['inner_train_folds']} ({nested_specs[4]['inner_train_count']} citra) | Fold {nested_specs[4]['inner_val_fold']} ({nested_specs[4]['inner_val_count']} citra) |

## 3. Status Safety Checks
- [x] Total citra unik ter-assign: 8.980 (PASS)
- [x] Total citra terprediksi OOF: 8.980 (PASS)
- [x] Konsistensi fold prediksi dan fold penugasan: 100% Cocok (PASS)
- [x] Kebocoran kelompok antar-fold (Group Leakage): 0 (PASS)
- [x] Nested Holdout Isolation (Zero Snooping into Checkpoint Selection): PASS

## 4. Langkah Selanjutnya
Artefak `oof_predictions_raw.csv` siap digunakan oleh `stage_8_2_build_difficulty_manifest.py`
untuk pencocokan greedy one-to-one class-agnostic matching dan kalkulasi skor kesulitan $D_i$ serta $D_\\text{{image}}$.
"""
    report_path.write_text(report_content, encoding="utf-8")
    print(f"[REPORT] Tersimpan: {report_path}")


if __name__ == "__main__":
    main()

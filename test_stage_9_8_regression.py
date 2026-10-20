#!/usr/bin/env python3
"""
Test Suite: test_stage_9_8_regression.py
Stage 9.8 — Training Configuration Integration, Final Validation & Readiness Gate
==================================================================================

Ruang Lingkup Pengujian:
Group A: Training Configuration (Defaults: 30 epochs, patience 10, batch 16, imgsz 640, workers 2, AMP True, plots False, close_mosaic 6, CLI validation, overrides)
Group B: Run Identity & Strict Reuse (Every config changes identity, checkpoint reuse rejected on mismatch, fail-closed on missing fields, partial run safety)
Group C: Dataset Integrity Preflight (Missing pairs, orphans, invalid coords/bbox, background images, txt subsets, invocation on Stage 8.1 and 8.4)
Group D: Ablation Configuration Consistency (EXP-0 to EXP-3 toggles, identical global settings, orchestrator execution, failure halting)
Group E: Data Loading, Augmentation & Trainer Overrides (Targeted crop toggles, coordinate validity, collate fn, worker/AMP propagation)
Group F: Output Safety, Historical Protection & Zero Dataset Mutation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
import torch

from research_utils.detection_metrics import (
    CLASS_NAMES,
    compute_file_sha256,
    compute_dataset_inventory_fingerprint,
    compute_pretrained_weights_sha256,
    derive_yolo_label_path,
    resolve_data_yaml_split,
    validate_output_directory_safety,
    validate_yolo_dataset_integrity,
    FORBIDDEN_HISTORICAL_DIRS,
)
from stage_8_3_adaptive_pipeline import AdaptiveYOLODataset
from stage_8_4_run_ablation_experiments import (
    EXPERIMENT_SPECS,
    build_experiment_run_identity,
    parse_args,
    resolve_pretrained_weights,
    run_ablation_pipeline,
    run_single_experiment,
    validate_training_args,
    verify_ablation_reuse_safety,
    verify_manifest_provenance_and_coverage,
)


class MockTrainer:
    """Mock trainer untuk pengujian orkestrator ablasi dan parameter propagation."""
    instances = []

    def __init__(self, overrides=None, enable_mixture_sampling=False,
                 enable_targeted_aug=False, manifest_path=None, aug_prob=0.50):
        self.overrides = overrides or {}
        self.enable_mixture_sampling = enable_mixture_sampling
        self.enable_targeted_aug = enable_targeted_aug
        self.manifest_path = manifest_path
        self.aug_prob = aug_prob
        self.epoch = 0
        self.best_epoch = 0
        MockTrainer.instances.append(self)

    def train(self):
        proj = Path(self.overrides.get("project", "."))
        name = self.overrides.get("name", "exp")
        weights_dir = proj / name / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(b"MOCK_YOLO11N_TRAINED_WEIGHTS_FOR_" + name.encode("utf-8"))
        self.epoch = self.overrides.get("epochs", 1) - 1
        self.best_epoch = self.epoch


class FailingMockTrainer(MockTrainer):
    """Mock trainer yang sengaja gagal untuk menguji kegagalan orkestrasi."""
    def train(self):
        raise RuntimeError("Simulated trainer failure during ablation execution!")


def mock_model_eval_fn(best_target, data_yaml_path, batch, imgsz, device):
    """Mock fungsi evaluasi model validation set."""
    return {
        "precision": 0.85,
        "recall": 0.80,
        "map50": 0.88,
        "map50_95": 0.65,
        "per_class_ap50": {"CBB": 0.85, "CBSD": 0.86, "CGM": 0.87, "CMD": 0.90, "HEALTHY": 0.92},
        "per_class_map50_95": {"CBB": 0.60, "CBSD": 0.62, "CGM": 0.65, "CMD": 0.70, "HEALTHY": 0.68},
    }


# ==============================================================================
# GROUP A: Training Configuration Integration
# ==============================================================================

class TestGroupATrainingConfiguration(unittest.TestCase):
    """Group A: Verifikasi konfigurasi pelatihan baru Stage 9.8 dan validasi CLI."""

    def test_default_training_parameters(self):
        # Memeriksa seluruh parameter default Stage 9.8
        parser = argparse.ArgumentParser()
        # Gunakan fungsi parse_args() resmi dengan argumen kosong
        args = parse_args.__wrapped__() if hasattr(parse_args, "__wrapped__") else None
        # Panggil parse_args melalui argv tiruan
        orig_argv = sys.argv
        sys.argv = ["stage_8_4_run_ablation_experiments.py"]
        try:
            parsed = parse_args()
        finally:
            sys.argv = orig_argv

        self.assertEqual(parsed.epochs, 30, "Default epochs harus 30")
        self.assertEqual(parsed.patience, 10, "Default patience harus 10")
        self.assertEqual(parsed.batch, 16, "Default batch harus 16")
        self.assertEqual(parsed.imgsz, 640, "Default imgsz harus 640")
        self.assertEqual(parsed.workers, 2, "Default workers harus 2")
        self.assertTrue(parsed.amp, "Default AMP harus aktif (True)")
        self.assertFalse(parsed.plots, "Default plots harus nonaktif (False)")
        self.assertEqual(parsed.close_mosaic, 6, "Default close_mosaic harus 6")
        self.assertEqual(parsed.optimizer, "auto", "Default optimizer harus 'auto'")

    def test_cli_boolean_flag_overrides(self):
        # Uji override flag boolean --no-amp dan --plots
        orig_argv = sys.argv
        sys.argv = [
            "stage_8_4_run_ablation_experiments.py",
            "--no-amp",
            "--plots",
            "--epochs", "25",
            "--patience", "5",
            "--workers", "4",
            "--close-mosaic", "4",
        ]
        try:
            parsed = parse_args()
        finally:
            sys.argv = orig_argv

        self.assertFalse(parsed.amp, "--no-amp harus menonaktifkan AMP")
        self.assertTrue(parsed.plots, "--plots harus mengaktifkan visualisasi plots")
        self.assertEqual(parsed.epochs, 25)
        self.assertEqual(parsed.patience, 5)
        self.assertEqual(parsed.workers, 4)
        self.assertEqual(parsed.close_mosaic, 4)

    def test_invalid_cli_arguments_rejected(self):
        # Memastikan nilai tidak valid ditolak sebelum training
        bad_args_cases = [
            argparse.Namespace(epochs=0, batch=16, imgsz=640, workers=2, patience=10, close_mosaic=6),
            argparse.Namespace(epochs=30, batch=0, imgsz=640, workers=2, patience=10, close_mosaic=6),
            argparse.Namespace(epochs=30, batch=16, imgsz=16, workers=2, patience=10, close_mosaic=6),
            argparse.Namespace(epochs=30, batch=16, imgsz=640, workers=-1, patience=10, close_mosaic=6),
            argparse.Namespace(epochs=30, batch=16, imgsz=640, workers=2, patience=-1, close_mosaic=6),
            argparse.Namespace(epochs=30, batch=16, imgsz=640, workers=2, patience=10, close_mosaic=-1),
            argparse.Namespace(epochs=30, batch=16, imgsz=640, workers=2, patience=10, close_mosaic=35),
            argparse.Namespace(epochs=30, batch=16, imgsz=640, workers=2, patience=10, close_mosaic=6, aug_prob=1.5),
        ]
        for bad in bad_args_cases:
            with self.assertRaises(ValueError):
                validate_training_args(bad)

    def test_run_identity_stores_all_effective_parameters(self):
        # Memastikan seluruh parameter pelatihan efektif tersimpan di identitas run
        spec = EXPERIMENT_SPECS[0]
        ident = build_experiment_run_identity(
            exp_id="EXP-0",
            spec=spec,
            script_path=__file__,
            yaml_path=Path("cassava-v7-standardized/data.yaml"),
            manifest_path=None,
            dataset_inventory_fingerprint="inv_fp_mock_123",
            pretrained_weights_path="yolo11n.pt",
            epochs=30,
            batch_size=16,
            imgsz=640,
            seed=42,
            device="0",
            workers=2,
            amp=True,
            plots=False,
            patience=10,
            close_mosaic=6,
            optimizer="auto",
            aug_prob=0.50,
            total_train_images=8980,
            total_val_images=2567,
        )
        required_keys = [
            "epochs", "patience", "batch_size", "batch", "imgsz", "seed",
            "device", "workers", "amp", "plots", "close_mosaic", "optimizer",
            "aug_prob", "dataset_inventory_fingerprint", "total_train_images",
            "total_val_images", "python_version", "torch_version", "cuda_available"
        ]
        for k in required_keys:
            self.assertIn(k, ident, f"Key '{k}' wajib tersimpan dalam run identity!")
        self.assertEqual(ident["epochs"], 30)
        self.assertEqual(ident["patience"], 10)
        self.assertEqual(ident["workers"], 2)
        self.assertTrue(ident["amp"])
        self.assertFalse(ident["plots"])
        self.assertEqual(ident["close_mosaic"], 6)


# ==============================================================================
# GROUP B: Run Identity & Strict Reuse Verification
# ==============================================================================

class TestGroupBRunIdentityAndReuse(unittest.TestCase):
    """Group B: Verifikasi bahwa modifikasi setting memicu penolakan reuse (--skip-if-done)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_8_identity_")
        self.temp_path = Path(self.temp_dir).resolve()
        self.ckpt_path = self.temp_path / "best.pt"
        self.ckpt_path.write_bytes(b"CHECKPOINT_DATA_BYTES")
        self.meta_path = self.temp_path / "metrics_summary.json"

        self.spec = EXPERIMENT_SPECS[0]
        self.base_identity = build_experiment_run_identity(
            exp_id="EXP-0",
            spec=self.spec,
            script_path=__file__,
            yaml_path=self.temp_path / "data.yaml",
            manifest_path=None,
            dataset_inventory_fingerprint="fp_base_98",
            pretrained_weights_path="yolo11n.pt",
            epochs=30,
            batch_size=16,
            imgsz=640,
            seed=42,
            device="0",
            workers=2,
            amp=True,
            plots=False,
            patience=10,
            close_mosaic=6,
            optimizer="auto",
            aug_prob=0.50,
            checkpoint_best_sha256=compute_file_sha256(self.ckpt_path),
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_metadata": self.base_identity}, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exact_match_allows_reuse(self):
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_path,
            expected_identity=self.base_identity,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertTrue(can_reuse, f"Exact match harus diizinkan reuse: {reason}")

    def test_parameter_change_rejects_reuse(self):
        # Modifikasi setiap parameter harus menyebabkan reuse ditolak
        param_variations = [
            ("epochs", 50),
            ("patience", 15),
            ("workers", 0),
            ("amp", False),
            ("plots", True),
            ("close_mosaic", 10),
            ("batch_size", 32),
            ("imgsz", 320),
            ("seed", 999),
            ("optimizer", "adamw"),
            ("dataset_inventory_fingerprint", "fp_tampered"),
        ]
        for param_name, new_val in param_variations:
            tampered = dict(self.base_identity)
            tampered[param_name] = new_val
            can_reuse, reason = verify_ablation_reuse_safety(
                meta_path=self.meta_path,
                expected_identity=tampered,
                checkpoint_best_path=self.ckpt_path,
            )
            self.assertFalse(can_reuse, f"Perubahan '{param_name}'={new_val} harus ditolak untuk reuse!")
            self.assertIn("mismatch", reason.lower())

    def test_missing_identity_field_causes_fail_closed(self):
        # Jika metadata JSON lama kehilangan field penting, reuse harus gagal tertutup
        tampered = dict(self.base_identity)
        tampered["custom_new_field"] = "required_value"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_path,
            expected_identity=tampered,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("mismatch", reason.lower())

    def test_partial_run_rejected_for_reuse(self):
        # Run parsial (summary tanpa best.pt atau best.pt tanpa summary) harus ditolak
        partial_dir = self.temp_path / "partial_run_test"
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "metrics_summary.json").write_text("{}")

        # Kasus 1: metadata tidak ada
        missing_best = partial_dir / "weights" / "best.pt"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=partial_dir / "run_metadata.json",
            expected_identity=self.base_identity,
            checkpoint_best_path=missing_best,
        )
        self.assertFalse(can_reuse)
        self.assertIn("not found", reason.lower())

        # Kasus 2: metadata ada tapi best.pt tidak ada
        (partial_dir / "run_metadata.json").write_text(json.dumps({"run_identity": self.base_identity}))
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=partial_dir / "run_metadata.json",
            expected_identity=self.base_identity,
            checkpoint_best_path=missing_best,
        )
        self.assertFalse(can_reuse)
        self.assertIn("checkpoint best.pt does not exist", reason.lower())

        # Direktori yang sudah berisi artefak juga ditolak jika clean run diminta
        with self.assertRaises(FileExistsError):
            validate_output_directory_safety(partial_dir, stage_name="stage_8_4")

    def test_validation_content_change_rejects_reuse(self):
        # Modifikasi konten validasi harus mengubah inventory fingerprint dan menolak reuse
        tampered = dict(self.base_identity)
        tampered["dataset_inventory_fingerprint"] = "tampered_validation_fp_777"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_path,
            expected_identity=tampered,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("dataset_inventory_fingerprint mismatch", reason)


# ==============================================================================
# GROUP C: Dataset Integrity Preflight Verification
# ==============================================================================

class TestGroupCDatasetIntegrity(unittest.TestCase):
    """Group C: Pengujian fail-fast validator integritas dataset pada jalur preflight."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_8_integrity_")
        self.temp_path = Path(self.temp_dir).resolve()
        self.img_dir = self.temp_path / "images"
        self.lbl_dir = self.temp_path / "labels"
        self.img_dir.mkdir()
        self.lbl_dir.mkdir()

        # Buat dataset valid: 2 citra beranotasi + 1 background image
        for i in range(2):
            (self.img_dir / f"leaf_{i}.jpg").write_bytes(b"IMG_CONTENT_" + str(i).encode())
            (self.lbl_dir / f"leaf_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        # 1 background image (label 0 byte)
        (self.img_dir / "leaf_bg.jpg").write_bytes(b"IMG_CONTENT_BG")
        (self.lbl_dir / "leaf_bg.txt").write_text("")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_dataset_passes_integrity(self):
        result = validate_yolo_dataset_integrity(
            images_dir=self.img_dir,
            labels_dir=self.lbl_dir,
            num_classes=5,
            allow_empty_labels=True,
        )
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["total_images"], 3)
        self.assertEqual(result["background_images"], 1)
        self.assertEqual(result["total_objects"], 2)

    def test_missing_label_file_rejected(self):
        (self.img_dir / "leaf_missing_lbl.jpg").write_bytes(b"DATA")
        with self.assertRaises((FileNotFoundError, ValueError)):
            validate_yolo_dataset_integrity(
                images_dir=self.img_dir,
                labels_dir=self.lbl_dir,
            )

    def test_orphan_label_file_rejected(self):
        (self.lbl_dir / "orphan_label.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaises(ValueError) as ctx:
            validate_yolo_dataset_integrity(
                images_dir=self.img_dir,
                labels_dir=self.lbl_dir,
            )
        self.assertIn("orphan label", str(ctx.exception).lower())

    def test_invalid_class_id_rejected(self):
        (self.lbl_dir / "leaf_0.txt").write_text("99 0.5 0.5 0.2 0.2\n")
        with self.assertRaises(ValueError) as ctx:
            validate_yolo_dataset_integrity(
                images_dir=self.img_dir,
                labels_dir=self.lbl_dir,
                num_classes=5,
            )
        self.assertIn("exceeds valid range", str(ctx.exception).lower())

    def test_zero_dimension_bbox_rejected(self):
        (self.lbl_dir / "leaf_0.txt").write_text("0 0.5 0.5 0.0 0.2\n")  # width = 0.0
        with self.assertRaises(ValueError) as ctx:
            validate_yolo_dataset_integrity(
                images_dir=self.img_dir,
                labels_dir=self.lbl_dir,
                num_classes=5,
            )
        self.assertIn("invalid bbox width", str(ctx.exception).lower())

    def test_txt_subset_resolution_preserves_exact_list(self):
        # Split berbasis file list .txt hanya memeriksa citra yang terdaftar
        txt_list = self.temp_path / "train_subset.txt"
        subset_images = [self.img_dir / "leaf_0.jpg", self.img_dir / "leaf_bg.jpg"]
        txt_list.write_text("\n".join(str(p) for p in subset_images) + "\n")

        yaml_p = self.temp_path / "data.yaml"
        yaml_p.write_text(f"path: {self.temp_path}\ntrain: {txt_list.name}\nval: {txt_list.name}\nnc: 5\nnames: [CBB, CBSD, CGM, CMD, HEALTHY]\n")

        res_dir, res_lbl, res_imgs = resolve_data_yaml_split(yaml_p, "train")
        self.assertEqual(len(res_imgs), 2, "Resolver tidak boleh memperluas .txt menjadi seluruh direktori!")
        self.assertEqual({p.name for p in res_imgs}, {"leaf_0.jpg", "leaf_bg.jpg"})

        # Validator pada image_files subset harus lolos
        result = validate_yolo_dataset_integrity(
            images_dir=self.img_dir,
            labels_dir=self.lbl_dir,
            image_files=res_imgs,
            num_classes=5,
        )
        self.assertEqual(result["total_images"], 2)

    def test_fingerprint_detects_same_size_byte_modification(self):
        # Modifikasi 1 byte dengan ukuran file tetap harus mengubah fingerprint
        fp_before = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        target_file = self.img_dir / "leaf_0.jpg"
        content = bytearray(target_file.read_bytes())
        content[0] = (content[0] + 1) % 256  # Ubah 1 byte tanpa ubah ukuran
        target_file.write_bytes(bytes(content))
        fp_after = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        self.assertNotEqual(fp_before, fp_after, "Fingerprint harus mendeteksi modifikasi byte dengan ukuran sama!")

    def test_empty_split_rejected(self):
        # Split kosong harus ditolak saat allow_empty=False
        empty_im = self.temp_path / "empty_split" / "images"
        empty_lb = self.temp_path / "empty_split" / "labels"
        empty_im.mkdir(parents=True)
        empty_lb.mkdir(parents=True)
        empty_yaml = self.temp_path / "empty.yaml"
        empty_yaml.write_text(f"path: {self.temp_path}\ntrain: empty_split/images\nnc: 5\nnames: [CBB, CBSD, CGM, CMD, HEALTHY]\n")
        with self.assertRaises(ValueError) as ctx:
            resolve_data_yaml_split(empty_yaml, "train", allow_empty=False)
        self.assertTrue("no valid images found" in str(ctx.exception).lower() or "kosong" in str(ctx.exception).lower())

    def test_validator_called_by_stage_8_4_preflight(self):
        # Verifikasi bahwa validator integritas benar-benar dipanggil di preflight Stage 8.4
        corrupt_ds = self.temp_path / "corrupt_ds"
        corrupt_train_im = corrupt_ds / "train" / "images"
        corrupt_train_lb = corrupt_ds / "train" / "labels"
        corrupt_val_im = corrupt_ds / "val" / "images"
        corrupt_val_lb = corrupt_ds / "val" / "labels"
        for d in [corrupt_train_im, corrupt_train_lb, corrupt_val_im, corrupt_val_lb]:
            d.mkdir(parents=True)

        im = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(corrupt_train_im / "bad.jpg"), im)
        cv2.imwrite(str(corrupt_val_im / "val.jpg"), im)
        (corrupt_train_lb / "bad.txt").write_text("0 NOT_A_FLOAT 0.5 0.2 0.2\n")
        (corrupt_val_lb / "val.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        bad_yaml = corrupt_ds / "bad.yaml"
        bad_yaml.write_text(f"path: {corrupt_ds}\ntrain: train/images\nval: val/images\nnc: 5\nnames: [CBB, CBSD, CGM, CMD, HEALTHY]\n")

        args = argparse.Namespace(
            data_yaml=str(bad_yaml),
            manifest_path=None,
            output_dir=str(self.temp_path / "bad_out"),
            epochs=1,
            batch=2,
            imgsz=64,
            seed=42,
            device="cpu",
            skip_if_done=False,
            dry_run=True,
        )
        with self.assertRaises(ValueError) as ctx:
            run_single_experiment(0, args, trainer_cls=MockTrainer)
        self.assertIn("non-numeric coordinate", str(ctx.exception).lower())


# ==============================================================================
# GROUP D: Ablation Configuration Consistency & Mock Orchestration
# ==============================================================================

class TestGroupDAblationConfiguration(unittest.TestCase):
    """Group D: Verifikasi konsistensi setting global 4 eksperimen dan orkestrasi."""

    def setUp(self):
        MockTrainer.instances.clear()
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_8_ablation_")
        self.temp_path = Path(self.temp_dir).resolve()

        # Buat dataset sintetis
        self.ds_dir = self.temp_path / "mock_dataset"
        self.train_im = self.ds_dir / "train" / "images"
        self.train_lb = self.ds_dir / "train" / "labels"
        self.val_im = self.ds_dir / "val" / "images"
        self.val_lb = self.ds_dir / "val" / "labels"
        for d in [self.train_im, self.train_lb, self.val_im, self.val_lb]:
            d.mkdir(parents=True)

        self.train_names = [f"leaf_{i:03d}.jpg" for i in range(4)]
        for name in self.train_names:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(self.train_im / name), img)
            (self.train_lb / name.replace(".jpg", ".txt")).write_text("0 0.5 0.5 0.2 0.2\n")

        val_name = "val_leaf_001.jpg"
        cv2.imwrite(str(self.val_im / val_name), np.zeros((64, 64, 3), dtype=np.uint8))
        (self.val_lb / val_name.replace(".jpg", ".txt")).write_text("0 0.5 0.5 0.2 0.2\n")

        self.yaml_p = self.temp_path / "data.yaml"
        with open(self.yaml_p, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "path": str(self.ds_dir),
                "train": "train/images",
                "val": "val/images",
                "nc": 5,
                "names": {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "HEALTHY"},
            }, f)

        # Buat raw OOF mock & manifest mock dengan provenance 2.0
        self.raw_oof_p = self.temp_path / "raw_oof.csv"
        self.raw_oof_p.write_text("image_id,pred\nleaf_000.jpg,0\n")
        raw_oof_sha = compute_file_sha256(self.raw_oof_p)

        self.manifest_p = self.temp_path / "difficulty_manifest.csv"
        fieldnames = [
            "image_id", "d_image", "p_sampling", "n_gt_objects",
            "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box",
            "box_format", "img_width", "img_height"
        ]
        with open(self.manifest_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name in self.train_names:
                writer.writerow({
                    "image_id": name,
                    "d_image": 0.5,
                    "p_sampling": 0.25,
                    "n_gt_objects": 1,
                    "has_gt": 1,
                    "has_target_small": "0",
                    "target_gt_id": 0,
                    "target_small_di": 0.5,
                    "target_small_box": "0.4,0.4,0.6,0.6",
                    "box_format": "norm_xyxy",
                    "img_width": 64,
                    "img_height": 64,
                })

        sidecar_p = self.manifest_p.parent / (self.manifest_p.stem + ".provenance.json")
        sidecar_data = {
            "provenance_schema_version": "2.0",
            "evaluator_version": "Stage-9.7-Canonical-v2",
            "clean_oof_run_id": "clean_oof_run_123",
            "manifest_csv_sha256": compute_file_sha256(self.manifest_p),
            "raw_oof_sha256": raw_oof_sha,
            "raw_oof_path": str(self.raw_oof_p),
            "dataset_inventory_fingerprint": compute_dataset_inventory_fingerprint(
                self.train_im, self.train_lb, data_yaml_path=self.yaml_p
            ),
            "data_yaml_sha256": compute_file_sha256(self.yaml_p),
            "total_samples": 4,
            "num_hard": 0,
        }
        sidecar_p.write_text(json.dumps(sidecar_data))

        self.out_dir = self.temp_path / "ablation_run"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_experiment_spec_toggles(self):
        # EXP-0: baseline uniform + standard aug
        self.assertFalse(EXPERIMENT_SPECS[0]["enable_mixture_sampling"])
        self.assertFalse(EXPERIMENT_SPECS[0]["enable_targeted_aug"])
        # EXP-1: adaptive sampling only
        self.assertTrue(EXPERIMENT_SPECS[1]["enable_mixture_sampling"])
        self.assertFalse(EXPERIMENT_SPECS[1]["enable_targeted_aug"])
        # EXP-2: targeted context augmentation only
        self.assertFalse(EXPERIMENT_SPECS[2]["enable_mixture_sampling"])
        self.assertTrue(EXPERIMENT_SPECS[2]["enable_targeted_aug"])
        # EXP-3: full proposed (sampling + aug)
        self.assertTrue(EXPERIMENT_SPECS[3]["enable_mixture_sampling"])
        self.assertTrue(EXPERIMENT_SPECS[3]["enable_targeted_aug"])

    def test_ablation_orchestrator_exp_all_mock_execution(self):
        args = argparse.Namespace(
            data_yaml=str(self.yaml_p),
            manifest_path=str(self.manifest_p),
            output_dir=str(self.out_dir),
            exp="all",
            epochs=2,
            patience=5,
            batch=2,
            imgsz=64,
            device="cpu",
            workers=2,
            amp=True,
            plots=False,
            close_mosaic=1,
            optimizer="auto",
            aug_prob=0.50,
            pretrained_weights="yolo11n.pt",
            seed=42,
            skip_if_done=False,
            dry_run=False,
            clean_oof_run_id="clean_oof_run_123",
        )
        summaries = run_ablation_pipeline(
            args,
            trainer_cls=MockTrainer,
            model_eval_fn=mock_model_eval_fn,
        )
        self.assertEqual(len(summaries), 4)
        self.assertEqual([s["exp_id"] for s in summaries], ["EXP-0", "EXP-1", "EXP-2", "EXP-3"])

        # Periksa bahwa seluruh trainer instance menerima konfigurasi global yang sama persis
        self.assertEqual(len(MockTrainer.instances), 4)
        for inst in MockTrainer.instances:
            self.assertEqual(inst.overrides["epochs"], 2)
            self.assertEqual(inst.overrides["patience"], 5)
            self.assertEqual(inst.overrides["batch"], 2)
            self.assertEqual(inst.overrides["imgsz"], 64)
            self.assertEqual(inst.overrides["workers"], 2)
            self.assertTrue(inst.overrides["amp"])
            self.assertFalse(inst.overrides["plots"])
            self.assertEqual(inst.overrides["close_mosaic"], 1)
            self.assertEqual(inst.overrides["optimizer"], "auto")

    def test_single_experiment_failure_halts_pipeline(self):
        args = argparse.Namespace(
            data_yaml=str(self.yaml_p),
            manifest_path=str(self.manifest_p),
            output_dir=str(self.out_dir),
            exp="all",
            epochs=2,
            patience=5,
            batch=2,
            imgsz=64,
            device="cpu",
            workers=2,
            amp=True,
            plots=False,
            close_mosaic=1,
            optimizer="auto",
            aug_prob=0.50,
            pretrained_weights="yolo11n.pt",
            seed=42,
            skip_if_done=False,
            dry_run=False,
        )
        with self.assertRaises(RuntimeError):
            run_ablation_pipeline(
                args,
                trainer_cls=FailingMockTrainer,
                model_eval_fn=mock_model_eval_fn,
            )
        # Laporan gabungan ABLATION_EXPERIMENTS_REPORT.md tidak boleh tercipta jika gagal
        report_p = self.out_dir / "ABLATION_EXPERIMENTS_REPORT.md"
        self.assertFalse(report_p.exists(), "Laporan komparasi tidak boleh dibuat saat eksperimen gagal!")


# ==============================================================================
# GROUP E: Data Loading, Augmentation Mechanics & Pretrained Resolution
# ==============================================================================

class TestGroupEDataLoadingAndAugmentation(unittest.TestCase):
    """Group E: Pengujian targeted context augmentation dan resolusi pretrained weights."""

    def test_pretrained_weights_resolution(self):
        # Bobot lokal konkret yolo11n.pt harus berhasil di-resolve
        resolved = resolve_pretrained_weights("yolo11n.pt")
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.name, "yolo11n.pt")

    def test_nonexistent_weights_raises_file_not_found(self):
        # Bobot fiktif tidak boleh diunduh otomatis
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_pretrained_weights("non_existent_weights_v999.pt")
        self.assertIn("dilarang keras", str(ctx.exception).lower())

    def test_stage_8_1_pretrained_weights_argument_and_fail_closed(self):
        # Stage 8.1 harus mendukung --pretrained-weights dan menolak file nonexistent
        import stage_8_1_group_oof_profiling as s81
        orig_argv = sys.argv
        sys.argv = ["stage_8_1_group_oof_profiling.py", "--pretrained-weights", "nonexistent_model.pt"]
        try:
            parsed = s81.parse_args()
            self.assertEqual(parsed.pretrained_weights, "nonexistent_model.pt")
        finally:
            sys.argv = orig_argv

    def test_context_crop_target_imgsz_fallback(self):
        # Verifikasi fallback target_imgsz ke self.imgsz saat hyp.imgsz None
        from ultralytics.utils.instance import Instances
        dataset = AdaptiveYOLODataset.__new__(AdaptiveYOLODataset)
        dataset.imgsz = 640
        dataset.context_margin = 2.0
        dataset.vis_thresh = 0.30
        dataset.hyp = argparse.Namespace()  # hyp without imgsz attribute
        dataset.crop_diagnostics = {
            "crop_attempts": 0,
            "crops_applied": 0,
            "no_target_skips": 0,
            "invalid_box_skips": 0,
            "visibility_skips": 0,
        }

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        instances = Instances(
            bboxes=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            keypoints=None,
            bbox_format="xywh",
            normalized=True,
        )
        dummy_label = {
            "img": dummy_img,
            "instances": instances,
            "cls": np.array([[0]], dtype=np.float32),
            "bboxes": np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
            "resized_shape": (640, 640),
        }
        target_info = {
            "has_target_small": True,
            "target_small_box": "0.45,0.45,0.55,0.55",
            "box_format": "norm_xyxy",
            "img_width": 640,
            "img_height": 640,
            "target_gt_id": 0,
        }
        res_label = dataset.apply_context_crop(dummy_label, target_info)
        self.assertEqual(res_label["img"].shape, (640, 640, 3))
        self.assertEqual(dataset.crop_diagnostics["crops_applied"], 1)

    def test_collate_fn_with_mixed_cropped_and_uncropped_items(self):
        # Pengujian collate_fn AdaptiveYOLODataset dengan item campuran
        item1 = {
            "img": torch.zeros((3, 640, 640), dtype=torch.float32),
            "cls": torch.tensor([[0], [1]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "batch_idx": torch.zeros((2,), dtype=torch.int64),
            "im_file": "img1.jpg",
            "resized_shape": (640, 640),
            "ori_shape": (640, 640),
        }
        item2 = {
            "img": torch.ones((3, 640, 640), dtype=torch.float32),
            "cls": torch.tensor([[2]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.4, 0.4, 0.3, 0.3]], dtype=torch.float32),
            "batch_idx": torch.zeros((1,), dtype=torch.int64),
            "im_file": "img2_cropped.jpg",
            "resized_shape": (640, 640),
            "ori_shape": (640, 640),
        }
        collated = AdaptiveYOLODataset.collate_fn([item1, item2])
        self.assertEqual(collated["img"].shape, (2, 3, 640, 640))
        self.assertEqual(collated["bboxes"].shape, (3, 4))
        self.assertEqual(collated["cls"].shape, (3, 1))
        self.assertEqual(collated["batch_idx"].tolist(), [0, 0, 1])


# ==============================================================================
# GROUP F: Output Safety, Historical Protection & Zero Dataset Mutation
# ==============================================================================

class TestGroupFOutputSafetyAndZeroMutation(unittest.TestCase):
    """Group F: Pengujian penolakan folder historis dan nol mutasi dataset."""

    def test_all_forbidden_historical_directories_rejected(self):
        # Memastikan setiap folder historis ditolak keras
        for forbidden in FORBIDDEN_HISTORICAL_DIRS:
            target = Path(forbidden)
            with self.assertRaises(PermissionError):
                validate_output_directory_safety(target, stage_name="stage_8_4")

            # Subdirektori anak juga wajib ditolak
            sub_target = Path(forbidden) / "child_experiment"
            with self.assertRaises(PermissionError):
                validate_output_directory_safety(sub_target, stage_name="stage_8_4")

    def test_clean_output_directory_permitted(self):
        temp_dir = tempfile.mkdtemp(prefix="test_stage_9_8_clean_")
        try:
            target = Path(temp_dir) / "clean_ablation_output"
            # Direktori baru harus lolos validasi tanpa error
            validate_output_directory_safety(target, stage_name="stage_8_4")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_zero_mutation_dataset_verification(self):
        # Memeriksa bahwa direktori standardized dataset tetap tidak termutasi
        data_yaml = Path("cassava-v7-standardized/data.yaml")
        if data_yaml.exists():
            split_info = resolve_data_yaml_split(data_yaml, "train")
            train_images_dir = split_info["images_dir"]
            # Pastikan tidak ada file .cache yang dibuat langsung di folder images
            cache_files = list(train_images_dir.glob("*.cache"))
            self.assertEqual(len(cache_files), 0, "Tidak boleh ada file .cache yang tertinggal di direktori citra penelitian!")

    def test_verify_dataset_files_and_checksums_unmutated_before_after(self):
        # Memastikan checksum dan fingerprint dataset persis sama sebelum dan sesudah verifikasi integritas
        temp_dir = tempfile.mkdtemp(prefix="test_stage_9_8_unmutated_")
        try:
            temp_path = Path(temp_dir)
            im_dir = temp_path / "images"
            lb_dir = temp_path / "labels"
            im_dir.mkdir(parents=True)
            lb_dir.mkdir(parents=True)

            # Buat sample dataset
            im = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(im_dir / "leaf_0.jpg"), im)
            cv2.imwrite(str(im_dir / "leaf_1.jpg"), im)
            (lb_dir / "leaf_0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (lb_dir / "leaf_1.txt").write_text("1 0.3 0.3 0.1 0.1\n")

            fp_before = compute_dataset_inventory_fingerprint(im_dir, lb_dir)

            # Jalankan validasi integritas (harus read-only)
            validate_yolo_dataset_integrity(
                images_dir=im_dir,
                labels_dir=lb_dir,
                num_classes=5,
            )

            fp_after = compute_dataset_inventory_fingerprint(im_dir, lb_dir)
            self.assertEqual(fp_before, fp_after, "Fingerprint termutasi setelah validasi integritas!")

            # Pastikan tidak ada file cache yang dibuat selama validasi integritas
            temp_cache_files = list(temp_path.rglob("*.cache"))
            self.assertEqual(len(temp_cache_files), 0, f"Validasi integritas tidak boleh membuat cache: {temp_cache_files}")

            # Pastikan tidak ada file cache pada direktori images standardized
            std_yaml = Path("cassava-v7-standardized/data.yaml")
            if std_yaml.exists():
                split_info = resolve_data_yaml_split(std_yaml, "train")
                train_im_cache = list(split_info["images_dir"].glob("*.cache"))
                self.assertEqual(len(train_im_cache), 0, "Direktori citra tidak boleh memiliki file cache!")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_stage_8_3_get_dataloader_no_invalid_device_kwargs(self):
        """FINDING-9.9-01: Memastikan AdaptiveDetectionTrainer.get_dataloader tidak mengoper kwarg 'device' yang tidak didukung ke build_dataloader."""
        from stage_8_3_adaptive_pipeline import AdaptiveDetectionTrainer
        import inspect
        from ultralytics.data.build import build_dataloader

        sig = inspect.signature(build_dataloader)
        self.assertNotIn("device", sig.parameters, "build_dataloader tidak boleh diasumsikan memiliki parameter 'device'")

        # Verifikasi source get_dataloader tidak mengandung device=self.device
        trainer_source = inspect.getsource(AdaptiveDetectionTrainer.get_dataloader)
        self.assertNotIn("device=self.device", trainer_source, "AdaptiveDetectionTrainer.get_dataloader tidak boleh mengoper device=self.device ke build_dataloader")


if __name__ == "__main__":
    unittest.main()


#!/usr/bin/env python3
"""
Test Suite: Stage 9.7 Pipeline Safety, Provenance & End-to-End Readiness Remediation
=====================================================================================

Membuktikan secara empiris dan menyeluruh bahwa seluruh edge-case pipeline, keamanan direktori,
provenansi manifest, integritas dataset, resolusi layout data.yaml, orkestrasi ablasi
end-to-end (EXP-0 s.d. EXP-3), dan mekanisme augmentasi adaptif telah mengeras (hardened).

Groups Tested:
- Group A: Output Safety & Historical Directory Traversal Protection
- Group B: Run Identity & Safe Reuse Verification
- Group C: Manifest Provenance & 100% Split Coverage Verification
- Group D: Dataset Inventory & Structural Integrity Validation
- Group E: Canonical Data YAML Resolver & Layout Handling
- Group F: Ablation Orchestrator Mock End-to-End Integration (P0)
- Group G: Pretrained Weights, Augmentation & Dataset Mechanics
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg

from research_utils.detection_metrics import (
    CLASS_NAMES,
    compute_file_sha256,
    compute_dataset_inventory_fingerprint,
    compute_pretrained_weights_sha256,
    derive_yolo_label_path,
    resolve_data_yaml_split,
    validate_output_directory_safety,
    validate_yolo_dataset_integrity,
)
from stage_8_3_adaptive_pipeline import AdaptiveYOLODataset
from stage_8_4_run_ablation_experiments import (
    EXPERIMENT_SPECS,
    build_experiment_run_identity,
    run_ablation_pipeline,
    run_single_experiment,
    verify_ablation_reuse_safety,
    verify_manifest_provenance_and_coverage,
)


class TestGroupAOutputDirectorySafety(unittest.TestCase):
    """Group A: Pengujian proteksi direktori output dan traversal direktori historis."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_out_")
        self.temp_path = Path(self.temp_dir).resolve()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_directory_passes(self):
        clean_dir = self.temp_path / "clean_experiment_dir"
        clean_dir.mkdir(parents=True, exist_ok=True)
        is_safe, msg = validate_output_directory_safety(clean_dir, stage_name="stage_test")
        self.assertTrue(is_safe)
        self.assertIn("clean", msg.lower())

    def test_historical_target_rejected(self):
        # Target tepat pada nama direktori historis terlarang
        hist_dir = self.temp_path / "stage_8_1_oof"
        hist_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises((PermissionError, RuntimeError, OSError)) as ctx:
            validate_output_directory_safety(hist_dir, stage_name="stage_test")
        self.assertIn("historical", str(ctx.exception).lower())

    def test_nested_subdirectories_rejected(self):
        # Target anak/nested di dalam direktori historis terlarang
        nested_dir = self.temp_path / "stage_8_ablation" / "exp0_baseline" / "nested_sub"
        nested_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises((PermissionError, RuntimeError, OSError)) as ctx:
            validate_output_directory_safety(nested_dir, stage_name="stage_test")
        self.assertIn("historical", str(ctx.exception).lower())

    def test_path_traversal_dotdot_rejected(self):
        # Path dengan `..` yang me-resolve ke direktori historis terlarang
        hist_parent = self.temp_path / "stage_8_2_difficulty"
        hist_parent.mkdir(parents=True, exist_ok=True)
        fake_clean = self.temp_path / "some_dir"
        fake_clean.mkdir(parents=True, exist_ok=True)
        traversal_path = fake_clean / ".." / "stage_8_2_difficulty"
        with self.assertRaises((PermissionError, RuntimeError, OSError)) as ctx:
            validate_output_directory_safety(traversal_path, stage_name="stage_test")
        self.assertIn("historical", str(ctx.exception).lower())

    def test_case_insensitivity_rejected(self):
        # Variasi kapitalisasi nama direktori historis (misal di OS case-insensitive)
        upper_dir = self.temp_path / "STAGE_8_1_OOF"
        upper_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises((PermissionError, RuntimeError, OSError)) as ctx:
            validate_output_directory_safety(upper_dir, stage_name="stage_test")
        self.assertIn("historical", str(ctx.exception).lower())

    def test_clean_sibling_permitted(self):
        # Direktori dengan prefix mirip tetapi nama berbeda (clean sibling) harus diizinkan
        clean_sibling = self.temp_path / "stage_8_ablation_clean"
        clean_sibling.mkdir(parents=True, exist_ok=True)
        is_safe, msg = validate_output_directory_safety(clean_sibling, stage_name="stage_test")
        self.assertTrue(is_safe)
        self.assertIn("clean", msg.lower())

    def test_partial_output_rejected(self):
        # Direktori non-empty tanpa artefak lengkap dan run_identity harus ditolak
        partial_dir = self.temp_path / "partial_dir"
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "random_file.tmp").write_text("unfinished")
        with self.assertRaises((FileExistsError, RuntimeError, OSError)) as ctx:
            validate_output_directory_safety(partial_dir, stage_name="stage_test", allow_reuse=False)
        self.assertIn("exists", str(ctx.exception).lower())

    def test_valid_reuse_passes(self):
        # Direktori dengan artefak lengkap dan run_identity valid diizinkan reuse jika allow_reuse=True
        reuse_dir = self.temp_path / "reuse_dir"
        reuse_dir.mkdir(parents=True, exist_ok=True)
        (reuse_dir / "weights").mkdir(parents=True, exist_ok=True)
        (reuse_dir / "weights" / "best.pt").write_bytes(b"dummy_weights")
        meta_content = {"run_id": "exp_run_123", "script_sha256": "abc"}
        (reuse_dir / "run_metadata.json").write_text(json.dumps(meta_content))

        is_safe, msg = validate_output_directory_safety(
            reuse_dir,
            stage_name="stage_test",
            allow_reuse=True,
            expected_identity={"run_id": "exp_run_123"},
            metadata_filename="run_metadata.json",
        )
        self.assertTrue(is_safe)

    def test_identity_mismatch_rejected(self):
        # Direktori dengan artefak lengkap tetapi run_identity tidak cocok dengan expected run_id
        reuse_dir = self.temp_path / "mismatch_dir"
        reuse_dir.mkdir(parents=True, exist_ok=True)
        (reuse_dir / "weights").mkdir(parents=True, exist_ok=True)
        (reuse_dir / "weights" / "best.pt").write_bytes(b"dummy_weights")
        meta_content = {"run_id": "exp_old_id", "script_sha256": "abc"}
        (reuse_dir / "run_metadata.json").write_text(json.dumps(meta_content))

        with self.assertRaises((PermissionError, RuntimeError, ValueError, OSError)) as ctx:
            validate_output_directory_safety(
                reuse_dir,
                stage_name="stage_test",
                allow_reuse=True,
                expected_identity={"run_id": "exp_new_id_expected"},
                metadata_filename="run_metadata.json",
            )
        self.assertIn("mismatch", str(ctx.exception).lower())


class TestGroupBRunIdentityVerification(unittest.TestCase):
    """Group B: Pengujian verifikasi identitas eksperimen dan proteksi reuse."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_id_")
        self.temp_path = Path(self.temp_dir).resolve()

        self.exp_dir = self.temp_path / "exp_test"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = self.exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_pt = weights_dir / "best.pt"
        self.best_pt.write_bytes(b"TEST_WEIGHTS_CONTENT_123")

        self.expected_id = {
            "run_id": "test_exp_0_abc",
            "script_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "data_yaml_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "pretrained_weights_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "dataset_inventory_fingerprint": "4444444444444444444444444444444444444444444444444444444444444444",
            "epochs": 10,
            "batch": 16,
            "seed": 42,
            "best_weight_sha256": compute_file_sha256(self.best_pt),
        }
        self.meta_path = self.exp_dir / "run_metadata.json"
        self.meta_path.write_text(json.dumps(self.expected_id))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_identity_exact_match_passes(self):
        is_safe, msg = verify_ablation_reuse_safety(self.meta_path, self.expected_id, self.best_pt)
        self.assertTrue(is_safe, f"Failed: {msg}")
        self.assertIn("verified", msg.lower())

    def test_identity_detects_script_modification(self):
        modified_id = dict(self.expected_id)
        modified_id["script_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        is_safe, msg = verify_ablation_reuse_safety(self.meta_path, modified_id, self.best_pt)
        self.assertFalse(is_safe)
        self.assertIn("script_sha256", msg)

    def test_identity_detects_data_yaml_modification(self):
        modified_id = dict(self.expected_id)
        modified_id["data_yaml_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        is_safe, msg = verify_ablation_reuse_safety(self.meta_path, modified_id, self.best_pt)
        self.assertFalse(is_safe)
        self.assertIn("data_yaml_sha256", msg)

    def test_identity_detects_pretrained_weights_modification(self):
        modified_id = dict(self.expected_id)
        modified_id["pretrained_weights_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        is_safe, msg = verify_ablation_reuse_safety(self.meta_path, modified_id, self.best_pt)
        self.assertFalse(is_safe)
        self.assertIn("pretrained_weights_sha256", msg)


class TestGroupCManifestProvenanceAndCoverage(unittest.TestCase):
    """Group C: Pengujian integritas manifest dan cakupan 100% split pelatihan."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_manifest_")
        self.temp_path = Path(self.temp_dir).resolve()

        # Buat dataset sintetis train split
        self.ds_dir = self.temp_path / "mock_dataset"
        self.train_im = self.ds_dir / "train" / "images"
        self.train_lb = self.ds_dir / "train" / "labels"
        self.train_im.mkdir(parents=True)
        self.train_lb.mkdir(parents=True)

        self.image_names = [f"image_{i:03d}.jpg" for i in range(4)]
        self.expected_image_files = []
        for img_name in self.image_names:
            img_p = self.train_im / img_name
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_p), img)
            (self.train_lb / img_name.replace(".jpg", ".txt")).write_text("0 0.5 0.5 0.2 0.2\n")
            self.expected_image_files.append(img_p)

        self.yaml_path = self.temp_path / "data.yaml"
        yaml_content = {
            "path": str(self.ds_dir),
            "train": "train/images",
            "val": "train/images",
            "names": {0: "CBB"},
        }
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        # Buat raw OOF mock
        self.raw_oof_path = self.temp_path / "raw_oof_predictions.csv"
        self.raw_oof_path.write_text("image_id,pred\nimage_000.jpg,0\n")
        self.raw_oof_sha = compute_file_sha256(self.raw_oof_path)

        # Buat manifest CSV yang valid (4 baris, sum=1.0)
        self.manifest_path = self.temp_path / "difficulty_manifest.csv"
        fieldnames = [
            "image_id", "d_image", "p_sampling", "n_gt_objects",
            "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box",
            "box_format", "img_width", "img_height"
        ]
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name in self.image_names:
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
                    "img_width": 100,
                    "img_height": 100,
                })

        self.manifest_sha = compute_file_sha256(self.manifest_path)
        self.data_yaml_sha = compute_file_sha256(self.yaml_path)
        self.dataset_fingerprint = compute_dataset_inventory_fingerprint(
            self.train_im, self.train_lb, data_yaml_path=self.yaml_path
        )

        # Sidecar JSON yang valid (.provenance.json)
        self.sidecar_path = self.manifest_path.parent / (self.manifest_path.stem + ".provenance.json")
        self.valid_sidecar = {
            "provenance_schema_version": "2.0",
            "evaluator_version": "Stage-9.7-Canonical-v2",
            "clean_oof_run_id": "clean_oof_run_999",
            "manifest_csv_sha256": self.manifest_sha,
            "raw_oof_sha256": self.raw_oof_sha,
            "raw_oof_path": str(self.raw_oof_path),
            "dataset_inventory_fingerprint": self.dataset_fingerprint,
            "data_yaml_sha256": self.data_yaml_sha,
            "total_samples": 4,
            "num_hard": 0,
        }
        self.sidecar_path.write_text(json.dumps(self.valid_sidecar))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_manifest_and_provenance_passes(self):
        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            self.manifest_path,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertTrue(is_valid, f"Verification failed: {msg}")
        self.assertIn("100%", msg)

    def test_manifest_tampered_csv_rejected(self):
        # Ubah isi manifest CSV tanpa mengupdate sidecar
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write("tampered_byte\n")
        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            self.manifest_path,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertFalse(is_valid)
        self.assertIn("mismatch", msg.lower())

    def test_manifest_tampered_raw_oof_rejected(self):
        # Ubah isi file raw OOF prediksi
        self.raw_oof_path.write_text("image_id,pred\nTAMPERED_OOF\n")
        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            self.manifest_path,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertFalse(is_valid)
        self.assertIn("raw oof", msg.lower())

    def test_manifest_historical_oof_id_rejected(self):
        # Ubah clean_oof_run_id menjadi direktori/id historis terlarang
        bad_sidecar = dict(self.valid_sidecar)
        bad_sidecar["clean_oof_run_id"] = "stage_8_1_oof"
        self.sidecar_path.write_text(json.dumps(bad_sidecar))
        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            self.manifest_path,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertFalse(is_valid)
        self.assertIn("historical", msg.lower())

    def test_manifest_missing_or_extra_coverage_rejected(self):
        # Manifest hanya memiliki 3 baris (kurang 1 dari train set)
        bad_manifest = self.temp_path / "bad_cov_manifest.csv"
        fieldnames = ["image_id", "d_image", "p_sampling", "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box"]
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(3):
                writer.writerow({
                    "image_id": self.image_names[i],
                    "d_image": 0.5,
                    "p_sampling": 0.333333,
                    "has_gt": 1,
                    "has_target_small": "0",
                    "target_gt_id": 0,
                    "target_small_di": 0.5,
                    "target_small_box": "0,0,1,1",
                })

        bad_sidecar_p = bad_manifest.parent / (bad_manifest.stem + ".provenance.json")
        sidecar = dict(self.valid_sidecar)
        sidecar["manifest_csv_sha256"] = compute_file_sha256(bad_manifest)
        bad_sidecar_p.write_text(json.dumps(sidecar))

        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            bad_manifest,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertFalse(is_valid)
        self.assertIn("missing", msg.lower())

    def test_manifest_duplicate_image_ids_rejected(self):
        # Manifest memiliki duplikasi image_id
        dup_manifest = self.temp_path / "dup_manifest.csv"
        fieldnames = ["image_id", "d_image", "p_sampling", "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box"]
        with open(dup_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name in [self.image_names[0], self.image_names[0], self.image_names[1], self.image_names[2]]:
                writer.writerow({
                    "image_id": name,
                    "d_image": 0.5,
                    "p_sampling": 0.25,
                    "has_gt": 1,
                    "has_target_small": "0",
                    "target_gt_id": 0,
                    "target_small_di": 0.5,
                    "target_small_box": "0,0,1,1",
                })

        dup_sidecar_p = dup_manifest.parent / (dup_manifest.stem + ".provenance.json")
        sidecar = dict(self.valid_sidecar)
        sidecar["manifest_csv_sha256"] = compute_file_sha256(dup_manifest)
        dup_sidecar_p.write_text(json.dumps(sidecar))

        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            dup_manifest,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertFalse(is_valid)
        self.assertIn("duplicate", msg.lower())

    def test_manifest_invalid_sampling_probs_rejected(self):
        # Probabilitas sampling non-positif
        bad_prob_manifest = self.temp_path / "bad_prob_manifest.csv"
        fieldnames = ["image_id", "d_image", "p_sampling", "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box"]
        with open(bad_prob_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"image_id": self.image_names[0], "d_image": 0.5, "p_sampling": 0.0, "has_gt": 1, "has_target_small": "0", "target_gt_id": 0, "target_small_di": 0.5, "target_small_box": "0,0,1,1"})
            for i in range(1, 4):
                writer.writerow({"image_id": self.image_names[i], "d_image": 0.5, "p_sampling": 0.333333, "has_gt": 1, "has_target_small": "0", "target_gt_id": 0, "target_small_di": 0.5, "target_small_box": "0,0,1,1"})

        bad_sidecar_p = bad_prob_manifest.parent / (bad_prob_manifest.stem + ".provenance.json")
        sidecar = dict(self.valid_sidecar)
        sidecar["manifest_csv_sha256"] = compute_file_sha256(bad_prob_manifest)
        bad_sidecar_p.write_text(json.dumps(sidecar))

        is_valid, msg, prov = verify_manifest_provenance_and_coverage(
            bad_prob_manifest,
            self.dataset_fingerprint,
            self.data_yaml_sha,
            self.expected_image_files,
        )
        self.assertFalse(is_valid)
        self.assertIn("p_sampling", msg.lower())


class TestGroupDDatasetInventoryAndIntegrity(unittest.TestCase):
    """Group D: Pengujian validasi inventaris konten citra dan struktur dataset YOLO."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_ds_")
        self.temp_path = Path(self.temp_dir).resolve()

        self.img_dir = self.temp_path / "images"
        self.lbl_dir = self.temp_path / "labels"
        self.img_dir.mkdir(parents=True)
        self.lbl_dir.mkdir(parents=True)

        # Buat 2 citra dan 2 label
        self.im1 = self.img_dir / "im1.jpg"
        self.im2 = self.img_dir / "im2.jpg"
        self.im1.write_bytes(b"A" * 100)
        self.im2.write_bytes(b"B" * 100)

        self.lb1 = self.lbl_dir / "im1.txt"
        self.lb2 = self.lbl_dir / "im2.txt"
        self.lb1.write_text("0 0.5 0.5 0.2 0.2\n")
        self.lb2.write_text("1 0.3 0.3 0.1 0.1\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_image_byte_modification_same_size_detected(self):
        # Modifikasi byte citra dengan ukuran file persis sama harus menghasilkan fingerprint berbeda
        fp_before = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        # Ganti byte dari 'A'*100 menjadi 'Z'*100 (ukuran sama persis 100 byte)
        self.im1.write_bytes(b"Z" * 100)
        fp_after = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        self.assertNotEqual(fp_before, fp_after, "Fingerprint harus mendeteksi perubahan isi byte citra!")

    def test_label_byte_modification_detected(self):
        # Modifikasi byte label harus mengubah fingerprint
        fp_before = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        self.lb1.write_text("0 0.5 0.5 0.2 0.3\n")
        fp_after = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        self.assertNotEqual(fp_before, fp_after, "Fingerprint harus mendeteksi perubahan isi byte label!")

    def test_validation_split_content_change_detected(self):
        # Modifikasi pada split validasi harus mengubah dataset inventory fingerprint
        val_im_dir = self.temp_path / "val_images"
        val_lb_dir = self.temp_path / "val_labels"
        val_im_dir.mkdir()
        val_lb_dir.mkdir()
        val_im = val_im_dir / "val1.jpg"
        val_im.write_bytes(b"V" * 50)
        val_lb = val_lb_dir / "val1.txt"
        val_lb.write_text("0 0.4 0.4 0.1 0.1\n")

        yaml_path = self.temp_path / "ds.yaml"
        yaml_content = {
            "path": str(self.temp_path),
            "train": str(self.img_dir),
            "val": str(val_im_dir),
            "names": {0: "CBB"},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        fp_before = compute_dataset_inventory_fingerprint(
            self.img_dir, self.lbl_dir, data_yaml_path=yaml_path
        )
        val_im.write_bytes(b"W" * 50)
        fp_after = compute_dataset_inventory_fingerprint(
            self.img_dir, self.lbl_dir, data_yaml_path=yaml_path
        )
        self.assertNotEqual(fp_before, fp_after, "Fingerprint harus mendeteksi perubahan konten split validasi!")

    def test_validate_yolo_dataset_integrity_missing_label_rejected(self):
        # Citra tanpa file label harus ditolak
        (self.img_dir / "orphan_img.jpg").write_bytes(b"C" * 100)
        with self.assertRaises((FileNotFoundError, ValueError)) as ctx:
            validate_yolo_dataset_integrity(self.img_dir, self.lbl_dir)
        self.assertIn("missing label file", str(ctx.exception).lower())

    def test_validate_yolo_dataset_integrity_orphan_label_rejected(self):
        # Label tanpa citra pasangan harus ditolak
        (self.lbl_dir / "orphan_lbl.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        with self.assertRaises(ValueError) as ctx:
            validate_yolo_dataset_integrity(self.img_dir, self.lbl_dir)
        self.assertIn("orphan label", str(ctx.exception).lower())

    def test_validate_yolo_dataset_integrity_empty_label_treated_as_background(self):
        # File label kosong (0-byte) adalah citra latar belakang (background image) yang sah
        bg_im = self.img_dir / "bg_img.jpg"
        bg_lb = self.lbl_dir / "bg_img.txt"
        bg_im.write_bytes(b"G" * 100)
        bg_lb.write_text("")  # 0 bytes, no objects

        # Harus lolos validasi tanpa error
        summary = validate_yolo_dataset_integrity(self.img_dir, self.lbl_dir)
        self.assertEqual(summary["total_images"], 3)
        self.assertEqual(summary["background_images"], 1)

    def test_validate_yolo_dataset_integrity_invalid_coordinates_rejected(self):
        # Label dengan koordinat di luar [0, 1] harus ditolak
        bad_lb = self.lbl_dir / "im1.txt"
        bad_lb.write_text("0 1.5 0.5 0.2 0.2\n")  # x_center > 1.0
        with self.assertRaises(ValueError) as ctx:
            validate_yolo_dataset_integrity(self.img_dir, self.lbl_dir)
        self.assertIn("outside [0.0, 1.0] range", str(ctx.exception))


class TestGroupEDataYamlResolver(unittest.TestCase):
    """Group E: Pengujian resolusi kanonis konfigurasi data.yaml dan variasi layout."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_resolver_")
        self.temp_path = Path(self.temp_dir).resolve()

        self.ds_dir = self.temp_path / "mock_dataset"
        self.train_im = self.ds_dir / "train" / "images"
        self.train_lb = self.ds_dir / "train" / "labels"
        self.train_im.mkdir(parents=True)
        self.train_lb.mkdir(parents=True)

        for i in range(3):
            (self.train_im / f"im_{i}.jpg").write_bytes(b"X" * 20)
            (self.train_lb / f"im_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolver_relative_and_absolute_paths(self):
        # data.yaml dengan path relatif vs absolut
        yaml_content_rel = {
            "path": str(self.ds_dir),
            "train": "train/images",
            "names": {0: "CBB"},
        }
        yaml_p = self.temp_path / "data_rel.yaml"
        with open(yaml_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content_rel, f)

        res_im, res_lb, img_list = resolve_data_yaml_split(yaml_p, "train")
        self.assertEqual(res_im, self.train_im.resolve())
        self.assertEqual(res_lb, self.train_lb.resolve())
        self.assertEqual(len(img_list), 3)

    def test_resolver_txt_file_subset(self):
        # data.yaml dengan split berupa .txt file list
        txt_file = self.ds_dir / "subset.txt"
        chosen_imgs = list(self.train_im.glob("*.jpg"))[:2]
        txt_file.write_text("\n".join(str(p.resolve()) for p in chosen_imgs) + "\n")

        yaml_content = {
            "path": str(self.ds_dir),
            "train": "subset.txt",
            "names": {0: "CBB"},
        }
        yaml_p = self.temp_path / "data_txt.yaml"
        with open(yaml_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        res_im, res_lb, img_list = resolve_data_yaml_split(yaml_p, "train")
        self.assertEqual(len(img_list), 2)
        self.assertTrue(all(p in chosen_imgs for p in img_list))

    def test_resolver_multi_directory_list(self):
        # data.yaml dengan split berupa list dari beberapa direktori
        train_im2 = self.ds_dir / "train2" / "images"
        train_lb2 = self.ds_dir / "train2" / "labels"
        train_im2.mkdir(parents=True)
        train_lb2.mkdir(parents=True)
        (train_im2 / "im_extra.jpg").write_bytes(b"Y" * 20)
        (train_lb2 / "im_extra.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        yaml_content = {
            "path": str(self.ds_dir),
            "train": ["train/images", "train2/images"],
            "names": {0: "CBB"},
        }
        yaml_p = self.temp_path / "data_multi.yaml"
        with open(yaml_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        res_im, res_lb, img_list = resolve_data_yaml_split(yaml_p, "train")
        self.assertEqual(len(img_list), 4)

    def test_resolver_empty_split_raises(self):
        empty_dir = self.ds_dir / "empty" / "images"
        empty_dir.mkdir(parents=True)
        (self.ds_dir / "empty" / "labels").mkdir(parents=True)
        yaml_content = {
            "path": str(self.ds_dir),
            "train": "empty/images",
            "names": {0: "CBB"},
        }
        yaml_p = self.temp_path / "data_empty.yaml"
        with open(yaml_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        with self.assertRaises(ValueError) as ctx:
            resolve_data_yaml_split(yaml_p, "train", allow_empty=False)
        self.assertIn("no valid images found", str(ctx.exception).lower())

    def test_resolver_missing_image_file_in_txt_raises(self):
        txt_file = self.ds_dir / "broken.txt"
        txt_file.write_text(str(self.ds_dir / "nonexistent.jpg") + "\n")
        yaml_content = {
            "path": str(self.ds_dir),
            "train": "broken.txt",
            "names": {0: "CBB"},
        }
        yaml_p = self.temp_path / "data_broken.yaml"
        with open(yaml_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        with self.assertRaises(FileNotFoundError):
            resolve_data_yaml_split(yaml_p, "train")


class MockTrainer:
    """Mock trainer untuk menguji orkestrasi Stage 8.4 tanpa full YOLO training."""
    instances = []

    def __init__(
        self,
        overrides: dict[str, Any] | None = None,
        enable_mixture_sampling: bool = False,
        enable_targeted_aug: bool = False,
        manifest_path: str | None = None,
        aug_prob: float = 0.5,
    ):
        self.overrides = overrides or {}
        self.enable_mixture_sampling = enable_mixture_sampling
        self.enable_targeted_aug = enable_targeted_aug
        self.manifest_path = manifest_path
        self.aug_prob = aug_prob
        MockTrainer.instances.append(self)

    def train(self):
        proj = Path(self.overrides.get("project", "."))
        name = self.overrides.get("name", "exp")
        weights_dir = proj / name / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        # Tulis mock weights best.pt
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(b"MOCK_YOLO11N_TRAINED_WEIGHTS_FOR_" + name.encode("utf-8"))


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


class TestGroupFAblationOrchestratorMockIntegration(unittest.TestCase):
    """
    Group F (P0): Pengujian orkestrasi ablasi end-to-end (EXP-0 s.d. EXP-3)
    menggunakan mock trainer dan dataset sintetis terisolasi.
    """

    def setUp(self):
        MockTrainer.instances.clear()
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_ablation_")
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
                "names": {0: "CBB"},
            }, f)

        # Buat raw OOF mock
        self.raw_oof_p = self.temp_path / "raw_oof.csv"
        self.raw_oof_p.write_text("image_id,pred\nleaf_000.jpg,0\n")
        raw_oof_sha = compute_file_sha256(self.raw_oof_p)

        # Buat difficulty manifest mock
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

        # Buat sidecar JSON
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

    def test_ablation_orchestrator_exp_all_mock_execution(self):
        # Eksekusi full --exp all menggunakan MockTrainer dan mock_model_eval_fn
        args = argparse.Namespace(
            data_yaml=str(self.yaml_p),
            manifest_path=str(self.manifest_p),
            output_dir=str(self.out_dir),
            exp="all",
            epochs=1,
            batch=2,
            imgsz=64,
            device="cpu",
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

        # 1. Seluruh 4 eksperimen harus selesai
        self.assertEqual(len(summaries), 4)
        exp_ids = [s["exp_id"] for s in summaries]
        self.assertEqual(exp_ids, ["EXP-0", "EXP-1", "EXP-2", "EXP-3"])

        # 2. MockTrainer harus dipanggil tepat 4 kali
        self.assertEqual(len(MockTrainer.instances), 4)

        # 3. EXP-0 tidak boleh menerima manifest_path (manifest_to_use harus None)
        self.assertIsNone(MockTrainer.instances[0].manifest_path)
        self.assertFalse(MockTrainer.instances[0].enable_mixture_sampling)
        self.assertFalse(MockTrainer.instances[0].enable_targeted_aug)

        # 4. EXP-1 harus mengaktifkan sampling dan menerima manifest
        self.assertIsNotNone(MockTrainer.instances[1].manifest_path)
        self.assertTrue(MockTrainer.instances[1].enable_mixture_sampling)
        self.assertFalse(MockTrainer.instances[1].enable_targeted_aug)

        # 5. EXP-2 harus mengaktifkan targeted aug dan menerima manifest
        self.assertIsNotNone(MockTrainer.instances[2].manifest_path)
        self.assertFalse(MockTrainer.instances[2].enable_mixture_sampling)
        self.assertTrue(MockTrainer.instances[2].enable_targeted_aug)

        # 6. EXP-3 harus mengaktifkan keduanya
        self.assertIsNotNone(MockTrainer.instances[3].manifest_path)
        self.assertTrue(MockTrainer.instances[3].enable_mixture_sampling)
        self.assertTrue(MockTrainer.instances[3].enable_targeted_aug)

        # 7. Laporan komparatif harus terbentuk di root output_dir
        report_file = self.out_dir / "ABLATION_EXPERIMENTS_REPORT.md"
        self.assertTrue(report_file.exists())
        report_content = report_file.read_text(encoding="utf-8")
        self.assertIn("EXP-0", report_content)
        self.assertIn("EXP-1", report_content)
        self.assertIn("EXP-2", report_content)
        self.assertIn("EXP-3", report_content)

    def test_single_experiment_failure_halts_pipeline(self):
        class FailingMockTrainer(MockTrainer):
            def train(self):
                if len(MockTrainer.instances) == 2:  # Gagal pada EXP-1
                    raise RuntimeError("Simulated GPU OOM or training crash on EXP-1")
                super().train()

        args = argparse.Namespace(
            data_yaml=str(self.yaml_p),
            manifest_path=str(self.manifest_p),
            output_dir=str(self.out_dir),
            exp="all",
            epochs=1,
            batch=2,
            imgsz=64,
            device="cpu",
            seed=42,
            skip_if_done=False,
            dry_run=False,
            clean_oof_run_id="clean_oof_run_123",
        )

        with self.assertRaises(RuntimeError) as ctx:
            run_ablation_pipeline(
                args,
                trainer_cls=FailingMockTrainer,
                model_eval_fn=mock_model_eval_fn,
            )
        self.assertIn("Simulated GPU OOM", str(ctx.exception))

        # Laporan komparasi akhir tidak boleh terbentuk jika pipeline gagal
        report_file = self.out_dir / "ABLATION_EXPERIMENTS_REPORT.md"
        self.assertFalse(report_file.exists())

    def test_skip_if_done_reuses_completed_experiments(self):
        # Jalankan dulu run awal
        args = argparse.Namespace(
            data_yaml=str(self.yaml_p),
            manifest_path=str(self.manifest_p),
            output_dir=str(self.out_dir),
            exp="all",
            epochs=1,
            batch=2,
            imgsz=64,
            device="cpu",
            seed=42,
            skip_if_done=False,
            dry_run=False,
            clean_oof_run_id="clean_oof_run_123",
        )
        run_ablation_pipeline(args, trainer_cls=MockTrainer, model_eval_fn=mock_model_eval_fn)
        self.assertEqual(len(MockTrainer.instances), 4)

        # Jalankan kembali dengan skip_if_done=True
        MockTrainer.instances.clear()
        args.skip_if_done = True
        summaries = run_ablation_pipeline(args, trainer_cls=MockTrainer, model_eval_fn=mock_model_eval_fn)
        self.assertEqual(len(summaries), 4)
        # MockTrainer.train() tidak boleh dipanggil ulang sama sekali
        self.assertEqual(len(MockTrainer.instances), 0)


class TestGroupGPretrainedAugmentationMechanics(unittest.TestCase):
    """Group G: Pengujian hashing pretrained weights, augmentasi adaptif & collation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_7_aug_")
        self.temp_path = Path(self.temp_dir).resolve()

        self.ds_dir = self.temp_path / "mock_dataset"
        self.img_dir = self.ds_dir / "images"
        self.lbl_dir = self.ds_dir / "labels"
        self.img_dir.mkdir(parents=True)
        self.lbl_dir.mkdir(parents=True)

        self.img_names = ["im_aug_0.jpg", "im_aug_1.jpg"]
        for name in self.img_names:
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            cv2.imwrite(str(self.img_dir / name), img)
            (self.lbl_dir / name.replace(".jpg", ".txt")).write_text("0 0.5 0.5 0.2 0.2\n")

        self.manifest_path = self.temp_path / "manifest.csv"
        fieldnames = [
            "image_id", "d_image", "p_sampling", "n_gt_objects",
            "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box",
            "box_format", "img_width", "img_height"
        ]
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, name in enumerate(self.img_names):
                writer.writerow({
                    "image_id": name,
                    "d_image": 0.85 if idx == 0 else 0.25,
                    "p_sampling": 0.5,
                    "n_gt_objects": 1,
                    "has_gt": 1,
                    "has_target_small": "1" if idx == 0 else "0",
                    "target_gt_id": 0,
                    "target_small_di": 0.85 if idx == 0 else 0.25,
                    "target_small_box": "0.45,0.45,0.55,0.55",
                    "box_format": "norm_xyxy",
                    "img_width": 128,
                    "img_height": 128,
                })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pretrained_weights_sha256_fixture_computation(self):
        # Pengujian deterministik komputasi sha256 bobot awal pada file fixture
        fixture_weights = self.temp_path / "mock_pretrained.pt"
        fixture_bytes = b"ULTRALYTICS_YOLO11N_MOCK_CHECKPOINT_DATA"
        fixture_weights.write_bytes(fixture_bytes)

        expected_sha = hashlib.sha256(fixture_bytes).hexdigest()
        actual_sha = compute_pretrained_weights_sha256(fixture_weights)
        self.assertEqual(actual_sha, expected_sha)
        self.assertEqual(len(actual_sha), 64)

        # File tidak ada harus raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            compute_pretrained_weights_sha256(self.temp_path / "nonexistent.pt")

    def test_adaptive_yolo_dataset_getitem_augment_true(self):
        hyp = get_cfg()
        hyp.mosaic = 0.0
        hyp.mixup = 0.0
        hyp.imgsz = 128

        dataset = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            imgsz=128,
            batch_size=2,
            augment=True,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix="",
            task="detect",
            classes=None,
            data={"names": {0: "CBB"}, "channels": 3},
            fraction=1.0,
            enable_mixture_sampling=True,
            enable_targeted_aug=True,
            manifest_path=str(self.manifest_path),
            aug_prob=1.0,  # Paksa aktivasi augmentasi context-crop
        )

        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("img", sample)
        self.assertIn("bboxes", sample)
        self.assertIn("cls", sample)
        self.assertGreater(dataset.crop_attempts, 0)
        self.assertGreater(dataset.crops_applied, 0)

    def test_adaptive_yolo_dataset_collate_fn(self):
        hyp = get_cfg()
        hyp.mosaic = 0.0
        hyp.mixup = 0.0
        hyp.imgsz = 128

        dataset = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            imgsz=128,
            batch_size=2,
            augment=True,
            hyp=hyp,
            data={"names": {0: "CBB"}, "channels": 3},
            enable_targeted_aug=True,
            manifest_path=str(self.manifest_path),
            aug_prob=1.0,
        )

        batch_list = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch_list)
        self.assertIn("img", collated)
        self.assertIn("batch_idx", collated)
        self.assertIn("cls", collated)
        self.assertIn("bboxes", collated)
        self.assertEqual(collated["img"].shape[0], 2)

    def test_experiment_mechanism_toggles(self):
        # Verifikasi toggles sampling dan augmentasi pada EXPERIMENT_SPECS
        self.assertEqual(len(EXPERIMENT_SPECS), 4)

        # EXP-0: baseline uniform + standard aug
        self.assertFalse(EXPERIMENT_SPECS[0]["enable_mixture_sampling"])
        self.assertFalse(EXPERIMENT_SPECS[0]["enable_targeted_aug"])

        # EXP-1: difficulty mixture sampling only
        self.assertTrue(EXPERIMENT_SPECS[1]["enable_mixture_sampling"])
        self.assertFalse(EXPERIMENT_SPECS[1]["enable_targeted_aug"])

        # EXP-2: targeted context aug only
        self.assertFalse(EXPERIMENT_SPECS[2]["enable_mixture_sampling"])
        self.assertTrue(EXPERIMENT_SPECS[2]["enable_targeted_aug"])

        # EXP-3: mixture sampling + targeted context aug
        self.assertTrue(EXPERIMENT_SPECS[3]["enable_mixture_sampling"])
        self.assertTrue(EXPERIMENT_SPECS[3]["enable_targeted_aug"])


if __name__ == "__main__":
    unittest.main()

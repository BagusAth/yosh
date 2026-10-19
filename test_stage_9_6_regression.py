#!/usr/bin/env python3
"""
Test Suite: Stage 9.6 Regression & Final Training Readiness Gate
================================================================

Memvalidasi seluruh perbaikan Stage 9.6 secara objektif dan mendalam:
1. Output Directory Safety & Historical Artefact Protection
2. Manifest Provenance & 100% Coverage Verification
3. Real-path Targeted Context Augmentation Integration & Collation
4. Content-Aware SHA-256 Dataset Fingerprinting
5. Single Source of Truth Dataset Path Resolution
6. Pretrained Weights SHA-256 Verification
"""

import csv
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from research_utils.detection_metrics import (
    compute_file_sha256,
    compute_dataset_inventory_fingerprint,
    compute_pretrained_weights_sha256,
    resolve_data_yaml_split,
    validate_output_directory_safety,
    CLASS_NAMES,
)
from stage_8_3_adaptive_pipeline import AdaptiveYOLODataset
from stage_8_4_run_ablation_experiments import (
    verify_manifest_provenance_and_coverage,
    verify_ablation_reuse_safety,
    build_experiment_run_identity,
    EXPERIMENT_SPECS,
)


class TestOutputDirectorySafety(unittest.TestCase):
    """Pengujian proteksi direktori output dan pencegahan penimpaan artefak historis."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_stage_9_6_out_")
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_output_directory_passes(self):
        empty_dir = self.temp_path / "clean_run"
        empty_dir.mkdir()
        is_safe, msg = validate_output_directory_safety(empty_dir, stage_name="test_stage")
        self.assertTrue(is_safe, f"Empty directory should be safe: {msg}")

    def test_historical_output_directory_forbidden(self):
        historical_names = [
            "d:/UNDIP/SKRIPSI/stage_8_1_oof",
            "d:/UNDIP/SKRIPSI/stage_8_2_difficulty",
            "d:/UNDIP/SKRIPSI/stage_8_ablation",
            "d:/UNDIP/SKRIPSI/cassava-v7-stage-7",
        ]
        for h_name in historical_names:
            with self.assertRaises(PermissionError):
                validate_output_directory_safety(h_name, stage_name="test_stage")

    def test_unverified_partial_output_rejected(self):
        partial_dir = self.temp_path / "partial_run"
        partial_dir.mkdir()
        # Buat file partial seperti fold_assignments atau YAML tanpa metadata
        (partial_dir / "fold_assignments.csv").write_text("dummy,content\n", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            validate_output_directory_safety(partial_dir, stage_name="stage_8_1")

    def test_metadata_mismatch_rejected(self):
        run_dir = self.temp_path / "mismatch_run"
        run_dir.mkdir()
        (run_dir / "metrics_summary.json").write_text(json.dumps({
            "run_metadata": {"epochs": 10, "seed": 42}
        }), encoding="utf-8")

        expected_identity = {"epochs": 50, "seed": 42}
        with self.assertRaises(PermissionError):
            validate_output_directory_safety(
                run_dir,
                stage_name="stage_8_4",
                allow_reuse=True,
                expected_identity=expected_identity,
            )

    def test_valid_verified_reuse_passes(self):
        run_dir = self.temp_path / "valid_reuse_run"
        run_dir.mkdir()
        expected_identity = {"epochs": 50, "seed": 42, "batch": 16}
        (run_dir / "metrics_summary.json").write_text(json.dumps({
            "run_metadata": expected_identity
        }), encoding="utf-8")

        is_safe, msg = validate_output_directory_safety(
            run_dir,
            stage_name="stage_8_4",
            allow_reuse=True,
            expected_identity=expected_identity,
        )
        self.assertTrue(is_safe, f"Matching metadata should be accepted for reuse: {msg}")

    def test_dry_run_does_not_mutate_directory(self):
        dry_dir = self.temp_path / "dry_run_target"
        # Directory does not exist yet
        is_safe, msg = validate_output_directory_safety(
            dry_dir,
            stage_name="stage_8_4",
            is_dry_run=True,
        )
        self.assertTrue(is_safe)
        self.assertFalse(dry_dir.exists(), "Dry run must not create the target directory")


class TestManifestProvenanceAndCoverage(unittest.TestCase):
    """Pengujian keabsahan manifest dari clean Stage 8.2 run untuk Stage 8.4."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_manifest_")
        self.temp_path = Path(self.temp_dir)

        # Buat dummy train images list
        self.train_images = [
            Path(f"img_{i:04d}.jpg") for i in range(10)
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_manifest_and_provenance(
        self,
        image_ids: list[str],
        dataset_fp: str = "ds_fingerprint_12345",
        yaml_sha: str = "yaml_sha_12345",
        clean_oof_id: str = "d:/UNDIP/SKRIPSI/stage_8_1_oof_clean",
    ) -> Path:
        csv_path = self.temp_path / "train_image_difficulty_manifest.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "d_image", "p_sampling", "n_gt_objects"])
            writer.writeheader()
            for img_id in image_ids:
                writer.writerow({
                    "image_id": img_id,
                    "d_image": 0.50,
                    "p_sampling": 1.0 / len(image_ids),
                    "n_gt_objects": 2,
                })

        csv_sha = compute_file_sha256(csv_path)
        prov_path = self.temp_path / "train_image_difficulty_manifest.provenance.json"
        prov_data = {
            "manifest_file": csv_path.name,
            "manifest_csv_sha256": csv_sha,
            "dataset_inventory_fingerprint": dataset_fp,
            "data_yaml_sha256": yaml_sha,
            "clean_oof_run_id": clean_oof_id,
            "image_count": len(image_ids),
            "unique_image_count": len(set(image_ids)),
        }
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(prov_data, f, indent=2)

        return csv_path

    def test_valid_manifest_and_provenance_passes(self):
        img_ids = [p.name for p in self.train_images]
        manifest_path = self._create_manifest_and_provenance(img_ids)

        ok, msg, prov = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertTrue(ok, f"Valid manifest should pass: {msg}")

    def test_missing_provenance_sidecar_rejected(self):
        img_ids = [p.name for p in self.train_images]
        manifest_path = self._create_manifest_and_provenance(img_ids)
        # Hapus sidecar
        prov_path = self.temp_path / "train_image_difficulty_manifest.provenance.json"
        prov_path.unlink()

        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Sidecar provenance JSON not found", msg)

    def test_manifest_csv_tamper_rejected(self):
        img_ids = [p.name for p in self.train_images]
        manifest_path = self._create_manifest_and_provenance(img_ids)
        # Modifikasi file CSV setelah provenance dibuat
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write("tampered,0.99,0.99,1\n")

        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Manifest CSV SHA-256 mismatch", msg)

    def test_manifest_historical_oof_rejected(self):
        img_ids = [p.name for p in self.train_images]
        manifest_path = self._create_manifest_and_provenance(
            img_ids,
            clean_oof_id="d:/UNDIP/SKRIPSI/stage_8_1_oof",  # Historical OOF dir
        )
        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Provenance references historical unverified OOF run", msg)

    def test_manifest_dataset_fingerprint_mismatch_rejected(self):
        img_ids = [p.name for p in self.train_images]
        manifest_path = self._create_manifest_and_provenance(img_ids, dataset_fp="old_fingerprint")

        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="new_fingerprint",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Dataset inventory fingerprint mismatch", msg)

    def test_manifest_duplicate_image_id_rejected(self):
        img_ids = [p.name for p in self.train_images]
        img_ids[1] = img_ids[0]  # Buat duplikasi
        manifest_path = self._create_manifest_and_provenance(img_ids)

        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Duplicate image_id", msg)

    def test_manifest_incomplete_coverage_rejected(self):
        # Hanya menyertakan 8 dari 10 citra
        img_ids = [p.name for p in self.train_images[:8]]
        manifest_path = self._create_manifest_and_provenance(img_ids)

        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Manifest coverage incomplete", msg)

    def test_manifest_extra_images_rejected(self):
        # Menyertakan citra tambahan di luar dataset
        img_ids = [p.name for p in self.train_images] + ["extra_phantom.jpg"]
        manifest_path = self._create_manifest_and_provenance(img_ids)

        ok, msg, _ = verify_manifest_provenance_and_coverage(
            manifest_path=manifest_path,
            expected_dataset_fingerprint="ds_fingerprint_12345",
            expected_data_yaml_sha256="yaml_sha_12345",
            expected_image_files=self.train_images,
        )
        self.assertFalse(ok)
        self.assertIn("Manifest contains 1 extraneous images", msg)


class TestRealAdaptiveYOLODatasetTargetedAugmentation(unittest.TestCase):
    """
    Membuktikan targeted context augmentation benar-benar berjalan pada jalur
    AdaptiveYOLODataset.__getitem__() dengan augment=True dan collate_fn berhasil.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_adaptive_ds_")
        self.temp_path = Path(self.temp_dir)

        # Buat dummy train images & labels
        self.img_dir = self.temp_path / "images"
        self.lbl_dir = self.temp_path / "labels"
        self.img_dir.mkdir()
        self.lbl_dir.mkdir()

        # Buat 4 citra sintetis (640x640)
        self.img_files = []
        for i in range(4):
            im_p = self.img_dir / f"syn_{i:02d}.jpg"
            img = np.full((640, 640, 3), fill_value=50 + i * 40, dtype=np.uint8)
            # Gambar lesi kecil (20x20 pixel) di tengah
            cv2.rectangle(img, (300, 300), (320, 320), (0, 0, 255), -1)
            cv2.imwrite(str(im_p), img)
            self.img_files.append(im_p)

            # Buat label YOLO: class 0 (CBB), center 310/640=0.484375, 310/640=0.484375, w=20/640=0.03125, h=20/640=0.03125
            lb_p = self.lbl_dir / f"syn_{i:02d}.txt"
            lb_p.write_text("0 0.484375 0.484375 0.03125 0.03125\n", encoding="utf-8")

        # Buat manifest kesulitan citra lengkap sesuai skema stage_8_2 (norm_xyxy)
        self.manifest_path = self.temp_path / "manifest.csv"
        fieldnames = [
            "image_id", "d_image", "p_sampling", "n_gt_objects",
            "has_gt", "has_target_small", "target_gt_id", "target_small_di", "target_small_box",
            "box_format", "img_width", "img_height"
        ]
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for im_p in self.img_files:
                writer.writerow({
                    "image_id": im_p.name,
                    "d_image": 0.85,
                    "p_sampling": 0.25,
                    "n_gt_objects": 1,
                    "has_gt": 1,
                    "has_target_small": "1",
                    "target_gt_id": 0,
                    "target_small_di": 0.85,
                    "target_small_box": "0.46875,0.46875,0.50000,0.50000",
                    "box_format": "norm_xyxy",
                    "img_width": 640,
                    "img_height": 640,
                })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_getitem_augment_applies_context_crop(self):
        from ultralytics.cfg import get_cfg
        hyp = get_cfg()
        hyp.mosaic = 0.0  # Nonaktifkan mosaic agar langsung menguji context crop
        hyp.mixup = 0.0

        # Inisialisasi dataset dengan augmentasi konteks aktif
        dataset = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            imgsz=640,
            batch_size=4,
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
            manifest_path=self.manifest_path,
            aug_prob=1.0,  # Paksa 100% augmentasi untuk verifikasi deterministik
        )

        self.assertEqual(len(dataset), 4)

        # Ambil sampel index 0 via __getitem__ (jalur produksi training riil)
        sample = dataset[0]

        # Validasi struktur return
        self.assertIsInstance(sample, dict)
        self.assertIn("img", sample)
        self.assertIn("bboxes", sample)
        self.assertIn("cls", sample)

        # Periksa dimensi citra output (3, 640, 640)
        img_tensor = sample["img"]
        if isinstance(img_tensor, torch.Tensor):
            self.assertEqual(img_tensor.shape, (3, 640, 640))
        elif isinstance(img_tensor, np.ndarray):
            self.assertEqual(img_tensor.shape, (640, 640, 3))

        # Periksa counter augmentasi
        self.assertGreater(dataset.crop_attempts, 0, "crop_attempts harus bertambah!")
        self.assertGreater(dataset.crops_applied, 0, "crops_applied harus bertambah ketika aug_prob=1.0!")

        # Periksa validitas bounding box
        bboxes = sample["bboxes"]
        self.assertGreater(len(bboxes), 0)
        # Bounding box harus berada dalam rentang ternormalisasi [0, 1]
        for b in bboxes:
            for coord in b:
                self.assertGreaterEqual(coord, -0.01)
                self.assertLessEqual(coord, 1.01)

    def test_collate_fn_with_mixed_cropped_and_uncropped_items(self):
        dataset_cropped = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            imgsz=640,
            batch_size=2,
            augment=True,
            data={"names": {0: "CBB"}, "channels": 3},
            enable_targeted_aug=True,
            manifest_path=self.manifest_path,
            aug_prob=1.0,  # Cropped
        )

        dataset_uncropped = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            imgsz=640,
            batch_size=2,
            augment=True,
            data={"names": {0: "CBB"}, "channels": 3},
            enable_targeted_aug=False,  # Uncropped
        )

        item_cropped = dataset_cropped[0]
        item_uncropped = dataset_uncropped[1]

        # Pastikan dictionary keys seragam sehingga collate_fn Ultralytics tidak melempar KeyError/IndexError
        self.assertEqual(
            set(item_cropped.keys()),
            set(item_uncropped.keys()),
            "Keys dictionary antara cropped dan uncropped sample harus identik!",
        )

        # Jalankan collate_fn resmi Ultralytics
        batch_list = [item_cropped, item_uncropped]
        batch = AdaptiveYOLODataset.collate_fn(batch_list)

        self.assertIn("img", batch)
        self.assertIn("bboxes", batch)
        self.assertIn("batch_idx", batch)
        self.assertEqual(batch["img"].shape, (2, 3, 640, 640))


class TestContentAwareSHA256Integrity(unittest.TestCase):
    """Pengujian pemeriksaan integritas dataset berbasis isi SHA-256."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_content_sha_")
        self.temp_path = Path(self.temp_dir)

        self.img_dir = self.temp_path / "train" / "images"
        self.lbl_dir = self.temp_path / "train" / "labels"
        self.img_dir.mkdir(parents=True)
        self.lbl_dir.mkdir(parents=True)

        for i in range(3):
            im_p = self.img_dir / f"test_{i}.jpg"
            im_p.write_bytes(b"A" * 1000)
            lb_p = self.lbl_dir / f"test_{i}.txt"
            lb_p.write_text(f"0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fingerprint_detects_byte_modification_same_size(self):
        fp_before = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)

        # Ubah 1 byte di dalam file citra tanpa mengubah ukuran file (tetap 1000 bytes)
        target_im = self.img_dir / "test_1.jpg"
        target_im.write_bytes(b"B" + b"A" * 999)

        self.assertEqual(target_im.stat().st_size, 1000)

        fp_after = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        self.assertNotEqual(
            fp_before,
            fp_after,
            "Fingerprint harus berubah meskipun ukuran file persis sama!",
        )

    def test_fingerprint_detects_label_modification(self):
        fp_before = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)

        # Ubah label class
        target_lb = self.lbl_dir / "test_0.txt"
        target_lb.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        fp_after = compute_dataset_inventory_fingerprint(self.img_dir, self.lbl_dir)
        self.assertNotEqual(
            fp_before,
            fp_after,
            "Fingerprint harus mendeteksi perubahan pada anotasi label!",
        )

    def test_fingerprint_fails_on_empty_dir(self):
        empty_dir = self.temp_path / "empty_im"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            compute_dataset_inventory_fingerprint(empty_dir, self.lbl_dir)


class TestSingleSourceOfTruthPathResolution(unittest.TestCase):
    """Pengujian resolusi path dataset kanonis resolve_data_yaml_split."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_resolve_yaml_")
        self.temp_path = Path(self.temp_dir)

        # Buat dummy dataset folder
        self.ds_dir = self.temp_path / "mock_dataset"
        self.train_im = self.ds_dir / "train" / "images"
        self.train_lb = self.ds_dir / "train" / "labels"
        self.train_im.mkdir(parents=True)
        self.train_lb.mkdir(parents=True)

        for i in range(2):
            (self.train_im / f"im_{i}.jpg").write_bytes(b"dummy")
            (self.train_lb / f"im_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_relative_path(self):
        yaml_content = {
            "path": str(self.ds_dir),
            "train": "train/images",
            "val": "train/images",
            "test": "train/images",
            "names": {0: "CBB"},
        }
        yaml_path = self.temp_path / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        im_dir, lb_dir, img_files = resolve_data_yaml_split(yaml_path, "train")
        self.assertEqual(im_dir, self.train_im.resolve())
        self.assertEqual(lb_dir, self.train_lb.resolve())
        self.assertEqual(len(img_files), 2)

    def test_resolve_txt_file_split(self):
        txt_path = self.ds_dir / "train_list.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in self.train_im.glob("*.jpg"):
                f.write(f"{p.resolve()}\n")

        yaml_content = {
            "path": str(self.ds_dir),
            "train": "train_list.txt",
            "names": {0: "CBB"},
        }
        yaml_path = self.temp_path / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        im_dir, lb_dir, img_files = resolve_data_yaml_split(yaml_path, "train")
        self.assertEqual(len(img_files), 2)
        self.assertTrue(im_dir.exists())

    def test_missing_split_key_raises(self):
        yaml_content = {
            "path": str(self.ds_dir),
            "val": "train/images",
        }
        yaml_path = self.temp_path / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        with self.assertRaises(KeyError):
            resolve_data_yaml_split(yaml_path, "train")


class TestPretrainedWeightsSHA256(unittest.TestCase):
    """Pengujian identifikasi bobot model awal (pretrained weights)."""

    def test_pretrained_weights_sha256_computation(self):
        weights_p = Path("yolo11n.pt")
        if weights_p.exists():
            sha = compute_pretrained_weights_sha256(weights_p)
            self.assertEqual(len(sha), 64)
            # Pastikan cocok dengan bobot yang ada di workspace
            actual = compute_file_sha256(weights_p)
            self.assertEqual(sha, actual)


if __name__ == "__main__":
    unittest.main()

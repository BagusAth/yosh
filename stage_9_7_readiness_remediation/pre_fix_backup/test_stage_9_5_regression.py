"""
Unit and Regression Test Suite for STAGE 9.5
Targeted Remediation, Artifact Integrity, and Independent Pipeline Verification

Test Suites:
1. Suite 1: P0 - Stage 8.1 Run Identity & Checkpoint Reuse Safety (10 tests)
2. Suite 2: P1 - Stage 8.4 Ablation Fingerprint & Skip-if-done Decision (7 tests)
3. Suite 3: P1 - Canonical Matching Equivalence across Stage 7 and Stage 8 (9 tests)
4. Suite 4: P1 - Dynamic Dimensions & Non-Square Image Geometry (4 tests)
5. Suite 5: P1 - Annotation Standardization & Verify-Only Integrity Checks (10 tests)
6. Suite 6: P1 - Real Ultralytics Dataset / Collate Integration Test (3 tests)
7. Suite 7: P2 - Mixture Sampling Formula & Mathematical Robustness (5 tests)
"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances

from execute_phase_b_standardization import (
    convert_polygon_to_bbox,
    parse_raw_label_line,
    verify_dataset_standardization,
)
from research_utils.detection_metrics import (
    calculate_mixture_weights,
    calculate_object_difficulty,
    classify_object_size,
    compute_dataset_inventory_fingerprint,
    compute_file_sha256,
    get_image_dimensions,
    greedy_bipartite_matching,
    parse_yolo_labels,
)
from stage_8_1_group_oof_profiling import (
    build_fold_run_identity,
    verify_fold_reuse_safety,
)
from stage_8_3_adaptive_pipeline import AdaptiveYOLODataset
from stage_8_4_run_ablation_experiments import (
    EXPERIMENT_SPECS,
    build_experiment_run_identity,
    verify_ablation_reuse_safety,
)


class TestStage81RunIdentityAndReuseSafety(unittest.TestCase):
    """Suite 1: Stage 8.1 20-field Run Identity and Checkpoint Reuse Guard."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        # Create dummy files for hashing
        self.script_path = Path(self.temp_dir) / "dummy_script.py"
        self.script_path.write_text("print('stage 8.1')", encoding="utf-8")

        self.yaml_path = Path(self.temp_dir) / "data.yaml"
        self.yaml_path.write_text("names: ['CBB']\nnc: 1", encoding="utf-8")

        self.inner_train = Path(self.temp_dir) / "inner_train.txt"
        self.inner_train.write_text("img1.jpg\nimg2.jpg\n", encoding="utf-8")

        self.inner_val = Path(self.temp_dir) / "inner_val.txt"
        self.inner_val.write_text("img3.jpg\n", encoding="utf-8")

        self.outer_holdout = Path(self.temp_dir) / "outer_holdout.txt"
        self.outer_holdout.write_text("img4.jpg\n", encoding="utf-8")

        self.ckpt_path = Path(self.temp_dir) / "best.pt"
        self.ckpt_path.write_bytes(b"dummy_weights_stage81_best")

        self.pred_path = Path(self.temp_dir) / "oof_predictions.csv"
        self.pred_path.write_text("image_name,fold\nimg4.jpg,0\n", encoding="utf-8")

        self.output_dir = Path(self.temp_dir) / "fold_0_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.output_dir / "fold_0_run_metadata.json"

        # Baseline identity
        self.base_identity = build_fold_run_identity(
            script_path=self.script_path,
            yaml_path=self.yaml_path,
            inner_train_txt=self.inner_train,
            inner_val_txt=self.inner_val,
            outer_holdout_txt=self.outer_holdout,
            dataset_inventory_fingerprint="abc123inventory",
            checkpoint_selection_mode="nested_inner_val",
            fold_id=0,
            epochs=20,
            batch_size=16,
            imgsz=640,
            seed=42,
            conf_thresh=0.25,
            model_variant="yolo11n.pt",
            checkpoint_eval_sha256=compute_file_sha256(self.ckpt_path),
            predictions_sha256=compute_file_sha256(self.pred_path),
            holdout_image_count=1,
        )

        # Write valid metadata
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.base_identity, f, indent=2)

    def test_identical_metadata_and_checkpoint_allows_reuse(self):
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertTrue(can_reuse)
        self.assertIn("verified", reason.lower())

    def test_script_hash_change_rejects_reuse(self):
        modified_identity = copy.deepcopy(self.base_identity)
        modified_identity["script_sha256"] = "different_script_sha"
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=modified_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("script_sha256 mismatch", reason)

    def test_yaml_hash_change_rejects_reuse(self):
        modified_identity = copy.deepcopy(self.base_identity)
        modified_identity["fold_yaml_sha256"] = "different_yaml_sha"
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=modified_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("fold_yaml_sha256 mismatch", reason)

    def test_inner_train_hash_change_rejects_reuse(self):
        modified_identity = copy.deepcopy(self.base_identity)
        modified_identity["inner_train_sha256"] = "different_train_sha"
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=modified_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("inner_train_sha256 mismatch", reason)

    def test_inner_val_hash_change_rejects_reuse(self):
        modified_identity = copy.deepcopy(self.base_identity)
        modified_identity["inner_val_sha256"] = "different_val_sha"
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=modified_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("inner_val_sha256 mismatch", reason)

    def test_outer_holdout_hash_change_rejects_reuse(self):
        modified_identity = copy.deepcopy(self.base_identity)
        modified_identity["outer_holdout_sha256"] = "different_holdout_sha"
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=modified_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("outer_holdout_sha256 mismatch", reason)

    def test_checkpoint_hash_change_detected_and_rejected(self):
        self.ckpt_path.write_bytes(b"tampered_weights")
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("checkpoint hash mismatch", reason.lower())

    def test_predictions_exist_but_metadata_missing_rejected(self):
        self.meta_file.unlink()
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("metadata json not found", reason.lower())

    def test_metadata_exists_but_checkpoint_missing_rejected(self):
        self.ckpt_path.unlink()
        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("checkpoint file does not exist", reason.lower())

    def test_safety_guard_rejects_reuse_when_metadata_mismatches_even_if_checkpoint_exists(self):
        """P0 Requirement: Old checkpoint must never be silently reused if metadata mismatches."""
        tampered_meta = copy.deepcopy(self.base_identity)
        tampered_meta["seed"] = 999  # different seed
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(tampered_meta, f, indent=2)

        can_reuse, reason = verify_fold_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_path=self.ckpt_path,
            predictions_path=self.pred_path,
            holdout_image_count=1,
        )
        self.assertFalse(can_reuse)
        self.assertIn("seed mismatch", reason)


class TestStage84AblationFingerprintSafety(unittest.TestCase):
    """Suite 2: Stage 8.4 Ablation Fingerprint & Skip-if-done Decision."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        self.script_path = Path(self.temp_dir) / "dummy_ablation.py"
        self.script_path.write_text("print('stage 8.4')", encoding="utf-8")

        self.yaml_path = Path(self.temp_dir) / "data.yaml"
        self.yaml_path.write_text("names: ['CBB']\nnc: 1", encoding="utf-8")

        self.manifest_path = Path(self.temp_dir) / "manifest.csv"
        self.manifest_path.write_text("image_name,difficulty\nimg1.jpg,0.5\n", encoding="utf-8")

        self.pretrained_path = Path(self.temp_dir) / "yolo11n.pt"
        self.pretrained_path.write_bytes(b"dummy_pretrained_weights")

        self.ckpt_path = Path(self.temp_dir) / "best.pt"
        self.ckpt_path.write_bytes(b"dummy_trained_ablation_best")

        self.output_dir = Path(self.temp_dir) / "exp1_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.output_dir / "run_metadata.json"

        self.spec = EXPERIMENT_SPECS[1]  # EXP-1

        self.base_identity = build_experiment_run_identity(
            exp_id="EXP-1",
            spec=self.spec,
            script_path=self.script_path,
            yaml_path=self.yaml_path,
            manifest_path=self.manifest_path,
            dataset_inventory_fingerprint="inv_fp_clean_456",
            pretrained_weights_path=self.pretrained_path,
            epochs=50,
            batch_size=16,
            imgsz=640,
            seed=42,
            device="cuda:0",
            checkpoint_best_sha256=compute_file_sha256(self.ckpt_path),
        )

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.base_identity, f, indent=2)

    def test_identical_metadata_allows_reuse(self):
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertTrue(can_reuse)
        self.assertIn("verified", reason.lower())

    def test_script_hash_change_rejects_ablation_reuse(self):
        mod = copy.deepcopy(self.base_identity)
        mod["script_sha256"] = "tampered_script_sha"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=mod,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("script_sha256 mismatch", reason)

    def test_data_yaml_hash_change_rejects_ablation_reuse(self):
        mod = copy.deepcopy(self.base_identity)
        mod["data_yaml_sha256"] = "tampered_yaml_sha"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=mod,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("data_yaml_sha256 mismatch", reason)

    def test_manifest_hash_change_rejects_exp1_reuse(self):
        mod = copy.deepcopy(self.base_identity)
        mod["manifest_sha256"] = "tampered_manifest_sha"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=mod,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("manifest_sha256 mismatch", reason)

    def test_epochs_batch_imgsz_seed_mismatch_rejects_reuse(self):
        for field, bad_val in [("epochs", 100), ("batch_size", 32), ("imgsz", 1280), ("seed", 0)]:
            mod = copy.deepcopy(self.base_identity)
            mod[field] = bad_val
            can_reuse, reason = verify_ablation_reuse_safety(
                meta_path=self.meta_file,
                expected_identity=mod,
                checkpoint_best_path=self.ckpt_path,
            )
            self.assertFalse(can_reuse, f"Should reject for mismatch in {field}")
            self.assertIn(f"{field} mismatch", reason)

    def test_pretrained_weight_hash_change_rejects_reuse(self):
        mod = copy.deepcopy(self.base_identity)
        mod["pretrained_weights_sha256"] = "tampered_pretrained_sha"
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=mod,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("pretrained_weights_sha256 mismatch", reason)

    def test_missing_metadata_treated_as_unverified(self):
        self.meta_file.unlink()
        can_reuse, reason = verify_ablation_reuse_safety(
            meta_path=self.meta_file,
            expected_identity=self.base_identity,
            checkpoint_best_path=self.ckpt_path,
        )
        self.assertFalse(can_reuse)
        self.assertIn("run_metadata.json not found", reason.lower())


class TestCanonicalMatchingEquivalence(unittest.TestCase):
    """Suite 3: Canonical Global Greedy Matching & Taxonomy across Stage 7 & 8."""

    def test_one_pred_competing_with_two_gt(self):
        gt_boxes = [[100, 100, 200, 200], [120, 120, 220, 220]]
        pred_boxes = [[130, 130, 230, 230]]
        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        self.assertIn(1, matches)
        self.assertEqual(matches[1][0], 0)
        self.assertNotIn(0, matches)
        self.assertEqual(len(unassigned_preds), 0)

    def test_two_preds_competing_with_one_gt(self):
        gt_boxes = [[100, 100, 200, 200]]
        pred_boxes = [[105, 105, 195, 195], [130, 130, 220, 220]]
        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        self.assertIn(0, matches)
        self.assertEqual(matches[0][0], 0)
        self.assertIn(1, unassigned_preds)

    def test_pairing_order_invariance(self):
        gt_boxes = [[0, 0, 50, 50], [100, 100, 150, 150]]
        pred_boxes = [[102, 102, 148, 148], [2, 2, 48, 48]]
        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        self.assertEqual(matches[0][0], 1)
        self.assertEqual(matches[1][0], 0)

    def test_iou_below_association_threshold(self):
        gt_boxes = [[0, 0, 100, 100]]
        pred_boxes = [[95, 95, 195, 195]]
        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unassigned_preds), 1)

    def test_iou_between_association_and_localization_threshold(self):
        gt_objects = [{"box": [0, 0, 100, 100], "class_id": 0, "size_category": "Medium", "size_penalty": 0.5}]
        pred_boxes = [[40, 0, 140, 100]]
        res = greedy_bipartite_matching(
            gt_objects=gt_objects,
            pred_boxes=pred_boxes,
            pred_classes=[0],
            pred_scores=[0.8],
            iou_thresh=0.10,
            tp_iou_thresh=0.50,
        )
        self.assertEqual(res["gt_assignments"][0]["error_type"], "Localization_Error")

    def test_iou_above_50_wrong_class(self):
        gt_objects = [{"box": [0, 0, 100, 100], "class_id": 0, "size_category": "Medium", "size_penalty": 0.5}]
        pred_boxes = [[5, 5, 95, 95]]
        res = greedy_bipartite_matching(
            gt_objects=gt_objects,
            pred_boxes=pred_boxes,
            pred_classes=[1],
            pred_scores=[0.9],
            iou_thresh=0.10,
            tp_iou_thresh=0.50,
        )
        self.assertEqual(res["gt_assignments"][0]["error_type"], "Classification_Confusion")

    def test_one_tp_and_one_unmatched_pred(self):
        gt_objects = [{"box": [0, 0, 100, 100], "class_id": 0, "size_category": "Medium", "size_penalty": 0.5}]
        pred_boxes = [[5, 5, 95, 95], [300, 300, 400, 400]]
        res = greedy_bipartite_matching(
            gt_objects=gt_objects,
            pred_boxes=pred_boxes,
            pred_classes=[0, 0],
            pred_scores=[0.9, 0.8],
            iou_thresh=0.10,
            tp_iou_thresh=0.50,
        )
        self.assertEqual(res["gt_assignments"][0]["error_type"], "True_Positive")
        self.assertIn(1, res["unassigned_preds"])

    def test_image_without_gt_and_with_false_positives(self):
        res = greedy_bipartite_matching(
            gt_objects=[],
            pred_boxes=[[10, 10, 50, 50], [100, 100, 150, 150]],
            pred_classes=[0, 1],
            pred_scores=[0.7, 0.8],
            iou_thresh=0.10,
            tp_iou_thresh=0.50,
        )
        self.assertEqual(len(res["matches"]), 0)
        self.assertEqual(res["unassigned_preds"], {0, 1})

    def test_object_size_classification_thresholds(self):
        self.assertEqual(classify_object_size(500)[0], "Small")
        self.assertEqual(classify_object_size(1023.9)[0], "Small")
        self.assertEqual(classify_object_size(1024.0)[0], "Medium")
        self.assertEqual(classify_object_size(5000)[0], "Medium")
        self.assertEqual(classify_object_size(9215.9)[0], "Medium")
        self.assertEqual(classify_object_size(9216.0)[0], "Large")
        self.assertEqual(classify_object_size(20000)[0], "Large")


class TestDynamicDimensionsAndGeometry(unittest.TestCase):
    """Suite 4: Dynamic Dimensions and Non-Square Image Geometry (1920x1080)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        self.img_path = Path(self.temp_dir) / "wide_sample.jpg"
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(self.img_path), img)

        self.label_path = Path(self.temp_dir) / "wide_sample.txt"
        self.label_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    def test_get_image_dimensions_dynamic(self):
        w, h = get_image_dimensions(self.img_path)
        self.assertEqual(w, 1920)
        self.assertEqual(h, 1080)

    def test_dynamic_parsing_and_area_calculation(self):
        parsed = parse_yolo_labels(self.label_path, img_width=1920, img_height=1080)
        self.assertEqual(len(parsed), 1)
        obj = parsed[0]
        self.assertAlmostEqual(obj["box"][0], 864.0, places=2)
        self.assertAlmostEqual(obj["box"][1], 486.0, places=2)
        self.assertAlmostEqual(obj["box"][2], 1056.0, places=2)
        self.assertAlmostEqual(obj["box"][3], 594.0, places=2)
        self.assertAlmostEqual(obj["area"], 20736.0, places=2)
        self.assertEqual(obj["size_category"], "Large")

    def test_non_square_context_crop_transformation(self):
        ds = AdaptiveYOLODataset(
            img_path=str(self.temp_dir),
            data={"names": {0: "CBB"}},
            augment=False,
        )
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        instances = Instances(
            bboxes=np.array([[0.5, 0.5, 0.02, 0.03]], dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        )
        lbl = {
            "img": img,
            "instances": instances,
            "cls": np.array([[0]], dtype=np.float32),
            "im_file": "wide_sample.jpg",
        }
        target_info = {
            "has_target_small": True,
            "target_small_box": "0.49,0.485,0.51,0.515",
            "box_format": "norm_xyxy",
            "img_width": 1920,
            "img_height": 1080,
            "target_gt_id": 0,
        }
        out = ds.apply_context_crop(lbl, target_info)
        self.assertTrue(out.get("_is_context_cropped", False))
        self.assertEqual(out["img"].shape, (640, 640, 3))
        out_box = out["instances"].bboxes[0]
        self.assertAlmostEqual(out_box[0], 0.5, places=2)
        self.assertAlmostEqual(out_box[1], 0.5, places=2)
        self.assertTrue(np.all(out["instances"].bboxes >= 0.0))
        self.assertTrue(np.all(out["instances"].bboxes <= 1.0))

    def test_hardcoded_640_fallback_is_eliminated_when_dimensions_provided(self):
        parsed_dyn = parse_yolo_labels(self.label_path, img_width=1920, img_height=1080)
        parsed_640 = parse_yolo_labels(self.label_path, img_width=640, img_height=640)
        self.assertNotEqual(parsed_dyn[0]["area"], parsed_640[0]["area"])
        self.assertEqual(parsed_640[0]["area"], 4096.0)
        self.assertEqual(parsed_dyn[0]["area"], 20736.0)


class TestAnnotationStandardizationAndIntegrity(unittest.TestCase):
    """Suite 5: Annotation Standardization & Verify-Only Integrity Checks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def test_valid_box_preservation_with_exact_geometry(self):
        line = "0 0.500000 0.500000 0.200000 0.100000"
        res = parse_raw_label_line(line)
        self.assertEqual(res["type"], "bbox")
        self.assertEqual(res["class_id"], 0)
        np.testing.assert_allclose([res["xc"], res["yc"], res["w"], res["h"]], [0.5, 0.5, 0.2, 0.1], atol=1e-6)

    def test_valid_polygon_converted_to_bounding_box(self):
        poly_coords = [0.1, 0.2, 0.5, 0.8, 0.3, 0.1]
        bbox = convert_polygon_to_bbox(poly_coords)
        self.assertAlmostEqual(bbox[0], 0.30, places=6)
        self.assertAlmostEqual(bbox[1], 0.45, places=6)
        self.assertAlmostEqual(bbox[2], 0.40, places=6)
        self.assertAlmostEqual(bbox[3], 0.70, places=6)

    def test_out_of_range_coordinate_strictly_rejected(self):
        bad_line = "0 1.250000 0.500000 0.200000 0.100000"
        with self.assertRaises(ValueError) as ctx:
            parse_raw_label_line(bad_line)
        self.assertIn("di luar batas [0, 1]", str(ctx.exception))

    def test_non_finite_coordinate_strictly_rejected(self):
        bad_line = "0 NaN 0.500000 0.200000 0.100000"
        with self.assertRaises(ValueError) as ctx:
            parse_raw_label_line(bad_line)
        self.assertIn("bukan nilai finite", str(ctx.exception).lower())

    def test_invalid_column_count_rejected(self):
        bad_line = "0 0.5 0.5 0.2"
        with self.assertRaises(ValueError) as ctx:
            parse_raw_label_line(bad_line)
        self.assertIn("jumlah kolom < 5", str(ctx.exception).lower())

    def test_class_id_out_of_range_rejected(self):
        bad_line = "5 0.5 0.5 0.2 0.2"
        with self.assertRaises(ValueError) as ctx:
            parse_raw_label_line(bad_line)
        self.assertIn("Class ID 5 di luar rentang valid", str(ctx.exception))

    def test_verify_only_detects_missing_destination_label(self):
        src_root = Path(self.temp_dir) / "src"
        dst_root = Path(self.temp_dir) / "dst"
        for r in [src_root, dst_root]:
            for sp in ["train", "valid", "test"]:
                (r / sp / "images").mkdir(parents=True, exist_ok=True)
                (r / sp / "labels").mkdir(parents=True, exist_ok=True)

        (src_root / "train" / "images" / "im1.jpg").write_bytes(b"data")
        (src_root / "train" / "labels" / "im1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (dst_root / "train" / "images" / "im1.jpg").write_bytes(b"data")

        res = verify_dataset_standardization(src_root, dst_root)
        self.assertFalse(res["verified"])
        self.assertIn("missing_lbls", str(res["failures"]).lower())

    def test_verify_only_detects_image_checksum_mismatch(self):
        src_root = Path(self.temp_dir) / "src"
        dst_root = Path(self.temp_dir) / "dst"
        for r in [src_root, dst_root]:
            for sp in ["train", "valid", "test"]:
                (r / sp / "images").mkdir(parents=True, exist_ok=True)
                (r / sp / "labels").mkdir(parents=True, exist_ok=True)

        (src_root / "train" / "images" / "im1.jpg").write_bytes(b"source_image_bytes")
        (src_root / "train" / "labels" / "im1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        (dst_root / "train" / "images" / "im1.jpg").write_bytes(b"corrupted_bytes")
        (dst_root / "train" / "labels" / "im1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        res = verify_dataset_standardization(src_root, dst_root)
        self.assertFalse(res["verified"])
        self.assertIn("mismatch", str(res["failures"]).lower())

    def test_verify_only_detects_orphan_destination_label(self):
        src_root = Path(self.temp_dir) / "src"
        dst_root = Path(self.temp_dir) / "dst"
        for r in [src_root, dst_root]:
            for sp in ["train", "valid", "test"]:
                (r / sp / "images").mkdir(parents=True, exist_ok=True)
                (r / sp / "labels").mkdir(parents=True, exist_ok=True)

        (src_root / "train" / "images" / "im1.jpg").write_bytes(b"data")
        (src_root / "train" / "labels" / "im1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        (dst_root / "train" / "images" / "im1.jpg").write_bytes(b"data")
        (dst_root / "train" / "labels" / "im1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (dst_root / "train" / "labels" / "im_orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        res = verify_dataset_standardization(src_root, dst_root)
        self.assertFalse(res["verified"])
        self.assertIn("orphan_lbls", str(res["failures"]).lower())

    def test_verify_only_detects_geometric_coordinate_mismatch(self):
        src_root = Path(self.temp_dir) / "src"
        dst_root = Path(self.temp_dir) / "dst"
        for r in [src_root, dst_root]:
            for sp in ["train", "valid", "test"]:
                (r / sp / "images").mkdir(parents=True, exist_ok=True)
                (r / sp / "labels").mkdir(parents=True, exist_ok=True)

        (src_root / "train" / "images" / "im1.jpg").write_bytes(b"data")
        (src_root / "train" / "labels" / "im1.txt").write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")

        (dst_root / "train" / "images" / "im1.jpg").write_bytes(b"data")
        (dst_root / "train" / "labels" / "im1.txt").write_text("0 0.550000 0.500000 0.100000 0.100000\n", encoding="utf-8")

        res = verify_dataset_standardization(src_root, dst_root)
        self.assertFalse(res["verified"])
        self.assertIn("Geometric mismatch", str(res["failures"]))


class TestUltralyticsDatasetIntegration(unittest.TestCase):
    """Suite 6: Lightweight Integration Test via Real Ultralytics Dataset / Collate."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        self.img_dir = Path(self.temp_dir) / "images"
        self.lbl_dir = Path(self.temp_dir) / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_data = {}
        for i in range(4):
            fn = f"sample_{i}.jpg"
            img_p = self.img_dir / fn
            lbl_p = self.lbl_dir / f"sample_{i}.txt"

            img = np.full((640, 640, 3), fill_value=i * 50, dtype=np.uint8)
            cv2.imwrite(str(img_p), img)

            lbl_p.write_text("0 0.5 0.5 0.05 0.05\n", encoding="utf-8")

            has_target = (i < 2)
            self.manifest_data[fn] = {
                "has_target_small": has_target,
                "target_small_box": "0.475,0.475,0.525,0.525",
                "box_format": "norm_xyxy",
                "img_width": 640,
                "img_height": 640,
                "target_gt_id": 0,
                "p_sampling": 0.25,
            }

        self.manifest_path = Path(self.temp_dir) / "manifest.csv"
        lines = ["image_id,d_image,p_sampling,has_gt,has_target_small,target_gt_id,target_small_di,target_small_box,box_format,img_width,img_height\n"]
        for fn, info in self.manifest_data.items():
            has_tg = "1" if info["has_target_small"] else "0"
            lines.append(f'{fn},0.8,{info["p_sampling"]},1,{has_tg},{info["target_gt_id"]},0.85,"{info["target_small_box"]}",{info["box_format"]},{info["img_width"]},{info["img_height"]}\n')
        self.manifest_path.write_text("".join(lines), encoding="utf-8")

        self.data_dict = {
            "names": {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "HEALTHY"},
            "nc": 5,
        }

    def test_ultralytics_dataset_item_retrieval_and_collate(self):
        dataset = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            data=self.data_dict,
            manifest_path=str(self.manifest_path),
            enable_mixture_sampling=True,
            enable_targeted_aug=True,
            aug_prob=1.0,
            augment=False,
        )
        self.assertEqual(len(dataset), 4)

        weights = dataset.get_sampling_weights()
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=4)

        batch_items = [dataset[i] for i in range(len(dataset))]
        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is not None:
            batch = collate_fn(batch_items)
            self.assertIn("img", batch)
            self.assertIn("bboxes", batch)
            self.assertIn("cls", batch)
            self.assertEqual(batch["img"].shape[0], 4)

    def test_targeted_augmentation_applied_when_conditions_met(self):
        dataset = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            data=self.data_dict,
            manifest_path=str(self.manifest_path),
            enable_targeted_aug=True,
            aug_prob=1.0,
            augment=False,
        )
        raw_lbl = dataset.get_image_and_label(0)
        tinfo = dataset.manifest_data["sample_0.jpg"]
        cropped = dataset.apply_context_crop(raw_lbl, tinfo)

        self.assertTrue(cropped.get("_is_context_cropped", False))
        self.assertEqual(cropped["img"].shape, (640, 640, 3))
        self.assertEqual(len(cropped["instances"]), 1)
        self.assertTrue(np.all(cropped["instances"].bboxes >= 0.0))
        self.assertTrue(np.all(cropped["instances"].bboxes <= 1.0))

    def test_original_labels_preserved_when_conditions_not_met(self):
        dataset = AdaptiveYOLODataset(
            img_path=str(self.img_dir),
            data=self.data_dict,
            manifest_path=str(self.manifest_path),
            enable_targeted_aug=True,
            aug_prob=1.0,
            augment=False,
        )
        raw_lbl = dataset.get_image_and_label(2)
        tinfo = dataset.manifest_data["sample_2.jpg"]
        orig_boxes = raw_lbl["instances"].bboxes.copy()

        result = dataset.apply_context_crop(raw_lbl, tinfo)
        self.assertFalse(result.get("_is_context_cropped", False))
        np.testing.assert_array_equal(result["instances"].bboxes, orig_boxes)


class TestMixtureSamplingFormulaAndRobustness(unittest.TestCase):
    """Suite 7: Mixture Sampling Formula P(k) = (1-lambda)/N + lambda*(D_k / sum(D))."""

    def test_formula_matches_exact_analytical_values(self):
        D = [0.2, 0.4, 0.6, 0.8]
        lam = 0.70
        probs = calculate_mixture_weights(D, mixture_lambda=lam)
        expected = [0.145, 0.215, 0.285, 0.355]
        np.testing.assert_allclose(probs, expected, atol=1e-5)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_sum_d_equals_zero_explicit_uniform_fallback(self):
        D = [0.0, 0.0, 0.0, 0.0]
        probs = calculate_mixture_weights(D, mixture_lambda=0.70)
        expected = [0.25, 0.25, 0.25, 0.25]
        np.testing.assert_allclose(probs, expected, atol=1e-5)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_invalid_lambda_raises_value_error(self):
        D = [0.5, 0.5]
        for bad_lam in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                calculate_mixture_weights(D, mixture_lambda=bad_lam)

    def test_negative_or_nan_difficulty_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_mixture_weights([0.5, -0.1], mixture_lambda=0.70)
        with self.assertRaises(ValueError):
            calculate_mixture_weights([0.5, float("nan")], mixture_lambda=0.70)

    def test_probabilities_are_strictly_positive(self):
        D = [0.0, 1.0]
        probs = calculate_mixture_weights(D, mixture_lambda=0.70)
        self.assertGreater(probs[0], 0.0)
        self.assertGreater(probs[1], 0.0)
        self.assertAlmostEqual(probs[0], 0.15, places=5)
        self.assertAlmostEqual(probs[1], 0.85, places=5)


if __name__ == "__main__":
    unittest.main()

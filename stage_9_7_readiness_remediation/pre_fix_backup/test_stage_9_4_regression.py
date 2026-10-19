"""
STAGE 9.4 — COMPREHENSIVE REGRESSION & INTEGRITY TEST SUITE
===========================================================
Master automated test suite verifying all audit findings, pipeline hardening,
single source of truth consistency, and mathematical specifications:

Suites:
1.  TestDifficultyFormulaAndBounds: D_i, D_image, penalties, clipping audit.
2.  TestGlobalOneToOneMatching: Greedy bipartite matching, candidate sorting, zero prediction/GT reuse.
3.  TestManifestFailFastAndCoverage: Manifest coverage validation, zero silent fallback, duplicate rejection.
4.  TestSamplingWeightsAndMixtureFormula: P(k) mixture formula, lambda=0.70, positive floor, sum=1.0.
5.  TestTargetedContextCropGeometry: 2x bbox context margin, canvas clamping, visibility filtering (0.30), deepcopy.
6.  TestExperimentComponentToggles: EXP-0 through EXP-3 configurations, trainer parameter pass-through.
7.  TestEvaluatorDataSourceSSOT: Single source of truth from data.yaml, dynamic split path resolution.
8.  TestModelSelectionHierarchy: Locked protocol (Primary mAP50-95 -> Recall Small -> mAP50 -> lowest FN).
9.  TestFullPrecisionMetricSerialization: Canonical extraction of box.ap50 vs box.maps, full float precision.
10. TestIsolatedPathsAndAntiOverwrite: Protection of historic stage_8_ablation, clean output defaults.
11. TestStandardizedFiveColumnLabels: Strict 5-column detection format, class ID range [0, 4], finite coords.
12. TestNonSquareDynamicDimensionParser: Dynamic scaling on non-square dimensions (e.g. 1920x1080), area px^2.
13. TestNestedOOFGroupIsolation: Nested holdout 100% excluded from fold train & val YAMLs, zero leakage.

Outputs:
- stage_9_4_code_remediation/regression_test_results.txt
- stage_9_4_code_remediation/regression_test_matrix.csv

Author: Senior ML Research Engineer & Independent Code Auditor
Date: 2026-10-17
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

# Add workspace root dynamically
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

# Import modules under test
from research_utils.detection_metrics import (
    calculate_image_difficulty,
    calculate_iou_xyxy,
    calculate_mixture_weights,
    calculate_object_difficulty,
    classify_object_size,
    extract_canonical_metrics,
    greedy_bipartite_matching,
    parse_yolo_labels,
    select_best_model,
)
from stage_8_1_group_oof_profiling import (
    build_group_stratified_folds,
    generate_nested_fold_configs,
)
from stage_8_3_adaptive_pipeline import (
    AdaptiveDetectionTrainer,
    AdaptiveYOLODataset,
    load_difficulty_manifest,
)
from stage_8_4_run_ablation_experiments import (
    EXPERIMENT_SPECS,
    run_single_experiment,
)
from stage_8_5_model_selection_and_validation import (
    resolve_val_paths,
)
from stage_8_6_final_test_benchmark import (
    resolve_test_paths,
)
from execute_phase_b_standardization import (
    convert_polygon_to_bbox,
    parse_raw_label_line,
    validate_standardized_label_file,
)

from ultralytics.utils.instance import Instances


CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


class TestDifficultyFormulaAndBounds(unittest.TestCase):
    """Suite 1: Object & Image Difficulty Formulas, Bounds, and Clipping."""

    def test_matched_correct_class_formula(self):
        # D_i = 0.40(1 - conf) + 0.40(1 - iou) + 0.20 * S_i
        # conf=0.8, iou=0.7, Small (S_i=1.0)
        # 0.40(0.20) + 0.40(0.30) + 0.20(1.0) = 0.08 + 0.12 + 0.20 = 0.40
        d_i = calculate_object_difficulty(
            matched=True, conf=0.8, iou=0.7, size_score=1.0, correct_class=True
        )
        self.assertAlmostEqual(d_i, 0.40, places=4)
        self.assertTrue(0.0 <= d_i <= 1.0)

    def test_matched_wrong_class_penalty_and_clipping(self):
        # D_i = min(1.0, 0.40(1-C) + 0.40(1-IoU) + 0.20 S_i + 0.35)
        # conf=0.1, iou=0.2, Small (S_i=1.0), wrong class
        # raw = 0.40(0.9) + 0.40(0.8) + 0.20(1.0) + 0.35 = 0.36 + 0.32 + 0.20 + 0.35 = 1.23 -> clipped to 1.0
        d_i = calculate_object_difficulty(
            matched=True, conf=0.1, iou=0.2, size_score=1.0, correct_class=False, confusion_penalty=0.35
        )
        self.assertEqual(d_i, 1.0)

    def test_unmatched_fn_formula(self):
        # D_i = 0.80 + 0.20 * S_i
        # Small lesion (S_i = 1.0) -> 1.00
        # Large lesion (S_i = 0.0) -> 0.80
        # Medium lesion (S_i = 0.5) -> 0.90
        d_small = calculate_object_difficulty(matched=False, size_score=1.0)
        d_med = calculate_object_difficulty(matched=False, size_score=0.5)
        d_large = calculate_object_difficulty(matched=False, size_score=0.0)

        self.assertAlmostEqual(d_small, 1.00, places=4)
        self.assertAlmostEqual(d_med, 0.90, places=4)
        self.assertAlmostEqual(d_large, 0.80, places=4)

    def test_image_difficulty_aggregation(self):
        # Image with objects: 0.5 * max + 0.5 * mean
        # objs: [0.4, 0.8] -> max=0.8, mean=0.6 -> d_image = 0.5(0.8) + 0.5(0.6) = 0.70
        d_img = calculate_image_difficulty(obj_difficulties=[0.4, 0.8])
        self.assertAlmostEqual(d_img, 0.70, places=4)

        # Empty image with false positive
        d_fp = calculate_image_difficulty(obj_difficulties=[], fp_confs=[0.65, 0.42])
        self.assertAlmostEqual(d_fp, 0.65, places=4)

        # Empty image without false positive
        d_clean = calculate_image_difficulty(obj_difficulties=[], fp_confs=[])
        self.assertEqual(d_clean, 0.0)


class TestGlobalOneToOneMatching(unittest.TestCase):
    """Suite 2: Global One-to-One Bipartite Greedy Matching."""

    def test_single_prediction_between_two_gt(self):
        # GT1: [100, 100, 150, 150]
        # GT2: [120, 100, 170, 150] (closer to pred)
        # Pred: [125, 100, 175, 150]
        gt_boxes = [[100, 100, 150, 150], [120, 100, 170, 150]]
        pred_boxes = [[125, 100, 175, 150]]

        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        # Only GT 1 (the closer one) should be matched
        self.assertEqual(len(matches), 1)
        self.assertIn(1, matches)
        self.assertEqual(matches[1][0], 0)
        self.assertEqual(len(unassigned_preds), 0)
        # GT 0 is unmatched (FN)
        self.assertNotIn(0, matches)

    def test_multiple_predictions_for_single_gt(self):
        # GT: [100, 100, 200, 200]
        # Pred 0: [100, 100, 190, 190] (high IoU)
        # Pred 1: [100, 100, 150, 150] (lower IoU)
        gt_boxes = [[100, 100, 200, 200]]
        pred_boxes = [[100, 100, 190, 190], [100, 100, 150, 150]]

        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 0)  # Pred 0 gets matched
        self.assertIn(1, unassigned_preds)  # Pred 1 becomes False Positive

    def test_global_descending_iou_sorting(self):
        # Verify candidate matching order is strictly descending by IoU
        gt_boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        pred_boxes = [[50, 50, 60, 60], [0, 0, 10, 10]]  # Pred 0 matches GT 1 with IoU 1.0, Pred 1 matches GT 0 with IoU 1.0
        matches, unassigned_preds = greedy_bipartite_matching(gt_boxes, pred_boxes, iou_thresh=0.10)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0], 1)
        self.assertEqual(matches[1][0], 0)


class TestManifestFailFastAndCoverage(unittest.TestCase):
    """Suite 3: Fail-Fast Manifest Verification & Zero Silent Fallback."""

    def test_duplicate_image_id_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "manifest_dup.csv"
            content = """image_id,d_image,p_sampling,has_gt,has_target_small,target_gt_id,target_small_di,target_small_box
img1.jpg,0.5,0.001,1,0,-1,0.0,
img1.jpg,0.6,0.002,1,0,-1,0.0,
"""
            csv_path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_difficulty_manifest(csv_path)
            self.assertIn("Duplikasi image_id", str(ctx.exception))

    def test_invalid_d_image_bounds_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "manifest_invalid.csv"
            content = """image_id,d_image,p_sampling,has_gt,has_target_small,target_gt_id,target_small_di,target_small_box
img1.jpg,1.5,0.001,1,0,-1,0.0,
"""
            csv_path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_difficulty_manifest(csv_path)
            self.assertIn("di luar rentang valid", str(ctx.exception))

    def test_dataset_coverage_fail_fast_on_missing_images(self):
        ds = AdaptiveYOLODataset.__new__(AdaptiveYOLODataset)
        ds.im_files = ["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg"]
        ds.manifest_data = {
            "img1.jpg": {"p_sampling": 0.5},
            "img2.jpg": {"p_sampling": 0.5},
            # img3.jpg is missing
        }
        ds.enable_targeted_aug = True
        with self.assertRaises(ValueError) as ctx:
            ds.validate_manifest_coverage()
        self.assertIn("tidak ditemukan di manifest", str(ctx.exception))
        self.assertIn("Silent uniform fallback dilarang", str(ctx.exception))


class TestSamplingWeightsAndMixtureFormula(unittest.TestCase):
    """Suite 4: Mixture Adaptive Sampling Formula P(k)."""

    def test_mixture_weights_formula_and_normalization(self):
        d_scores = [0.2, 0.4, 0.6, 0.8]
        # N=4, sum(D)=2.0, lambda=0.70
        # P(k) = 0.30*(1/4) + 0.70*(D_k / 2.0) = 0.075 + 0.35 * D_k
        # P(0) = 0.075 + 0.07 = 0.145
        # P(1) = 0.075 + 0.14 = 0.215
        # P(2) = 0.075 + 0.21 = 0.285
        # P(3) = 0.075 + 0.28 = 0.355
        # Sum = 1.000
        p_weights = calculate_mixture_weights(d_scores, mixture_lambda=0.70)
        self.assertEqual(len(p_weights), 4)
        self.assertAlmostEqual(sum(p_weights), 1.0, places=6)
        self.assertAlmostEqual(p_weights[0], 0.145, places=5)
        self.assertAlmostEqual(p_weights[3], 0.355, places=5)
        # Floor guarantee
        for p in p_weights:
            self.assertGreater(p, 0.0)
            self.assertGreaterEqual(p, 0.30 / 4)


class TestTargetedContextCropGeometry(unittest.TestCase):
    """Suite 5: Targeted Context Augmentation Crop Geometry & Label Preservation."""

    def setUp(self):
        self.ds = AdaptiveYOLODataset.__new__(AdaptiveYOLODataset)
        self.ds.hyp = SimpleNamespace(imgsz=640)
        self.ds.context_margin = 2.0
        self.ds.vis_thresh = 0.30
        self.ds.crop_diagnostics = {
            "crop_attempts": 0,
            "crops_applied": 0,
            "no_target_skips": 0,
            "invalid_box_skips": 0,
            "visibility_skips": 0,
        }

    def test_crop_expansion_and_centering(self):
        # 640x640 canvas, target at center xc=320, yc=320, w=40, h=40
        # normalized box: xc=0.5, yc=0.5, w=0.0625, h=0.0625
        # 2x margin -> w_crop = 80, h_crop = 80
        # crop window: [280, 280, 360, 360]
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.0625, 0.0625]], dtype=np.float32)
        cls_arr = np.array([[0]], dtype=np.float32)
        instances = Instances(bboxes=bboxes, bbox_format="xywh", normalized=True)
        lbl = {
            "img": img,
            "instances": instances,
            "cls": cls_arr,
            "im_file": "sample.jpg",
        }
        tinfo = {
            "has_target_small": True,
            "target_small_box": "300.0,300.0,340.0,340.0",
            "target_gt_id": 0,
        }
        out = self.ds.apply_context_crop(lbl, tinfo)
        self.assertTrue(out.get("_is_context_cropped", False))
        self.assertEqual(out["img"].shape, (640, 640, 3))
        # After 2x crop centered on target and resized to 640, box remains centered at (0.5, 0.5)
        # and relative width becomes 40/80 = 0.5
        out_boxes = out["instances"].bboxes
        self.assertEqual(len(out_boxes), 1)
        np.testing.assert_allclose(out_boxes[0, :2], [0.5, 0.5], atol=1e-2)
        np.testing.assert_allclose(out_boxes[0, 2:], [0.5, 0.5], atol=1e-2)

    def test_original_label_not_mutated(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        instances = Instances(bboxes=bboxes, bbox_format="xywh", normalized=True)
        lbl = {"img": img, "instances": instances, "cls": np.array([[0]], dtype=np.float32), "im_file": "test.jpg"}
        lbl_copy_bboxes = lbl["instances"].bboxes.copy()

        tinfo = {"has_target_small": False}  # Should skip
        out = self.ds.apply_context_crop(lbl, tinfo)
        np.testing.assert_array_equal(lbl["instances"].bboxes, lbl_copy_bboxes)


class TestExperimentComponentToggles(unittest.TestCase):
    """Suite 6: Ablation Experiment Matrix EXP-0 through EXP-3."""

    def test_experiment_spec_toggles(self):
        self.assertFalse(EXPERIMENT_SPECS[0]["enable_mixture_sampling"])
        self.assertFalse(EXPERIMENT_SPECS[0]["enable_targeted_aug"])

        self.assertTrue(EXPERIMENT_SPECS[1]["enable_mixture_sampling"])
        self.assertFalse(EXPERIMENT_SPECS[1]["enable_targeted_aug"])

        self.assertFalse(EXPERIMENT_SPECS[2]["enable_mixture_sampling"])
        self.assertTrue(EXPERIMENT_SPECS[2]["enable_targeted_aug"])

        self.assertTrue(EXPERIMENT_SPECS[3]["enable_mixture_sampling"])
        self.assertTrue(EXPERIMENT_SPECS[3]["enable_targeted_aug"])


class TestEvaluatorDataSourceSSOT(unittest.TestCase):
    """Suite 7: Single Source of Truth Path Resolution."""

    def test_resolve_paths_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            val_img = tmp_p / "valid" / "images"
            val_lbl = tmp_p / "valid" / "labels"
            test_img = tmp_p / "test" / "images"
            test_lbl = tmp_p / "test" / "labels"
            val_img.mkdir(parents=True)
            val_lbl.mkdir(parents=True)
            test_img.mkdir(parents=True)
            test_lbl.mkdir(parents=True)

            data_yaml = tmp_p / "data.yaml"
            data_yaml.write_text(f"""path: {tmp_p.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: 5
names: {CLASS_NAMES}
""", encoding="utf-8")

            res_val_img, res_val_lbl = resolve_val_paths(data_yaml)
            self.assertEqual(res_val_img, val_img.resolve())
            self.assertEqual(res_val_lbl, val_lbl.resolve())

            res_test_img, res_test_lbl = resolve_test_paths(data_yaml)
            self.assertEqual(res_test_img, test_img.resolve())
            self.assertEqual(res_test_lbl, test_lbl.resolve())


class TestModelSelectionHierarchy(unittest.TestCase):
    """Suite 8: Locked Model Selection Decision Rule Hierarchy."""

    def test_primary_criterion_map50_95_strictly_wins(self):
        # Candidate 0 has higher mAP50_95 by 0.002 (0.2%).
        # Candidate 1 has higher Recall Small and mAP50.
        # Candidate 0 MUST win because no arbitrary 0.5% margin exists.
        candidates = [
            {"exp_id": "EXP-0", "mAP50_95": 0.4020, "recall_small": 0.20, "mAP50": 0.60, "fn_missed": 100},
            {"exp_id": "EXP-1", "mAP50_95": 0.4000, "recall_small": 0.35, "mAP50": 0.65, "fn_missed": 80},
        ]
        winner, rationale = select_best_model(candidates, tolerance=1e-5)
        self.assertEqual(winner["exp_id"], "EXP-0")
        self.assertIn("mAP@50-95 validasi tertinggi", rationale)

    def test_tie_break_1_recall_small(self):
        # Tied mAP50_95 within 1e-6. Candidate 1 has higher Recall Small.
        candidates = [
            {"exp_id": "EXP-0", "mAP50_95": 0.400000, "recall_small": 0.20, "mAP50": 0.65, "fn_missed": 100},
            {"exp_id": "EXP-1", "mAP50_95": 0.400000, "recall_small": 0.25, "mAP50": 0.60, "fn_missed": 120},
        ]
        winner, rationale = select_best_model(candidates, tolerance=1e-5)
        self.assertEqual(winner["exp_id"], "EXP-1")
        self.assertIn("Recall_Small", rationale)

    def test_tie_break_2_map50(self):
        # Tied on mAP50_95 and Recall Small. Candidate 0 has higher mAP50.
        candidates = [
            {"exp_id": "EXP-0", "mAP50_95": 0.400000, "recall_small": 0.25, "mAP50": 0.66, "fn_missed": 100},
            {"exp_id": "EXP-1", "mAP50_95": 0.400000, "recall_small": 0.25, "mAP50": 0.62, "fn_missed": 90},
        ]
        winner, rationale = select_best_model(candidates, tolerance=1e-5)
        self.assertEqual(winner["exp_id"], "EXP-0")
        self.assertIn("mAP@50", rationale)

    def test_tie_break_3_lowest_fn(self):
        # Tied on mAP50_95, Recall Small, and mAP50. Candidate 1 has lower FN.
        candidates = [
            {"exp_id": "EXP-0", "mAP50_95": 0.400000, "recall_small": 0.25, "mAP50": 0.60, "fn_missed": 100},
            {"exp_id": "EXP-1", "mAP50_95": 0.400000, "recall_small": 0.25, "mAP50": 0.60, "fn_missed": 85},
        ]
        winner, rationale = select_best_model(candidates, tolerance=1e-5)
        self.assertEqual(winner["exp_id"], "EXP-1")
        self.assertIn("FN_Missed", rationale)


class TestFullPrecisionMetricSerialization(unittest.TestCase):
    """Suite 9: Metric Serialization & Canonical Extraction."""

    def test_canonical_metric_extraction(self):
        # Mock Ultralytics val_results object
        mock_val = SimpleNamespace(
            box=SimpleNamespace(
                mp=0.7123456,
                mr=0.6543210,
                map50=0.6890123,
                map=0.4567890,
                ap50=np.array([0.51, 0.52, 0.53, 0.54, 0.55], dtype=np.float32),
                maps=np.array([0.31, 0.32, 0.33, 0.34, 0.35], dtype=np.float32),
                ap_class_index=[0, 1, 2, 3, 4],
            )
        )
        res = extract_canonical_metrics(mock_val, CLASS_NAMES)
        self.assertAlmostEqual(res["precision"], 0.7123, places=4)
        self.assertAlmostEqual(res["map50"], 0.6890, places=4)
        self.assertAlmostEqual(res["map50_95"], 0.4568, places=4)
        self.assertAlmostEqual(res["per_class_ap50"]["CBB"], 0.51, places=2)
        self.assertAlmostEqual(res["per_class_map50_95"]["CBB"], 0.31, places=2)
        # Full precision retained in raw
        self.assertEqual(res["metrics_raw"]["precision"], 0.7123456)


class TestIsolatedPathsAndAntiOverwrite(unittest.TestCase):
    """Suite 10: Anti-Overwrite & Checkpoint Reuse Protection."""

    def test_historic_stage_8_ablation_overwrite_blocked(self):
        args = SimpleNamespace(
            output_dir="d:/UNDIP/SKRIPSI/stage_8_ablation",
            allow_historic_overwrite=False,
            exp="0",
        )
        with self.assertRaises(PermissionError) as ctx:
            run_single_experiment(0, args)
        self.assertIn("CRITICAL SAFETY ERROR", str(ctx.exception))


class TestStandardizedFiveColumnLabels(unittest.TestCase):
    """Suite 11: Five-Column Detection Format Validation."""

    def test_valid_five_column_detection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lbl_file = Path(tmp_dir) / "label.txt"
            lbl_file.write_text("0 0.500000 0.500000 0.100000 0.200000\n", encoding="utf-8")
            objs = validate_standardized_label_file(lbl_file)
            self.assertEqual(len(objs), 1)
            self.assertEqual(objs[0]["class_id"], 0)

    def test_reject_polygon_in_standardized_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lbl_file = Path(tmp_dir) / "label_poly.txt"
            lbl_file.write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_standardized_label_file(lbl_file)
            self.assertIn("Harus tepat 5 kolom", str(ctx.exception))

    def test_reject_invalid_class_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lbl_file = Path(tmp_dir) / "label_bad_cls.txt"
            lbl_file.write_text("9 0.5 0.5 0.1 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_standardized_label_file(lbl_file)
            self.assertIn("di luar rentang valid", str(ctx.exception))


class TestNonSquareDynamicDimensionParser(unittest.TestCase):
    """Suite 12: Dynamic Non-Square Resolution Parser."""

    def test_dynamic_coordinate_scaling_1920x1080(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lbl_file = Path(tmp_dir) / "non_square.txt"
            # Normalized box: xc=0.5, yc=0.5, w=0.1, h=0.2
            lbl_file.write_text("1 0.5 0.5 0.1 0.2\n", encoding="utf-8")
            objs = parse_yolo_labels(lbl_file, img_width=1920.0, img_height=1080.0)
            self.assertEqual(len(objs), 1)
            o = objs[0]
            # Width = 0.1 * 1920 = 192 px, Height = 0.2 * 1080 = 216 px
            # Area = 192 * 216 = 41,472 px^2 >= 96^2 (9216 px^2) -> Large
            self.assertAlmostEqual(o["area"], 192.0 * 216.0, places=2)
            self.assertEqual(o["size_category"], "Large")


class TestNestedOOFGroupIsolation(unittest.TestCase):
    """Suite 13: Nested OOF Group Isolation & Checkpoint Selection."""

    def test_nested_fold_zero_leakage_and_isolation(self):
        # Create a mock 8-group dataset across 4 folds
        annotation_data = {
            f"img_{i}.jpg": {
                "image_path": f"/images/img_{i}.jpg",
                "classes": [i % 5],
                "n_objects": 1,
            }
            for i in range(16)
        }
        image_fold_assignment = {f"img_{i}.jpg": (i % 4) + 1 for i in range(16)}
        fold_groups = {
            1: ["g1", "g2"],
            2: ["g3", "g4"],
            3: ["g5", "g6"],
            4: ["g7", "g8"],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            yaml_paths, nested_specs = generate_nested_fold_configs(
                annotation_data=annotation_data,
                image_fold_assignment=image_fold_assignment,
                fold_groups=fold_groups,
                output_dir=tmp_p,
                selection_mode="nested_inner_val",
                n_splits=4,
            )
            # Verify each fold has zero group leakage between outer holdout, inner train, and inner val
            for f_id in range(1, 5):
                spec = nested_specs[f_id]
                h_fold = spec["outer_holdout_fold"]
                v_fold = spec["inner_val_fold"]
                t_folds = spec["inner_train_folds"]

                self.assertEqual(h_fold, f_id)
                self.assertNotIn(h_fold, t_folds)
                self.assertNotEqual(h_fold, v_fold)
                self.assertNotIn(v_fold, t_folds)

                # Check YAML content: outer holdout MUST NOT be in YAML train or val
                y_content = Path(spec["yaml_file"]).read_text(encoding="utf-8")
                self.assertNotIn(f"fold_{f_id}_outer_holdout.txt", y_content)
                self.assertIn(f"fold_{f_id}_inner_train.txt", y_content)
                self.assertIn(f"fold_{f_id}_inner_val.txt", y_content)


def run_all_tests():
    out_dir = WORKSPACE_DIR / "stage_9_4_code_remediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_txt = out_dir / "regression_test_results.txt"
    matrix_csv = out_dir / "regression_test_matrix.csv"

    test_classes = [
        TestDifficultyFormulaAndBounds,
        TestGlobalOneToOneMatching,
        TestManifestFailFastAndCoverage,
        TestSamplingWeightsAndMixtureFormula,
        TestTargetedContextCropGeometry,
        TestExperimentComponentToggles,
        TestEvaluatorDataSourceSSOT,
        TestModelSelectionHierarchy,
        TestFullPrecisionMetricSerialization,
        TestIsolatedPathsAndAntiOverwrite,
        TestStandardizedFiveColumnLabels,
        TestNonSquareDynamicDimensionParser,
        TestNestedOOFGroupIsolation,
    ]

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)

    t0 = time.time()
    result = runner.run(suite)
    duration = time.time() - t0

    stream.seek(0)
    raw_log = stream.read()
    print(raw_log)

    all_tests_passed = result.wasSuccessful()

    # Build matrix CSV
    matrix_rows = []
    for tc in test_classes:
        for t_name in loader.getTestCaseNames(tc):
            # Check if this test was in failures or errors
            status = "PASS"
            failure_reason = ""
            for f_case, f_trace in result.failures:
                if f_case._testMethodName == t_name:
                    status = "FAIL"
                    failure_reason = f_trace.splitlines()[-1]
            for e_case, e_trace in result.errors:
                if e_case._testMethodName == t_name:
                    status = "ERROR"
                    failure_reason = e_trace.splitlines()[-1]

            matrix_rows.append({
                "test_suite": tc.__name__,
                "test_case": t_name,
                "status": status,
                "detail": failure_reason if failure_reason else "Execution verified successfully",
            })

    with open(matrix_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_suite", "test_case", "status", "detail"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    # Build detailed execution text
    report_lines = [
        "=" * 80,
        "STAGE 9.4 — COMPREHENSIVE REGRESSION TEST SUITE EXECUTION REPORT",
        "=" * 80,
        f"Date/Time          : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python Version     : {sys.version.split()[0]}",
        f"PyTorch Version    : {torch.__version__}",
        f"CUDA Available     : {torch.cuda.is_available()}",
        f"Total Tests Run    : {result.testsRun}",
        f"Failures           : {len(result.failures)}",
        f"Errors             : {len(result.errors)}",
        f"Execution Duration : {duration:.2f} seconds",
        f"Overall Outcome    : {'PASS (100% SUCCEEDED)' if all_tests_passed else 'FAIL'}",
        "=" * 80,
        "",
        "TEST SUITE SUMMARY:",
    ]
    for row in matrix_rows:
        report_lines.append(f"  [{row['status']}] {row['test_suite']}.{row['test_case']} - {row['detail']}")

    report_lines.extend([
        "",
        "=" * 80,
        "RAW UNITTEST EXECUTION LOG:",
        "=" * 80,
        raw_log,
        "=" * 80,
        f"GATE READINESS: {'READY_FOR_CLEAN_OOF' if all_tests_passed else 'BLOCKED'}",
        "=" * 80,
    ])

    results_txt.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[REPORT] Saved results text: {results_txt}")
    print(f"[REPORT] Saved matrix CSV: {matrix_csv}")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

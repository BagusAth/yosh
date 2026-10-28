#!/usr/bin/env python3
"""
Test Suite: test_fase1_v2_regression.py
Fase 1 (Pakem v1.2) — Pencocokan v2, skor kualitas label, mode EXP-0R, log crop
================================================================================

Group A (hanya numpy, bisa dijalankan di mana saja):
  A1  Kotak junk ber-IoU tinggi tidak lagi "mencuri" TP (bug utama v1)
  A2  Salah kelas yang riil terdeteksi dan diberi penalti
  A3  Localization error, Both, dan Missed sesuai urutan prioritas
  A4  Kandidat berkelas sama ber-confidence rendah tetap dipakai untuk skor (bukan rumus FN)
  A5  Pencocokan TP satu-satu dan berurutan menurut confidence
  A6  Input kosong (tanpa GT / tanpa prediksi)
  A7  Skor kualitas label: anotasi hilang, kelas tertukar, kotak meleset, label bersih
Group B (butuh torch + ultralytics, jalankan di .venv-cuda):
  B1  Spesifikasi EXP-0R dan identitas run
  B2  Identitas EXP-0..3 tidak berubah oleh patch Fase 1
  B3  Log crop per proses menulis CSV dengan kolom yang benar
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from research_utils.detection_metrics import (
    TAXONOMY_BOTH,
    TAXONOMY_CLS,
    TAXONOMY_LOC,
    TAXONOMY_MISSED,
    TAXONOMY_TP,
    calculate_object_difficulty_v2,
    compute_image_label_quality,
    greedy_bipartite_matching,
    match_detections_v2,
)

try:
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def gt(box, cls):
    return {"box_xyxy": list(box), "class_id": cls}


def pr(box, cls, conf):
    return {"box_xyxy": list(box), "class_id": cls, "confidence": conf}


BOX = (100, 100, 200, 200)


class TestMatchingV2(unittest.TestCase):
    def test_a1_junk_box_does_not_steal_tp(self):
        gts = [gt(BOX, 2)]
        preds = [pr((105, 105, 205, 205), 2, 0.80), pr((101, 101, 201, 201), 4, 0.004)]
        r = match_detections_v2(gts, preds)["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_TP)
        self.assertAlmostEqual(r["c_star"], 0.80, places=6)
        self.assertFalse(r["wrong_class"])
        # D_i v2: 0.4*(1-0.8) + 0.4*(1-IoU) + 0 -> jauh di bawah 0.5
        self.assertLess(calculate_object_difficulty_v2(r, size_score=0.0), 0.2)
        # Bukti bug v1: pencocokan v1 memilih kotak junk (IoU tertinggi) -> indeks 1
        matches, _ = greedy_bipartite_matching(
            gt_boxes=[g["box_xyxy"] for g in gts], pred_boxes=[p["box_xyxy"] for p in preds], iou_thresh=0.10
        )
        self.assertEqual(matches[0][0], 1)

    def test_a2_real_confusion_flagged_and_penalized(self):
        gts = [gt(BOX, 2)]
        preds = [pr(BOX, 3, 0.60), pr(BOX, 2, 0.10)]
        r = match_detections_v2(gts, preds)["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_CLS)
        self.assertTrue(r["wrong_class"])
        self.assertAlmostEqual(r["c_star"], 0.10, places=6)
        expected = min(1.0, 0.4 * 0.9 + 0.4 * 0.0 + 0.2 * 0.5 + 0.35)
        self.assertAlmostEqual(calculate_object_difficulty_v2(r, size_score=0.5), expected, places=6)

    def test_a2b_no_penalty_when_other_class_below_tau(self):
        gts = [gt(BOX, 2)]
        preds = [pr(BOX, 3, 0.20), pr(BOX, 2, 0.10)]
        r = match_detections_v2(gts, preds, tau=0.25)["per_gt"][0]
        self.assertFalse(r["wrong_class"])
        self.assertEqual(r["taxonomy"], TAXONOMY_MISSED)

    def test_a3_localization_both_and_missed(self):
        # Localization: kelas sama, IoU ~0.33
        r = match_detections_v2([gt(BOX, 1)], [pr((150, 100, 250, 200), 1, 0.50)])["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_LOC)
        self.assertAlmostEqual(r["error_pred_conf"], 0.50, places=6)
        # Both: kelas lain, IoU ~0.33
        r = match_detections_v2([gt(BOX, 1)], [pr((150, 100, 250, 200), 4, 0.50)])["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_BOTH)
        # Missed: tidak ada prediksi
        r = match_detections_v2([gt(BOX, 1)], [])["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_MISSED)
        self.assertFalse(r["has_same_class_candidate"])
        self.assertAlmostEqual(calculate_object_difficulty_v2(r, size_score=1.0), 1.0, places=6)
        self.assertAlmostEqual(calculate_object_difficulty_v2(r, size_score=0.0), 0.80, places=6)

    def test_a3b_localization_has_priority_over_classification(self):
        preds = [pr((150, 100, 250, 200), 1, 0.40), pr(BOX, 3, 0.90)]
        r = match_detections_v2([gt(BOX, 1)], preds)["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_LOC)
        self.assertTrue(r["wrong_class"])  # bukti salah kelas tetap tercatat untuk penalti

    def test_a4_low_conf_same_class_candidate_used_for_score(self):
        r = match_detections_v2([gt(BOX, 0)], [pr((110, 100, 210, 200), 0, 0.10)])["per_gt"][0]
        self.assertEqual(r["taxonomy"], TAXONOMY_MISSED)
        self.assertTrue(r["has_same_class_candidate"])
        iou = r["iou_star"]
        expected = 0.4 * 0.9 + 0.4 * (1.0 - iou) + 0.2 * 0.5
        self.assertAlmostEqual(calculate_object_difficulty_v2(r, size_score=0.5), expected, places=6)

    def test_a5_one_to_one_and_confidence_order(self):
        # Dua GT berkelas sama yang bertumpuk, satu prediksi -> hanya satu TP
        gts = [gt(BOX, 2), gt((102, 102, 202, 202), 2)]
        res = match_detections_v2(gts, [pr((101, 101, 201, 201), 2, 0.9)])
        taxo = [r["taxonomy"] for r in res["per_gt"]]
        self.assertEqual(taxo.count(TAXONOMY_TP), 1)
        self.assertEqual(len(res["tp_pred_indices"]), 1)
        # Satu GT, dua prediksi kelas sama: yang confidence-nya lebih tinggi menang walau IoU-nya lebih kecil
        preds = [pr((100, 100, 200, 200), 2, 0.30), pr((100, 100, 180, 200), 2, 0.70)]
        r = match_detections_v2([gt(BOX, 2)], preds)["per_gt"][0]
        self.assertEqual(r["tp_pred"], 1)
        self.assertAlmostEqual(r["c_star"], 0.70, places=6)

    def test_a6_empty_inputs(self):
        res = match_detections_v2([], [pr(BOX, 1, 0.9)])
        self.assertEqual(res["per_gt"], [])
        self.assertEqual(res["unmatched_pred_indices"], [0])
        res = match_detections_v2([], [])
        self.assertEqual(res["per_gt"], [])

    def test_a6b_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            match_detections_v2([gt(BOX, 1)], [], tau=1.5)
        with self.assertRaises(ValueError):
            match_detections_v2([gt(BOX, 1)], [], tp_iou=0.3, cand_iou=0.5)


class TestLabelQuality(unittest.TestCase):
    def test_a7_clean_label(self):
        gts = [gt(BOX, 2)]
        preds = [pr((102, 102, 202, 202), 2, 0.9)]
        lq = compute_image_label_quality(gts, preds, match_detections_v2(gts, preds))
        self.assertAlmostEqual(lq["label_quality"], 1.0, places=6)
        self.assertEqual(lq["n_suspect_missing"], 0)

    def test_a7_missing_annotation(self):
        gts = [gt(BOX, 2)]
        preds = [pr((102, 102, 202, 202), 2, 0.9), pr((400, 400, 500, 500), 4, 0.8)]
        lq = compute_image_label_quality(gts, preds, match_detections_v2(gts, preds))
        self.assertAlmostEqual(lq["missing_score"], 0.8, places=6)
        self.assertEqual(lq["n_suspect_missing"], 1)
        self.assertAlmostEqual(lq["label_quality"], 0.2, places=6)

    def test_a7_background_image_with_confident_detection(self):
        preds = [pr((400, 400, 500, 500), 4, 0.7)]
        lq = compute_image_label_quality([], preds, match_detections_v2([], preds))
        self.assertAlmostEqual(lq["missing_score"], 0.7, places=6)

    def test_a7_swap_and_badloc(self):
        gts = [gt(BOX, 2)]
        preds = [pr(BOX, 3, 0.70), pr(BOX, 2, 0.20)]
        lq = compute_image_label_quality(gts, preds, match_detections_v2(gts, preds))
        self.assertAlmostEqual(lq["swap_score"], 0.50, places=6)
        gts = [gt(BOX, 1)]
        preds = [pr((150, 100, 250, 200), 1, 0.60)]
        lq = compute_image_label_quality(gts, preds, match_detections_v2(gts, preds))
        self.assertAlmostEqual(lq["badloc_score"], 0.60, places=6)
        self.assertEqual(lq["n_suspect_badloc"], 1)


@unittest.skipUnless(HAS_TORCH, "Butuh torch + ultralytics (jalankan di .venv-cuda)")
class TestPipelineFase1(unittest.TestCase):
    def test_b1_exp0r_spec_and_identity(self):
        from stage_8_4_run_ablation_experiments import EXPERIMENT_SPECS, build_experiment_run_identity

        spec = EXPERIMENT_SPECS[4]
        self.assertEqual(spec["id"], "EXP-0R")
        self.assertTrue(spec["enable_uniform_replacement"])
        self.assertFalse(spec["enable_mixture_sampling"])
        self.assertFalse(spec["enable_targeted_aug"])
        ident = build_experiment_run_identity(
            exp_id="EXP-0R", spec=spec, script_path=None, yaml_path=None, manifest_path=None,
            dataset_inventory_fingerprint="x", pretrained_weights_path=None, epochs=1, batch_size=2,
            imgsz=64, seed=42, device="cpu", workers=0, amp=False, plots=False, patience=1,
            close_mosaic=0, optimizer="auto", aug_prob=0.5, raw_oof_sha256="", clean_oof_run_id="",
            total_train_images=1, total_val_images=1,
        )
        self.assertTrue(ident.get("enable_uniform_replacement"))

    def test_b2_identity_unchanged_for_original_experiments(self):
        from stage_8_4_run_ablation_experiments import EXPERIMENT_SPECS, build_experiment_run_identity

        for k in (0, 1, 2, 3):
            spec = EXPERIMENT_SPECS[k]
            self.assertNotIn("enable_uniform_replacement", spec)
            ident = build_experiment_run_identity(
                exp_id=spec["id"], spec=spec, script_path=None, yaml_path=None, manifest_path=None,
                dataset_inventory_fingerprint="x", pretrained_weights_path=None, epochs=1, batch_size=2,
                imgsz=64, seed=42, device="cpu", workers=0, amp=False, plots=False, patience=1,
                close_mosaic=0, optimizer="auto", aug_prob=0.5, raw_oof_sha256="", clean_oof_run_id="",
                total_train_images=1, total_val_images=1,
            )
            self.assertNotIn("enable_uniform_replacement", ident)

    def test_b3_crop_event_log_written(self):
        from stage_8_3_adaptive_pipeline import CROP_LOG_FIELDS, AdaptiveYOLODataset

        ds = object.__new__(AdaptiveYOLODataset)  # tanpa memuat citra
        with tempfile.TemporaryDirectory() as td:
            ds.crop_log_dir = Path(td) / "crop_logs"
            ds._log_crop_event({"time": "1.0", "image_id": "a.jpg", "event": "applied", "zoom_x": "10.000"})
            ds._log_crop_event({"time": "2.0", "image_id": "b.jpg", "event": "skipped"})
            files = list(ds.crop_log_dir.glob("crop_events_pid*.csv"))
            self.assertEqual(len(files), 1)
            with open(files[0], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), CROP_LOG_FIELDS)
            self.assertEqual([r["event"] for r in rows], ["applied", "skipped"])
            self.assertEqual(rows[0]["zoom_x"], "10.000")


if __name__ == "__main__":
    unittest.main(verbosity=2)

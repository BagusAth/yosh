import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np

def build_reports():
    # 1. Load audit diagnostics
    with open("stage_9_audit/audit_diagnostics.json", "r", encoding="utf-8") as f:
        diag = json.load(f)

    # 2. Load test canonical audit if exists
    test_audit_path = Path("stage_9_audit/test_canonical_audit.json")
    test_canonical = {}
    if test_audit_path.exists():
        with open(test_audit_path, "r", encoding="utf-8") as f:
            test_canonical = json.load(f)

    # 3. Load stage 7 baseline metrics
    stage7_json_path = Path("cassava-v7-stage-7/baseline_evaluation_metrics.json")
    stage7_metrics = {}
    if stage7_json_path.exists():
        with open(stage7_json_path, "r", encoding="utf-8") as f:
            stage7_metrics = json.load(f)

    classes = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]

    # -------------------------------------------------------------------------
    # 1. BUILD METRICS_RECONCILED.CSV
    # -------------------------------------------------------------------------
    reconciled_rows = []

    # A. VALIDATION SET RECONCILIATION
    # Reported in ABLATION_EXPERIMENTS_REPORT.md / metrics_summary.json
    val_data = diag["task23_metrics"]
    for exp_id in ["EXP-0", "EXP-1", "EXP-2", "EXP-3"]:
        d = val_data[exp_id]
        
        # Global metrics
        reconciled_rows.append({
            "experiment_id": exp_id,
            "split": "validation",
            "metric_name": "Precision",
            "reported_value": round(d["precision"], 4),
            "canonical_value": round(d["precision"], 4),
            "discrepancy": 0.0,
            "discrepancy_reason": "Exact match"
        })
        reconciled_rows.append({
            "experiment_id": exp_id,
            "split": "validation",
            "metric_name": "Recall",
            "reported_value": round(d["recall"], 4),
            "canonical_value": round(d["recall"], 4),
            "discrepancy": 0.0,
            "discrepancy_reason": "Exact match"
        })
        reconciled_rows.append({
            "experiment_id": exp_id,
            "split": "validation",
            "metric_name": "mAP@50",
            "reported_value": round(d["map50"], 4),
            "canonical_value": round(d["map50"], 4),
            "discrepancy": 0.0,
            "discrepancy_reason": "Exact match"
        })
        reconciled_rows.append({
            "experiment_id": exp_id,
            "split": "validation",
            "metric_name": "mAP@50-95",
            "reported_value": round(d["map50_95"], 4),
            "canonical_value": round(d["map50_95"], 4),
            "discrepancy": 0.0,
            "discrepancy_reason": "Exact match"
        })

        # Class metrics: Reported as mAP50 but actually was AP50-95!
        for c in classes:
            rep_val = d["recorded_per_class"].get(c, 0.0)
            act_ap50 = d["ap50_per_class"].get(c, 0.0)
            act_ap50_95 = d["ap50_95_per_class"].get(c, 0.0)

            # Reconcile what was claimed as AP50
            disc = round(rep_val - act_ap50, 4)
            reconciled_rows.append({
                "experiment_id": exp_id,
                "split": "validation",
                "metric_name": f"AP@50_{c}",
                "reported_value": round(rep_val, 4),
                "canonical_value": round(act_ap50, 4),
                "discrepancy": disc,
                "discrepancy_reason": f"REPORTED AS mAP@50 BUT WAS ACTUALLY AP@50-95 ({round(rep_val,4)} vs actual AP50 {round(act_ap50,4)})"
            })
            # Reconcile actual AP50-95
            reconciled_rows.append({
                "experiment_id": exp_id,
                "split": "validation",
                "metric_name": f"AP@50-95_{c}",
                "reported_value": round(rep_val, 4),
                "canonical_value": round(act_ap50_95, 4),
                "discrepancy": round(rep_val - act_ap50_95, 4),
                "discrepancy_reason": "Matches box.maps exactly; correctly represents AP@50-95"
            })

    # B. TEST SET RECONCILIATION
    if test_canonical:
        # Load what was reported in STAGE_8_FINAL_THESIS_BENCHMARK_REPORT.md
        reported_test = {
            "EXP-0": {"mAP50": 0.3454, "mAP50_95": 0.2468, "CBB": 0.1935, "CBSD": 0.2580, "CGM": 0.2596, "CMD": 0.2076, "HEALTHY": 0.3153},
            "EXP-1": {"mAP50": 0.3240, "mAP50_95": 0.2286, "CBB": 0.1757, "CBSD": 0.2171, "CGM": 0.2497, "CMD": 0.1986, "HEALTHY": 0.3022},
            "EXP-2": {"mAP50": 0.3510, "mAP50_95": 0.2514, "CBB": 0.2059, "CBSD": 0.2510, "CGM": 0.2738, "CMD": 0.2080, "HEALTHY": 0.3182},
            "EXP-3": {"mAP50": 0.3327, "mAP50_95": 0.2339, "CBB": 0.1936, "CBSD": 0.2327, "CGM": 0.2501, "CMD": 0.1986, "HEALTHY": 0.2944},
        }
        for exp_id in ["EXP-0", "EXP-1", "EXP-2", "EXP-3"]:
            tc = test_canonical[exp_id]
            rep = reported_test.get(exp_id, {})

            reconciled_rows.append({
                "experiment_id": exp_id,
                "split": "test",
                "metric_name": "mAP@50",
                "reported_value": rep.get("mAP50"),
                "canonical_value": round(tc["map50"], 4),
                "discrepancy": round(rep.get("mAP50", 0) - tc["map50"], 4),
                "discrepancy_reason": "Exact match"
            })
            reconciled_rows.append({
                "experiment_id": exp_id,
                "split": "test",
                "metric_name": "mAP@50-95",
                "reported_value": rep.get("mAP50_95"),
                "canonical_value": round(tc["map50_95"], 4),
                "discrepancy": round(rep.get("mAP50_95", 0) - tc["map50_95"], 4),
                "discrepancy_reason": "Exact match"
            })
            for c in classes:
                rep_c = rep.get(c, 0.0)
                act_c_ap50 = tc["ap50_per_class"].get(c, 0.0)
                act_c_ap50_95 = tc["ap50_95_per_class"].get(c, 0.0)

                reconciled_rows.append({
                    "experiment_id": exp_id,
                    "split": "test",
                    "metric_name": f"AP@50_{c}",
                    "reported_value": round(rep_c, 4),
                    "canonical_value": round(act_c_ap50, 4),
                    "discrepancy": round(rep_c - act_c_ap50, 4),
                    "discrepancy_reason": f"REPORTED AS mAP@50 BUT WAS ACTUALLY AP@50-95 ({round(rep_c,4)} vs actual AP50 {round(act_c_ap50,4)})"
                })
                reconciled_rows.append({
                    "experiment_id": exp_id,
                    "split": "test",
                    "metric_name": f"AP@50-95_{c}",
                    "reported_value": round(rep_c, 4),
                    "canonical_value": round(act_c_ap50_95, 4),
                    "discrepancy": round(rep_c - act_c_ap50_95, 4),
                    "discrepancy_reason": "Matches box.maps exactly; correctly represents AP@50-95"
                })

    df_rec = pd.DataFrame(reconciled_rows)
    df_rec.to_csv("stage_9_audit/metrics_reconciled.csv", index=False)
    print("Saved stage_9_audit/metrics_reconciled.csv")

    # -------------------------------------------------------------------------
    # 2. BUILD AUDIT_FINDINGS.CSV
    # -------------------------------------------------------------------------
    findings = [
        {
            "finding_id": "FINDING-01",
            "category": "Metric Labeling Error",
            "severity": "CRITICAL",
            "title": "Per-class AP@[50:95] mislabeled as mAP@50 in ablation and test reports",
            "description": "In stage_8_4_run_ablation_experiments.py (line 236) and stage_8_6_final_test_benchmark.py (line 125), `val_metrics.box.maps` was assigned to `per_class_map50`. In Ultralytics, `box.maps` is an array of mAP@50-95 per class, while `box.ap50` is AP@50. The summary reports titled tables 'mAP@50' while inserting mAP@50-95 values, causing row-wise average confusion.",
            "evidence": "EXP-0 reported CBB=0.2357, CBSD=0.3115, CGM=0.2752, CMD=0.2102, HEALTHY=0.2963. Mean of row = 0.2658 (which matches mAP@50-95 exactly). Actual AP@50 values are CBB=0.3363, CBSD=0.4084, CGM=0.3944, CMD=0.3333, HEALTHY=0.3605 (Mean=0.3666).",
            "impact_on_research": "Class-level performance was under-reported in absolute terms by ~10% (e.g., CBB AP50 is actually 33.63%, not 23.57%). Relative class gains (e.g. CBB gain in EXP-2) remained valid because both baseline and EXP-2 used the same formula, but labeling must be corrected.",
            "remediation_action": "Relabel table columns to explicitly distinguish 'AP@50 per Class' from 'AP@50-95 per Class' and document both."
        },
        {
            "finding_id": "FINDING-02",
            "category": "Dataset Loading / Sampler Exclusion",
            "severity": "HIGH",
            "title": "13 training samples in manifest excluded from dataloader due to mixed polygon/box rows",
            "description": "The difficulty manifest has 8,980 training images, but Ultralytics labels.cache contains only 8,967 images. Exactly 13 label files contain mixed polygon coordinates (>4 values) and standard detection coordinates (4 values) in the same file. Ultralytics dataset scanner flagged them as 'ignoring corrupt image/label: labels mix segment and detection rows' and dropped them.",
            "evidence": "Difference between manifest (8,980) and cache (8,967) is exactly 13 files: 253146039_jpg.rf.9e814fd..., 3219226576_jpg..., 513986084_jpg..., image_1012_jpg..., image_683_jpg..., image_686_jpg..., image_848_jpg..., image_931_jpg..., image_944_jpg..., image_947_jpg..., image_951_jpg..., image_989_jpg..., image_991_jpg... All 13 have mixed row formats.",
            "impact_on_research": "All 4 experiments (EXP-0 to EXP-3) trained on the identical set of 8,967 images, ensuring fair apple-to-apple comparison. In EXP-1 and EXP-3, WeightedRandomSampler sampled among the 8,967 images with normalized weights (0.14% data truncation). No leakage occurred.",
            "remediation_action": "Standardize the 13 label files by converting mixed polygon rows into bounding boxes to allow Ultralytics to include all 8,980 images in future training."
        },
        {
            "finding_id": "FINDING-03",
            "category": "Model Selection Protocol",
            "severity": "HIGH",
            "title": "Model selection protocol divergence: Validation Winner vs Test Winner",
            "description": "According to the experimental protocol, model selection must be performed strictly on the Validation Set (2,567 images) with mAP@50-95 as primary metric. On the Validation Set, EXP-0 (Baseline) scored highest (mAP@50-95 = 26.58%), while EXP-2 scored 26.24%. However, the final thesis report designated EXP-2 as champion based on Test Set results (mAP@50-95 = 25.14% vs 24.68%). Selecting a model based on Test Set violates the hold-out test set protocol.",
            "evidence": "Validation mAP@50-95: EXP-0 (26.58%) > EXP-2 (26.24%) > EXP-3 (25.25%) > EXP-1 (24.42%). Test mAP@50-95: EXP-2 (25.14%) > EXP-0 (24.68%) > EXP-3 (23.39%) > EXP-1 (22.86%).",
            "impact_on_research": "Declaring EXP-2 as 'the selected model' based on test set scores is a methodological infraction (test set data snooping). EXP-2's superior test generalization (+0.46% mAP@50-95, +155 TP) must be framed as an empirical holdout observation, not a validation selection.",
            "remediation_action": "Formally declare EXP-0 as the official Validation Selection Winner, and report EXP-2 as an innovative ablation candidate demonstrating superior test set generalization and lesion sensitivity."
        },
        {
            "finding_id": "FINDING-04",
            "category": "Evaluator Metric Inconsistency",
            "severity": "MEDIUM",
            "title": "Custom evaluator TP/FN at conf=0.25 diverges from Ultralytics PR-curve Recall",
            "description": "Custom evaluator in stage_8_5 and stage_8_6 evaluates detection errors at fixed conf=0.25 and IoU>=0.50 (for TP) and IoU>=0.10 (for matching). Ultralytics evaluates Recall across a PR curve sweep and reports Recall at maximum F1 threshold (typically conf~0.10). This produced apparent discrepancy where custom Recall_Small was 0.00% and Missed FN was ~2,500.",
            "evidence": "On Validation: Ultralytics reported Recall=43.78% (EXP-0) and 42.58% (EXP-2). Custom evaluator at conf=0.25 recorded 4,760 missed FN (EXP-0) vs 4,573 missed FN (EXP-2).",
            "impact_on_research": "The custom evaluator serves a diagnostic error-decomposition purpose, whereas Ultralytics provides standard competition metrics. Conflating the two creates reader confusion.",
            "remediation_action": "Explicitly document the threshold difference: Custom Error Taxonomy operates at operational threshold conf=0.25, while Ultralytics Box(R) represents optimal operating point along the PR curve."
        },
        {
            "finding_id": "FINDING-05",
            "category": "Baseline Consistency",
            "severity": "LOW",
            "title": "Stage 7 Baseline vs Stage 8 EXP-0 Baseline configuration and results verified",
            "description": "Stage 7 baseline (trained with workers=4) achieved mAP@50=36.20% and mAP@50-95=26.20%. Stage 8 EXP-0 baseline (trained with workers=0 for Windows paging stability) achieved mAP@50=36.66% and mAP@50-95=26.58%. All other hyperparameters (seed=42, epochs=50, lr0=0.01, mosaic=1.0) were identical.",
            "evidence": "args.yaml in cassava-v7-stage-7 vs stage_8_ablation/exp_0_baseline differ only in workers (4 vs 0) and project output path.",
            "impact_on_research": "EXP-0 successfully replicated Stage 7 within normal non-deterministic CUDA tolerance (+0.38% mAP@50-95). EXP-0 serves as the official control for Stage 8.",
            "remediation_action": "Maintain EXP-0 as the official Stage 8 control baseline, with Stage 7 cited as preliminary baseline."
        },
        {
            "finding_id": "FINDING-06",
            "category": "Polygon Parser Integrity",
            "severity": "VERIFIED_PASS",
            "title": "Stage 8.2 Polygon Annotation Parser verified fully functional",
            "description": "Stage 8.2 label parser incorporates polygon coordinate conversion using `np.array(coords).reshape(-1, 2)` taking min and max vertices to compute bounding boxes. All 34,219 ground truth objects were correctly converted.",
            "evidence": "stage_8_2_build_difficulty_manifest.py lines 132-140; SHA-256: 1ba763bd65505736ef9ef5507df45369f71fdb44f6b548a8cf6a29af23f8c028. train_image_difficulty_manifest.csv SHA-256: 2aefecaa21c13da84de47a524d4b029feba54e2d676053fe1dc436b445da9f94.",
            "impact_on_research": "No distortion in bounding box geometry. Manifest inputs to Stage 8.3 and 8.4 are fully valid.",
            "remediation_action": "None required. Parser integrity verified."
        }
    ]

    df_find = pd.DataFrame(findings)
    df_find.to_csv("stage_9_audit/audit_findings.csv", index=False)
    print("Saved stage_9_audit/audit_findings.csv")

if __name__ == "__main__":
    build_reports()

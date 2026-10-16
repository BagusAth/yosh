import os
import csv
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def build_traceability_csv():
    records = [
        {
            "evaluation_run": "Stage 8.5 Validation Custom Error Taxonomy (EXP-0)",
            "checkpoint": "stage_8_ablation/exp_0_baseline/weights/best.pt",
            "split": "validation",
            "total_images": 2567,
            "total_gt_instances": 9632,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1776,
            "false_negatives_fn": 4760,
            "localization_error": 947,
            "classification_confusion": 249,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.4378 (43.78% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_5_MODEL_SELECTION_REPORT.md",
            "traceability_explanation": "Evaluated on VALIDATION SPLIT (9,632 GT objects in 2,567 images). At operational conf=0.25, 4,760 lesions had no detection with IoU>=0.10 or conf>=0.25. THIS IS THE SOURCE OF THE FN=4,760 FIGURE."
        },
        {
            "evaluation_run": "Stage 8.5 Validation Custom Error Taxonomy (EXP-1)",
            "checkpoint": "stage_8_ablation/exp_1_adaptive_sampling/weights/best.pt",
            "split": "validation",
            "total_images": 2567,
            "total_gt_instances": 9632,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1798,
            "false_negatives_fn": 4769,
            "localization_error": 951,
            "classification_confusion": 245,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.4034 (40.34% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_5_MODEL_SELECTION_REPORT.md",
            "traceability_explanation": "Evaluated on VALIDATION SPLIT (9,632 GT objects in 2,567 images) for EXP-1."
        },
        {
            "evaluation_run": "Stage 8.5 Validation Custom Error Taxonomy (EXP-2)",
            "checkpoint": "stage_8_ablation/exp_2_targeted_aug/weights/best.pt",
            "split": "validation",
            "total_images": 2567,
            "total_gt_instances": 9632,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1899,
            "false_negatives_fn": 4573,
            "localization_error": 979,
            "classification_confusion": 255,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.4258 (42.58% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_5_MODEL_SELECTION_REPORT.md",
            "traceability_explanation": "Evaluated on VALIDATION SPLIT (9,632 GT objects in 2,567 images) for EXP-2. FN reduced by 187 (from 4,760 down to 4,573)."
        },
        {
            "evaluation_run": "Stage 8.5 Validation Custom Error Taxonomy (EXP-3)",
            "checkpoint": "stage_8_ablation/exp_3_full_proposed/weights/best.pt",
            "split": "validation",
            "total_images": 2567,
            "total_gt_instances": 9632,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1719,
            "false_negatives_fn": 4860,
            "localization_error": 947,
            "classification_confusion": 250,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.4090 (40.90% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_5_MODEL_SELECTION_REPORT.md",
            "traceability_explanation": "Evaluated on VALIDATION SPLIT (9,632 GT objects in 2,567 images) for EXP-3."
        },
        {
            "evaluation_run": "Stage 8.6 Locked Test Set Benchmark (EXP-0)",
            "checkpoint": "stage_8_ablation/exp_0_baseline/weights/best.pt",
            "split": "test",
            "total_images": 1282,
            "total_gt_instances": 5084,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1776,
            "false_negatives_fn": 2654,
            "localization_error": 443,
            "classification_confusion": 151,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.3954 (39.54% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_FINAL_THESIS_BENCHMARK_REPORT.md",
            "traceability_explanation": "Evaluated on LOCKED TEST SPLIT (5,084 GT objects in 1,282 images). Because test split has roughly half the images of valid split, FN count is naturally lower: exactly 2,654. THIS IS THE SOURCE OF THE FN=2,654 FIGURE."
        },
        {
            "evaluation_run": "Stage 8.6 Locked Test Set Benchmark (EXP-1)",
            "checkpoint": "stage_8_ablation/exp_1_adaptive_sampling/weights/best.pt",
            "split": "test",
            "total_images": 1282,
            "total_gt_instances": 5084,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1798,
            "false_negatives_fn": 2613,
            "localization_error": 501,
            "classification_confusion": 119,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.3936 (39.36% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_FINAL_THESIS_BENCHMARK_REPORT.md",
            "traceability_explanation": "Evaluated on LOCKED TEST SPLIT (5,084 GT objects in 1,282 images) for EXP-1."
        },
        {
            "evaluation_run": "Stage 8.6 Locked Test Set Benchmark (EXP-2)",
            "checkpoint": "stage_8_ablation/exp_2_targeted_aug/weights/best.pt",
            "split": "test",
            "total_images": 1282,
            "total_gt_instances": 5084,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1899,
            "false_negatives_fn": 2499,
            "localization_error": 474,
            "classification_confusion": 142,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.3948 (39.48% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_FINAL_THESIS_BENCHMARK_REPORT.md",
            "traceability_explanation": "Evaluated on LOCKED TEST SPLIT (5,084 GT objects in 1,282 images) for EXP-2. FN reduced by 155 (from 2,654 down to 2,499)."
        },
        {
            "evaluation_run": "Stage 8.6 Locked Test Set Benchmark (EXP-3)",
            "checkpoint": "stage_8_ablation/exp_3_full_proposed/weights/best.pt",
            "split": "test",
            "total_images": 1282,
            "total_gt_instances": 5084,
            "conf_threshold": "0.25 (fixed operational)",
            "iou_matching_threshold": "0.10 (class-agnostic greedy bipartite)",
            "matching_policy": "Class-agnostic greedy bipartite matching at IoU>=0.10; TP defined as IoU>=0.50 AND correct class",
            "true_positives_tp": 1719,
            "false_negatives_fn": 2687,
            "localization_error": 479,
            "classification_confusion": 136,
            "loc_and_class_error": 0,
            "ultralytics_recall": "0.3803 (38.03% at PR-curve F1-peak)",
            "source_report": "stage_8_ablation/STAGE_8_FINAL_THESIS_BENCHMARK_REPORT.md",
            "traceability_explanation": "Evaluated on LOCKED TEST SPLIT (5,084 GT objects in 1,282 images) for EXP-3."
        },
    ]

    out_csv = Path("stage_9_1/error_metric_traceability.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

def copy_metrics_reconciled():
    src = Path("stage_9_audit/metrics_reconciled.csv")
    dst = Path("stage_9_1/metrics_reconciled.csv")
    shutil.copy2(src, dst)
    print(f"Copied metrics_reconciled.csv to {dst}")

if __name__ == "__main__":
    build_traceability_csv()
    copy_metrics_reconciled()

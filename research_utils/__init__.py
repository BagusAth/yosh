"""
Research Utils Package — YOLO11n Cassava Disease Detection
===========================================================
Shared mathematical and evaluation utilities ensuring 100% metric consistency across all pipeline stages.
"""

from .detection_metrics import (
    CLASS_NAMES,
    calculate_iou_xyxy,
    parse_yolo_labels,
    classify_object_size,
    greedy_bipartite_matching,
    calculate_object_difficulty,
    calculate_image_difficulty,
    calculate_mixture_weights,
    select_best_model,
    extract_canonical_metrics,
)

__all__ = [
    "CLASS_NAMES",
    "calculate_iou_xyxy",
    "parse_yolo_labels",
    "classify_object_size",
    "greedy_bipartite_matching",
    "calculate_object_difficulty",
    "calculate_image_difficulty",
    "calculate_mixture_weights",
    "select_best_model",
    "extract_canonical_metrics",
]

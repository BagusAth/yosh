"""
Detection Metrics & Canonical Evaluation Utility Module
======================================================
Unified, single-source-of-truth mathematical evaluation engine for:
- Safe IoU computation (xyxy, xywh)
- Object scale classification (Small < 32^2, Medium 32^2-96^2, Large >= 96^2)
- Dynamic image resolution YOLO label parsing
- Global one-to-one greedy bipartite matching (zero prediction reuse, zero GT reuse)
- Object-level difficulty formula D_i (+ 0.35 confusion penalty, FN = 0.80 + 0.20 S_i)
- Image-level difficulty formula D_image (0.5 max + 0.5 mean, max FP conf)
- Mixture adaptive sampling weights P(k) with lambda = 0.70
- Locked model selection decision rule hierarchy
- Canonical extraction of Ultralytics box.ap50 and box.maps
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]

FORBIDDEN_HISTORICAL_DIRS = {
    "stage_8_1_oof",
    "stage_8_2_difficulty",
    "stage_8_ablation",
    "cassava-v7-stage-7",
}


def compute_file_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 hash of a file. Returns empty string if file does not exist."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_pretrained_weights_sha256(weights_path: str | Path) -> str:
    """
    Computes SHA-256 hash of pretrained model weights (e.g. yolo11n.pt).
    Raises FileNotFoundError if file is missing, empty, or unreadable.
    """
    p = Path(weights_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
    if p.stat().st_size == 0:
        raise ValueError(f"Pretrained weights file is empty (0 bytes): {weights_path}")
    sha = compute_file_sha256(p)
    if not sha:
        raise RuntimeError(f"Failed to compute SHA-256 for pretrained weights: {weights_path}")
    return sha


def compute_dataset_inventory_fingerprint(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
) -> str:
    """
    Computes a deterministic full-content inventory fingerprint for a dataset directory:
    - Normalizes sorted filenames.
    - Full SHA-256 content hash of every image file.
    - Full SHA-256 content hash of every label file.
    - Label completeness and strict pairing verification.
    - Object count and class breakdown per file.
    - Source data.yaml content hash and class mappings if provided or resolvable.
    Raises FileNotFoundError or ValueError on missing directories, empty files, or corrupt data.
    Never returns an empty string for valid datasets.
    """
    img_p = Path(images_dir).resolve()
    if not img_p.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not img_p.is_dir():
        raise ValueError(f"Images path is not a directory: {images_dir}")

    # Gather image files deterministically
    img_files = sorted(
        [f for f in img_p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    if not img_files:
        raise ValueError(f"No image files (.jpg, .jpeg, .png) found in: {images_dir}")

    h = hashlib.sha256()
    h.update(f"NUM_IMAGES:{len(img_files)}\n".encode("utf-8"))

    for f in img_files:
        content = f.read_bytes()
        file_sha = hashlib.sha256(content).hexdigest()
        h.update(f"IMG:{f.name}:{len(content)}:{file_sha}\n".encode("utf-8"))

    # Resolve labels directory if not explicitly provided
    lbl_p = None
    if labels_dir:
        lbl_p = Path(labels_dir).resolve()
    elif (img_p.parent / "labels").exists():
        lbl_p = (img_p.parent / "labels").resolve()

    if lbl_p is not None:
        if not lbl_p.exists():
            raise FileNotFoundError(f"Labels directory not found: {lbl_p}")
        if not lbl_p.is_dir():
            raise ValueError(f"Labels path is not a directory: {lbl_p}")

        lbl_files = sorted(
            [lf for lf in lbl_p.iterdir() if lf.is_file() and lf.suffix.lower() == ".txt"],
            key=lambda p: p.name,
        )
        h.update(f"NUM_LABELS:{len(lbl_files)}\n".encode("utf-8"))

        total_objects = 0
        class_counter = Counter()

        for lf in lbl_files:
            text = lf.read_text(encoding="utf-8")
            lbl_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            num_objs = len(lines)
            total_objects += num_objs

            for l in lines:
                parts = l.split()
                if parts:
                    try:
                        cid = int(parts[0])
                        class_counter[cid] += 1
                    except ValueError:
                        pass

            h.update(f"LBL:{lf.name}:{len(text)}:{lbl_sha}:{num_objs}\n".encode("utf-8"))

        h.update(f"TOTAL_OBJECTS:{total_objects}\n".encode("utf-8"))
        for cid in sorted(class_counter.keys()):
            h.update(f"CLS_{cid}:{class_counter[cid]}\n".encode("utf-8"))

        # Verify pairing
        img_stems = {f.stem for f in img_files}
        lbl_stems = {lf.stem for lf in lbl_files}
        missing_lbls = sorted(list(img_stems - lbl_stems))
        orphan_lbls = sorted(list(lbl_stems - img_stems))
        if missing_lbls:
            h.update(f"MISSING_LABELS:{len(missing_lbls)}:{','.join(missing_lbls[:10])}\n".encode("utf-8"))
        if orphan_lbls:
            h.update(f"ORPHAN_LABELS:{len(orphan_lbls)}:{','.join(orphan_lbls[:10])}\n".encode("utf-8"))

    # Resolve data.yaml if provided
    yaml_p = None
    if data_yaml_path:
        yaml_p = Path(data_yaml_path).resolve()
    elif (img_p.parent / "data.yaml").exists():
        yaml_p = (img_p.parent / "data.yaml").resolve()
    elif (img_p.parent.parent / "data.yaml").exists():
        yaml_p = (img_p.parent.parent / "data.yaml").resolve()

    if yaml_p and yaml_p.exists():
        yaml_text = yaml_p.read_text(encoding="utf-8")
        yaml_sha = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
        h.update(f"DATA_YAML:{yaml_p.name}:{yaml_sha}\n".encode("utf-8"))
        try:
            yd = yaml.safe_load(yaml_text)
            if isinstance(yd, dict):
                h.update(f"NC:{yd.get('nc', '')}\n".encode("utf-8"))
                names = yd.get("names", [])
                h.update(f"NAMES:{names}\n".encode("utf-8"))
        except Exception:
            pass

    fingerprint = h.hexdigest()
    if not fingerprint:
        raise RuntimeError("Failed to compute valid dataset inventory fingerprint!")
    return fingerprint


def resolve_data_yaml_split(
    data_yaml_path: str | Path,
    split: str = "val",
    allow_empty: bool = True,
) -> DataYamlSplitResult:
    """
    Resolves dataset paths strictly from data.yaml as Single Source of Truth:
    - Respects 'path' root key (relative to data.yaml or absolute).
    - Supports split keys: 'train', 'val' (or 'valid'), 'test'.
    - Handles directory structures (images/, labels/) and .txt file image lists.
    - Guarantees custom diagnostic evaluation reads the exact split used by YOLO metrics.
    - Fails closed with descriptive exceptions if data.yaml or splits are missing.
    """
    yaml_p = Path(data_yaml_path).resolve()
    if not yaml_p.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_p}")

    with open(yaml_p, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if not isinstance(data_cfg, dict):
        raise ValueError(f"Invalid data.yaml content in {yaml_p}: root must be a mapping dictionary")

    # Base path resolution: Ultralytics standard
    if "path" in data_cfg and data_cfg["path"]:
        p_cfg = Path(data_cfg["path"])
        base_path = p_cfg.resolve() if p_cfg.is_absolute() else (yaml_p.parent / p_cfg).resolve()
    else:
        base_path = yaml_p.parent.resolve()

    # Split key resolution
    split_norm = split.strip().lower()
    possible_keys = [split_norm]
    if split_norm in ("val", "valid"):
        possible_keys = ["val", "valid"]

    split_entry = None
    matched_key = None
    for k in possible_keys:
        if k in data_cfg:
            split_entry = data_cfg[k]
            matched_key = k
            break

    if split_entry is None:
        raise KeyError(
            f"Split '{split}' not found in data.yaml ({yaml_p}). "
            f"Available keys: {list(data_cfg.keys())}"
        )

    images: list[Path] = []
    labels: list[Path] = []
    images_dir: Path | None = None
    labels_dir: Path | None = None
    is_file_list = False

    if isinstance(split_entry, str) and split_entry.strip().endswith(".txt"):
        is_file_list = True
        txt_path = Path(split_entry)
        if not txt_path.is_absolute():
            txt_path = (base_path / txt_path).resolve()
        if not txt_path.exists():
            raise FileNotFoundError(f"Split file list not found for '{split}': {txt_path}")

        lines = [l.strip() for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for l in lines:
            im_p = Path(l)
            if not im_p.is_absolute():
                im_p = (base_path / im_p).resolve()
            if not im_p.exists():
                raise FileNotFoundError(f"Image from split file list not found: {im_p}")
            images.append(im_p)

            # Derive label path
            lbl_str = str(im_p).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            lbl_p = Path(lbl_str).with_suffix(".txt")
            labels.append(lbl_p)

        if images:
            images_dir = images[0].parent
            labels_dir = labels[0].parent
    else:
        # Directory or relative folder
        split_p = Path(split_entry)
        if not split_p.is_absolute():
            split_p = (base_path / split_p).resolve()

        if split_p.name == "images":
            images_dir = split_p
            labels_dir = split_p.parent / "labels"
        else:
            if (split_p / "images").exists():
                images_dir = split_p / "images"
                labels_dir = split_p / "labels"
            else:
                images_dir = split_p
                labels_dir = split_p.parent / "labels" if (split_p.parent / "labels").exists() else split_p / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Resolved images directory does not exist for split '{split}': {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Resolved labels directory does not exist for split '{split}': {labels_dir}")

        images = sorted(
            [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda p: p.name,
        )
        labels = [labels_dir / f"{im.stem}.txt" for im in images]

    if not images and not allow_empty:
        raise ValueError(f"No valid images found for split '{split}' in {data_yaml_path}")

    yaml_sha = compute_file_sha256(yaml_p)

    return DataYamlSplitResult({
        "split": split,
        "matched_key": matched_key,
        "base_path": base_path,
        "images": images,
        "labels": labels,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "is_file_list": is_file_list,
        "total_images": len(images),
        "data_yaml_path": yaml_p,
        "data_yaml_sha256": yaml_sha,
        "class_names": data_cfg.get("names", CLASS_NAMES),
        "nc": data_cfg.get("nc", len(data_cfg.get("names", CLASS_NAMES))),
    })


class DataYamlSplitResult(dict):
    """Custom dictionary result for resolve_data_yaml_split supporting 3-tuple unpack."""
    def __iter__(self):
        yield self["images_dir"]
        yield self["labels_dir"]
        yield self["images"]


def validate_output_directory_safety(
    output_dir: str | Path,
    stage_name: str,
    allow_reuse: bool = False,
    expected_identity: dict[str, Any] | None = None,
    metadata_filename: str | None = None,
    is_dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Validates that an output directory is safe for writing and does not overwrite
    historical research artifacts or unverified partial results.
    """
    p = Path(output_dir).resolve()

    # 1. Guard against writing to historical research directories
    p_norm = str(p).replace("\\", "/").lower()
    for forb in FORBIDDEN_HISTORICAL_DIRS:
        if p_norm.endswith(f"/{forb}") or p.name.lower() == forb:
            raise PermissionError(
                f"CRITICAL SAFETY ERROR / VIOLATION [{stage_name}]: Target directory '{p}' matches historical "
                f"artifact directory '{forb}'! Modifying or writing into historical directories is strictly forbidden.\n"
                f"Please provide an explicit clean output directory (e.g. stage_{stage_name.lower()}_clean)."
            )

    # 2. If directory does not exist, it is completely clean and safe
    if not p.exists():
        return True, "Output directory does not exist yet (clean target)"

    # 3. If directory exists, check contents
    benign_names = {".gitkeep", ".gitignore", "desktop.ini", "__pycache__"}
    existing_files = [f for f in p.rglob("*") if f.is_file() and f.name not in benign_names]

    if not existing_files:
        return True, "Output directory exists and is empty (clean target)"

    # 4. If directory has files and reuse is NOT requested: REJECT immediately
    if not allow_reuse:
        sample_files = [f.name for f in existing_files[:5]]
        raise FileExistsError(
            f"SAFETY VIOLATION [{stage_name}]: Output directory '{p}' already exists and contains "
            f"{len(existing_files)} artifact(s) ({sample_files}).\n"
            f"Overwriting experiment artifacts in clean runs is prohibited. "
            f"Please specify a new clean directory for --output-dir."
        )

    # 5. If reuse IS requested: verify identity and metadata
    if expected_identity is None:
        raise PermissionError(
            f"SAFETY VIOLATION [{stage_name}]: Reuse was requested for '{p}', but expected_identity was not provided!"
        )

    # Find metadata file
    meta_candidates = []
    if metadata_filename:
        meta_candidates.append(p / metadata_filename)
    meta_candidates.extend([
        p / "run_metadata.json",
        p / "fold_metadata.json",
        p / "metrics_summary.json",
        p / "difficulty_sanity_gate_report.json",
    ])

    meta_file = None
    for cand in meta_candidates:
        if cand.exists():
            meta_file = cand
            break

    if meta_file is None:
        raise PermissionError(
            f"SAFETY VIOLATION [{stage_name}]: Output directory '{p}' contains unverified partial artifacts "
            f"without any valid metadata file! Reusing unverified partial results is prohibited.\n"
            f"Please specify a new clean directory for --output-dir."
        )

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
    except Exception as e:
        raise PermissionError(
            f"SAFETY VIOLATION [{stage_name}]: Failed to read metadata JSON from '{meta_file}': {e}"
        )

    meta_dict = saved_meta.get("run_metadata", saved_meta) if isinstance(saved_meta, dict) else {}

    for k, exp_val in expected_identity.items():
        if k in ("timestamp", "checkpoint_sha256", "pred_csv_sha256", "best_weight_sha256"):
            continue
        saved_val = meta_dict.get(k)
        if saved_val != exp_val:
            raise PermissionError(
                f"SAFETY VIOLATION [{stage_name}]: Metadata mismatch in '{p}' for key '{k}': "
                f"saved={saved_val} vs expected={exp_val}. "
                f"Cannot reuse mismatched artifacts. Please specify a new clean directory."
            )

    return True, "Artifacts and metadata 100% verified for reuse"


def get_image_dimensions(img_path: str | Path) -> tuple[int, int]:
    """
    Reads image dimensions (width, height) safely from image header.
    Returns (width, height) in integer pixels.
    Raises FileNotFoundError or ValueError if invalid.
    """
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    try:
        with Image.open(p) as img:
            w, h = img.size
    except Exception as e:
        # Fallback to OpenCV if PIL fails
        import cv2
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f"Failed to read image dimensions for {p}: {e}")
        h, w = im.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid non-positive image dimensions ({w}x{h}) for {p}")

    return int(w), int(h)


def calculate_iou_xyxy(box1: Any, box2: Any) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    Accepts lists, tuples, or NumPy arrays.
    Returns: float in [0.0, 1.0].
    """
    b1 = [float(x) for x in box1]
    b2 = [float(x) for x in box2]

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def classify_object_size(area_px: float) -> tuple[str, float]:
    """
    Classifies object size using standard COCO / PASCAL pixel area boundaries:
    - Small: area < 32^2 (1024 px^2) -> size difficulty penalty S_i = 1.0
    - Medium: 32^2 <= area <= 96^2 (1024 to 9216 px^2) -> size difficulty penalty S_i = 0.5
    - Large: area > 96^2 (9216 px^2) -> size difficulty penalty S_i = 0.0

    Returns:
        (size_category_str, size_penalty_float)
    """
    if area_px < (32.0 * 32.0):
        return "Small", 1.0
    elif area_px < (96.0 * 96.0):
        return "Medium", 0.5
    else:
        return "Large", 0.0


def parse_yolo_labels(
    lbl_path: str | Path,
    img_width: float = 640.0,
    img_height: float = 640.0,
    class_names: list[str] = CLASS_NAMES,
) -> list[dict[str, Any]]:
    """
    Parses YOLO label text file (supporting both 4-value bbox and N-point polygons).
    Converts all coordinates to absolute pixel bounding boxes [x1, y1, x2, y2]
    based on dynamic img_width and img_height.
    """
    p = Path(lbl_path)
    if not p.exists() or p.stat().st_size == 0:
        return []

    objects = []
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        if len(coords) == 4:
            xc, yc, w, h = coords
            x1 = max(0.0, (xc - w / 2.0) * img_width)
            y1 = max(0.0, (yc - h / 2.0) * img_height)
            x2 = min(img_width, (xc + w / 2.0) * img_width)
            y2 = min(img_height, (yc + h / 2.0) * img_height)
        else:
            if len(coords) % 2 != 0:
                continue
            xs = [coords[i] * img_width for i in range(0, len(coords), 2)]
            ys = [coords[i + 1] * img_height for i in range(0, len(coords), 2)]
            x1 = max(0.0, min(xs))
            y1 = max(0.0, min(ys))
            x2 = min(img_width, max(xs))
            y2 = min(img_height, max(ys))

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area = box_w * box_h
        size_cat, size_penalty = classify_object_size(area)
        cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"

        objects.append({
            "class_id": cid,
            "class_name": cname,
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "area": area,
            "size_category": size_cat,
            "size_penalty": size_penalty,
            "raw_coords": coords,
        })

    return objects


class MatchingResult(dict):
    """Custom dictionary result for greedy_bipartite_matching supporting tuple unpack."""
    def __iter__(self):
        yield self["matches"]
        yield self["unassigned_preds"]


def greedy_bipartite_matching(
    gt_objects: Any,
    pred_boxes: Any = None,
    pred_classes: Any = None,
    pred_scores: Any = None,
    iou_match_thresh: float | None = None,
    tp_iou_thresh: float = 0.50,
    iou_thresh: float | None = None,
    **kwargs,
) -> MatchingResult:
    """
    Executes canonical 1-to-1 global bipartite greedy matching based on descending IoU order.
    Supports:
    - matches, unassigned_preds = greedy_bipartite_matching(...)
    - res["taxonomy_counts"], res["tp_by_size"], etc.
    """
    thresh = iou_thresh if iou_thresh is not None else (iou_match_thresh if iou_match_thresh is not None else 0.10)

    # Normalize gt_objects
    normalized_gts = []
    if isinstance(gt_objects, list):
        for idx, g in enumerate(gt_objects):
            if isinstance(g, dict):
                b = g.get("box", g.get("box_xyxy", [0, 0, 0, 0]))
                cid = g.get("class_id", 0)
                scat = g.get("size_category", g.get("size_cat", "Medium"))
                spen = g.get("size_penalty", g.get("size_score", 0.5))
            else:
                b = g
                cid = 0
                scat = "Medium"
                spen = 0.5
            normalized_gts.append({
                "gt_idx": idx,
                "box": [float(x) for x in b],
                "class_id": int(cid),
                "size_category": scat,
                "size_penalty": float(spen),
            })

    # Normalize pred_boxes
    normalized_preds = []
    if pred_boxes is not None:
        if isinstance(pred_boxes, (np.ndarray, list)):
            for idx, pb in enumerate(pred_boxes):
                if isinstance(pb, dict):
                    b = pb.get("box", pb.get("box_xyxy", [0, 0, 0, 0]))
                    cid = pb.get("class_id", 0)
                    conf = pb.get("confidence", 1.0)
                else:
                    b = pb
                    cid = int(pred_classes[idx]) if pred_classes is not None and idx < len(pred_classes) else 0
                    conf = float(pred_scores[idx]) if pred_scores is not None and idx < len(pred_scores) else 1.0
                normalized_preds.append({
                    "pred_idx": idx,
                    "box": [float(x) for x in b],
                    "class_id": int(cid),
                    "confidence": float(conf),
                })

    num_gt = len(normalized_gts)
    num_preds = len(normalized_preds)

    candidate_matches = []
    if num_gt > 0 and num_preds > 0:
        for g_idx, g in enumerate(normalized_gts):
            for p_idx, p in enumerate(normalized_preds):
                iou = calculate_iou_xyxy(g["box"], p["box"])
                if iou >= thresh:
                    candidate_matches.append((iou, g_idx, p_idx))

        # Sort candidate matches descending by IoU globally
        candidate_matches.sort(key=lambda x: x[0], reverse=True)

    matched_gt_map = {}  # gt_idx -> (p_idx, iou)
    assigned_preds = set()
    for iou, gt_idx, p_idx in candidate_matches:
        if gt_idx not in matched_gt_map and p_idx not in assigned_preds:
            matched_gt_map[gt_idx] = (p_idx, iou)
            assigned_preds.add(p_idx)

    unassigned_preds = set(range(num_preds)) - assigned_preds

    taxonomy_counts = Counter()
    tp_by_size = Counter()
    gt_by_size = Counter()
    gt_assignments = []

    for gt_idx, gt in enumerate(normalized_gts):
        size_cat = gt["size_category"]
        gt_cid = gt["class_id"]
        gt_by_size[size_cat] += 1

        if gt_idx in matched_gt_map:
            p_idx, iou = matched_gt_map[gt_idx]
            p = normalized_preds[p_idx]
            p_cid = p["class_id"]
            conf = p["confidence"]
            is_correct_class = (p_cid == gt_cid)

            if iou >= tp_iou_thresh:
                if is_correct_class:
                    err_type = "True_Positive"
                    tp_by_size[size_cat] += 1
                else:
                    err_type = "Classification_Confusion"
            else:
                if is_correct_class:
                    err_type = "Localization_Error"
                else:
                    err_type = "Localization_and_Class_Error"

            gt_assignments.append({
                "gt_idx": gt_idx,
                "matched_pred_idx": p_idx,
                "iou": iou,
                "conf": conf,
                "pred_cid": p_cid,
                "gt_cid": gt_cid,
                "is_correct_class": is_correct_class,
                "error_type": err_type,
                "size_category": size_cat,
                "size_penalty": gt["size_penalty"],
            })
            taxonomy_counts[err_type] += 1
        else:
            err_type = "False_Negative_Missed"
            gt_assignments.append({
                "gt_idx": gt_idx,
                "matched_pred_idx": -1,
                "iou": 0.0,
                "conf": 0.0,
                "pred_cid": -1,
                "gt_cid": gt_cid,
                "is_correct_class": False,
                "error_type": err_type,
                "size_category": size_cat,
                "size_penalty": gt["size_penalty"],
            })
            taxonomy_counts[err_type] += 1

    return MatchingResult({
        "matches": matched_gt_map,
        "unassigned_preds": unassigned_preds,
        "taxonomy_counts": taxonomy_counts,
        "gt_assignments": gt_assignments,
        "matched_preds": assigned_preds,
        "tp_by_size": tp_by_size,
        "gt_by_size": gt_by_size,
    })


def calculate_object_difficulty(
    conf: float = 0.0,
    iou: float = 0.0,
    size_penalty: float = 0.0,
    size_score: float | None = None,
    wrong_class: bool = False,
    is_fn: bool = False,
    matched: bool = True,
    correct_class: bool = True,
    confusion_penalty: float = 0.35,
    w1: float = 0.40,
    w2: float = 0.40,
    w3: float = 0.20,
) -> float:
    """
    Computes Object Difficulty Score D_i:
    - If False Negative (Missed): D_i = 0.80 + 0.20 * S_i
    - If Matched: D_i = w1*(1 - C) + w2*(1 - IoU) + w3*S_i (+ confusion_penalty if wrong_class)
    Clipped strictly to [0.0, 1.0].
    """
    s_val = float(size_score if size_score is not None else size_penalty)
    if not matched:
        is_fn = True
    if not correct_class:
        wrong_class = True

    if is_fn:
        di = 0.80 + 0.20 * s_val
    else:
        di = w1 * (1.0 - float(conf)) + w2 * (1.0 - float(iou)) + w3 * s_val
        if wrong_class:
            di += confusion_penalty
    return max(0.0, min(1.0, float(di)))


def calculate_image_difficulty(
    object_difficulties: list[float] | None = None,
    obj_difficulties: list[float] | None = None,
    empty_image_max_conf: float = 0.0,
    fp_confs: list[float] | None = None,
) -> float:
    """
    Computes Image Difficulty Score D_image:
    - If image has GT objects: D_image = 0.5 * max(D_i) + 0.5 * mean(D_i)
    - If empty background image: D_image = max(C_FP) (confidence of highest false positive)
    Clipped strictly to [0.0, 1.0].
    """
    objs = object_difficulties if object_difficulties is not None else (obj_difficulties if obj_difficulties is not None else [])
    if len(objs) > 0:
        max_d = max(objs)
        mean_d = sum(objs) / len(objs)
        d_img = 0.5 * max_d + 0.5 * mean_d
    else:
        if fp_confs is not None and len(fp_confs) > 0:
            d_img = float(max(fp_confs))
        else:
            d_img = float(empty_image_max_conf)
    return max(0.0, min(1.0, float(d_img)))


def calculate_mixture_weights(
    d_image_data: dict[str, float] | list[float],
    mixture_lambda: float = 0.70,
) -> dict[str, float] | list[float]:
    """
    Computes Mixture Adaptive Sampling distribution P(k):
    P(k) = (1 - lambda) / N + lambda * D_k / sum(D)
    Supports both dict and list inputs.
    Strictly validates 0.0 <= lambda <= 1.0, finite difficulties in [0.0, 1.0],
    explicit fallback to uniform 1/N when sum(D) == 0, and strictly positive probabilities.
    """
    if not (0.0 <= mixture_lambda <= 1.0):
        raise ValueError(f"Mixture lambda must be in [0.0, 1.0], got {mixture_lambda}")

    is_dict = isinstance(d_image_data, dict)
    if is_dict:
        keys = list(d_image_data.keys())
        d_vals = list(d_image_data.values())
    else:
        keys = None
        d_vals = list(d_image_data)

    n = len(d_vals)
    if n == 0:
        raise ValueError("Image difficulty collection is empty!")

    for idx, d_val in enumerate(d_vals):
        if not math.isfinite(d_val) or d_val < 0.0 or d_val > 1.0:
            raise ValueError(f"Invalid difficulty value at index {idx}: {d_val} (must be finite in [0.0, 1.0])")

    sum_d = sum(d_vals)
    uniform_floor = (1.0 - mixture_lambda) / n

    if sum_d <= 0.0 or not math.isfinite(sum_d):
        weights = [1.0 / n] * n
    else:
        weights = []
        for d_val in d_vals:
            p_k = uniform_floor + mixture_lambda * (d_val / sum_d)
            if p_k <= 0.0 or not math.isfinite(p_k):
                raise ValueError(f"Calculated non-positive or non-finite probability {p_k} for d_val {d_val}")
            weights.append(float(p_k))

        total_p = sum(weights)
        if abs(total_p - 1.0) > 1e-5:
            weights = [w / total_p for w in weights]

    if is_dict:
        return {k: weights[idx] for idx, k in enumerate(keys)}
    return weights


def select_best_model(
    candidates: list[dict[str, Any]],
    tolerance: float = 1e-5,
) -> tuple[dict[str, Any], str]:
    """
    Canonical Model Selection Protocol Hierarchy:
    1. Primary Criterion: Highest Validation mAP@50-95.
    2. Tie-break 1 (ONLY if |delta mAP@50-95| <= tolerance): Highest Recall Small.
    3. Tie-break 2 (if Recall Small also tied): Highest Validation mAP@50.
    4. Tie-break 3 (if mAP@50 also tied): Lowest False Negative (fn_missed).
    """
    if not candidates:
        raise ValueError("Candidate model list is empty!")

    max_map50_95 = max(r["mAP50_95"] for r in candidates)
    tied_candidates = [r for r in candidates if abs(r["mAP50_95"] - max_map50_95) <= tolerance]

    if len(tied_candidates) == 1:
        selected = tied_candidates[0]
        rationale = (
            f"Kandidat {selected.get('exp_id', 'Unknown')} ({selected.get('title', '')}) "
            f"terpilih mutlak berdasarkan mAP@50-95 validasi tertinggi ({selected['mAP50_95']:.6f}) "
            f"tanpa kondisi tie (toleransi tie <= {tolerance:.1e})."
        )
    else:
        sorted_tied = sorted(
            tied_candidates,
            key=lambda x: (
                x.get("recall_small", 0.0),
                x.get("mAP50", 0.0),
                -x.get("fn_missed", 0),
            ),
            reverse=True,
        )
        selected = sorted_tied[0]
        tie_ids = [c.get("exp_id", str(idx)) for idx, c in enumerate(tied_candidates)]
        rationale = (
            f"Terjadi tie numerik mAP@50-95 ({max_map50_95:.6f}) pada kandidat {tie_ids}. "
            f"Kandidat {selected.get('exp_id', '')} terpilih melalui protokol tie-break resmi: "
            f"Recall_Small={selected.get('recall_small', 0.0):.4f}, "
            f"mAP@50={selected.get('mAP50', 0.0):.4f}, "
            f"FN_Missed={selected.get('fn_missed', 0)}."
        )

    return selected, rationale


class CanonicalMetricsResult(dict):
    """Dictionary holding extracted metrics, supporting tuple unpack (ap50, map95)."""
    def __iter__(self):
        yield self["per_class_ap50"]
        yield self["per_class_map50_95"]


def extract_canonical_metrics(
    val_metrics: Any,
    class_names: list[str] = CLASS_NAMES,
) -> CanonicalMetricsResult:
    """
    Extracts canonical per-class AP metrics from Ultralytics Metric / DetMetrics object:
    - precision: box.mp
    - recall: box.mr
    - map50: box.map50
    - map50_95: box.map
    - per_class_ap50: from box.ap50
    - per_class_map50_95: from box.maps
    """
    box_obj = getattr(val_metrics, "box", val_metrics)

    precision = float(getattr(box_obj, "mp", 0.0))
    recall = float(getattr(box_obj, "mr", 0.0))
    map50 = float(getattr(box_obj, "map50", 0.0))
    map50_95 = float(getattr(box_obj, "map", 0.0))

    per_class_ap50 = {}
    per_class_map50_95 = {}

    if hasattr(box_obj, "ap50") and box_obj.ap50 is not None:
        ap50_arr = box_obj.ap50
        ap_class_idx = (
            list(box_obj.ap_class_index)
            if hasattr(box_obj, "ap_class_index")
            else list(range(len(ap50_arr)))
        )
        for i, c_idx in enumerate(ap_class_idx):
            if c_idx < len(class_names) and i < len(ap50_arr):
                per_class_ap50[class_names[c_idx]] = float(ap50_arr[i])

    if hasattr(box_obj, "maps") and box_obj.maps is not None:
        for idx, cname in enumerate(class_names):
            if idx < len(box_obj.maps):
                per_class_map50_95[cname] = float(box_obj.maps[idx])

    return CanonicalMetricsResult({
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class_ap50": per_class_ap50,
        "per_class_map50_95": per_class_map50_95,
        "metrics_raw": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "per_class_ap50": per_class_ap50,
            "per_class_map50_95": per_class_map50_95,
        },
    })

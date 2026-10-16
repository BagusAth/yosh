from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
import yaml
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Ganti dengan lokasi dataset Cassava Anda
DATASET_DIR = Path("cassava-dataset")

# Folder output hasil audit
OUTPUT_DIR = Path("cassava_audit_report")

# Random seed agar sampling konsisten
RANDOM_SEED = 42

# IoU threshold untuk pemeriksaan overlap antar-GT
OVERLAP_IOU_THRESHOLD = 0.5

# Batas kategori ukuran object
# Berdasarkan relative bounding-box area
SIZE_BINS = {
    "very_small": (0.00, 0.01),
    "small": (0.01, 0.05),
    "medium": (0.05, 0.20),
    "large": (0.20, 0.50),
    "very_large": (0.50, 1.01),
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# UTILITY
# ============================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_yaml(dataset_dir):
    candidates = [
        dataset_dir / "data.yaml",
        dataset_dir / "data.yml",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(dataset_dir.rglob("data.yaml"))
    if matches:
        return matches[0]

    matches = list(dataset_dir.rglob("data.yml"))
    if matches:
        return matches[0]

    return None


def load_dataset_config(dataset_dir):
    yaml_path = find_data_yaml(dataset_dir)

    if yaml_path is None:
        print("[WARNING] data.yaml tidak ditemukan.")
        return {
            "yaml_path": None,
            "names": {},
        }

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})

    if isinstance(names, list):
        names = {
            i: name
            for i, name in enumerate(names)
        }

    elif isinstance(names, dict):
        names = {
            int(k): v
            for k, v in names.items()
        }

    else:
        names = {}

    return {
        "yaml_path": str(yaml_path),
        "names": names,
        "config": config,
    }


def find_split_dirs(dataset_dir):
    possible_names = {
        "train": ["train"],
        "valid": ["valid", "val", "validation"],
        "test": ["test"],
    }

    result = {}

    for split, names in possible_names.items():
        for name in names:
            candidate = dataset_dir / name

            if candidate.exists() and candidate.is_dir():
                result[split] = candidate
                break

    return result


def find_images(directory):
    if directory is None or not directory.exists():
        return []

    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_label_file(image_path):
    """
    Mendukung struktur:
        images/train/image.jpg
        labels/train/image.txt

    maupun:
        train/image.jpg
        train/labels/image.txt
    """

    stem = image_path.stem

    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / f"{stem}.txt",
        image_path.parent.parent / "labels" / image_path.parent.name / f"{stem}.txt",
        image_path.parent.parent / "labels" / f"{stem}.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: cari file dengan nama yang sama
    for parent in [
        image_path.parent,
        image_path.parent.parent,
        image_path.parent.parent.parent,
    ]:
        if parent.exists():
            matches = list(parent.rglob(f"{stem}.txt"))

            if matches:
                return matches[0]

    return None


def calculate_iou(box_a, box_b):
    """
    Box format:
    x1, y1, x2, y2
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def yolo_to_xyxy(x_center, y_center, width, height):
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]


def classify_size(area):
    for category, (low, high) in SIZE_BINS.items():
        if low <= area < high:
            return category

    return "unknown"


def file_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# DATASET SCANNING
# ============================================================

def scan_dataset():
    config_info = load_dataset_config(DATASET_DIR)
    names = config_info["names"]

    split_dirs = find_split_dirs(DATASET_DIR)

    if not split_dirs:
        raise RuntimeError(
            "Folder train/valid/test tidak ditemukan. "
            "Pastikan DATASET_DIR mengarah ke root dataset YOLO."
        )

    print("\n=== DATASET CONFIGURATION ===")

    if config_info["yaml_path"]:
        print(f"data.yaml : {config_info['yaml_path']}")
    else:
        print("data.yaml : tidak ditemukan")

    print(f"classes   : {names}")

    all_image_records = []
    all_object_records = []
    invalid_records = []
    image_hash_records = []

    for split, split_dir in split_dirs.items():

        print(f"\nScanning split: {split}")
        print(f"Folder: {split_dir}")

        images = find_images(split_dir)

        print(f"Images ditemukan: {len(images)}")

        for image_path in images:

            relative_path = image_path.relative_to(DATASET_DIR)

            # ------------------------------------------------
            # IMAGE INFO
            # ------------------------------------------------

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode

            except Exception as e:

                invalid_records.append({
                    "split": split,
                    "image": str(relative_path),
                    "problem": "image_read_error",
                    "details": str(e),
                })

                continue

            label_path = find_label_file(image_path)

            objects = []
            image_invalid_count = 0

            # ------------------------------------------------
            # LABEL INFO
            # ------------------------------------------------

            if label_path is not None:

                try:
                    with open(label_path, "r", encoding="utf-8") as f:
                        lines = [
                            line.strip()
                            for line in f
                            if line.strip()
                        ]

                except Exception as e:

                    lines = []

                    invalid_records.append({
                        "split": split,
                        "image": str(relative_path),
                        "problem": "label_read_error",
                        "details": str(e),
                    })

                for line_number, line in enumerate(lines, start=1):

                    parts = line.split()

                    if len(parts) != 5:

                        invalid_records.append({
                            "split": split,
                            "image": str(relative_path),
                            "problem": "invalid_label_format",
                            "line": line_number,
                            "details": line,
                        })

                        image_invalid_count += 1
                        continue

                    class_id = parts[0]

                    try:
                        class_id = int(class_id)

                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])

                    except ValueError:

                        invalid_records.append({
                            "split": split,
                            "image": str(relative_path),
                            "problem": "non_numeric_label",
                            "line": line_number,
                            "details": line,
                        })

                        image_invalid_count += 1
                        continue

                    # ----------------------------------------
                    # RANGE VALIDATION
                    # ----------------------------------------

                    values = [
                        x_center,
                        y_center,
                        box_width,
                        box_height,
                    ]

                    if any(not math.isfinite(v) for v in values):

                        invalid_records.append({
                            "split": split,
                            "image": str(relative_path),
                            "problem": "non_finite_value",
                            "line": line_number,
                            "details": line,
                        })

                        image_invalid_count += 1
                        continue

                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 < box_width <= 1
                        and 0 < box_height <= 1
                    ):

                        invalid_records.append({
                            "split": split,
                            "image": str(relative_path),
                            "problem": "value_out_of_range",
                            "line": line_number,
                            "details": line,
                        })

                        image_invalid_count += 1

                    # ----------------------------------------
                    # CLASS VALIDATION
                    # ----------------------------------------

                    if names and class_id not in names:

                        invalid_records.append({
                            "split": split,
                            "image": str(relative_path),
                            "problem": "unknown_class_id",
                            "line": line_number,
                            "details": class_id,
                        })

                        image_invalid_count += 1

                    # ----------------------------------------
                    # BOUNDING BOX
                    # ----------------------------------------

                    area = box_width * box_height

                    aspect_ratio = (
                        box_width / box_height
                        if box_height > 0
                        else None
                    )

                    size_category = classify_size(area)

                    xyxy = yolo_to_xyxy(
                        x_center,
                        y_center,
                        box_width,
                        box_height,
                    )

                    object_record = {
                        "split": split,
                        "image": str(relative_path),
                        "class_id": class_id,
                        "class_name": names.get(
                            class_id,
                            f"class_{class_id}"
                        ),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": box_width,
                        "height": box_height,
                        "area_ratio": area,
                        "aspect_ratio": aspect_ratio,
                        "size_category": size_category,
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    }

                    objects.append(object_record)
                    all_object_records.append(object_record)

            # ------------------------------------------------
            # IMAGE STATISTICS
            # ------------------------------------------------

            areas = [
                obj["area_ratio"]
                for obj in objects
            ]

            image_record = {
                "split": split,
                "image": str(relative_path),
                "width": width,
                "height": height,
                "mode": mode,
                "objects": len(objects),
                "has_annotation": label_path is not None,
                "invalid_annotations": image_invalid_count,
                "mean_bbox_area": (
                    float(np.mean(areas))
                    if areas
                    else 0
                ),
                "min_bbox_area": (
                    float(np.min(areas))
                    if areas
                    else 0
                ),
                "max_bbox_area": (
                    float(np.max(areas))
                    if areas
                    else 0
                ),
                "multi_object": len(objects) > 1,
            }

            all_image_records.append(image_record)

            # ------------------------------------------------
            # HASH
            # ------------------------------------------------

            try:
                image_hash = file_md5(image_path)

                image_hash_records.append({
                    "split": split,
                    "image": str(relative_path),
                    "md5": image_hash,
                })

            except Exception:
                pass

    return (
        config_info,
        all_image_records,
        all_object_records,
        invalid_records,
        image_hash_records,
    )


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

def analyze_overlap(image_records, object_records):

    objects_by_image = defaultdict(list)

    for obj in object_records:
        objects_by_image[obj["image"]].append(obj)

    records = []

    for image, objects in objects_by_image.items():

        if len(objects) < 2:
            continue

        for i in range(len(objects)):

            for j in range(i + 1, len(objects)):

                obj_a = objects[i]
                obj_b = objects[j]

                iou = calculate_iou(
                    [
                        obj_a["x1"],
                        obj_a["y1"],
                        obj_a["x2"],
                        obj_a["y2"],
                    ],
                    [
                        obj_b["x1"],
                        obj_b["y1"],
                        obj_b["x2"],
                        obj_b["y2"],
                    ],
                )

                records.append({
                    "image": image,
                    "class_a": obj_a["class_name"],
                    "class_b": obj_b["class_name"],
                    "iou": iou,
                    "significant_overlap": (
                        iou >= OVERLAP_IOU_THRESHOLD
                    ),
                })

    return records


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(hash_records):

    hash_to_images = defaultdict(list)

    for record in hash_records:
        hash_to_images[record["md5"]].append(
            record["image"]
        )

    duplicate_records = []

    for md5, images in hash_to_images.items():

        if len(images) > 1:

            for image in images:

                duplicate_records.append({
                    "md5": md5,
                    "image": image,
                    "duplicate_count": len(images),
                    "duplicate_group": " | ".join(images),
                })

    return duplicate_records


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    config_info,
    image_records,
    object_records,
    invalid_records,
    overlap_records,
    duplicate_records,
):

    image_df = pd.DataFrame(image_records)
    object_df = pd.DataFrame(object_records)

    summary = {
        "dataset_dir": str(DATASET_DIR),
        "data_yaml": config_info.get("yaml_path"),
        "classes": config_info.get("names", {}),
        "total_images": len(image_records),
        "total_objects": len(object_records),
        "invalid_annotations": len(invalid_records),
        "images_without_annotations": int(
            sum(
                not r["has_annotation"]
                for r in image_records
            )
        ),
        "multi_object_images": int(
            sum(
                r["multi_object"]
                for r in image_records
            )
        ),
        "overlap_pairs": len(overlap_records),
        "significant_overlap_pairs": int(
            sum(
                r["significant_overlap"]
                for r in overlap_records
            )
        ),
        "exact_duplicate_images": len(
            duplicate_records
        ),
    }

    if len(image_df) > 0:

        summary["images_by_split"] = (
            image_df["split"]
            .value_counts()
            .to_dict()
        )

        summary["objects_per_image"] = {
            "mean": float(
                image_df["objects"].mean()
            ),
            "median": float(
                image_df["objects"].median()
            ),
            "max": int(
                image_df["objects"].max()
            ),
        }

    if len(object_df) > 0:

        summary["objects_by_class"] = (
            object_df["class_name"]
            .value_counts()
            .to_dict()
        )

        summary["objects_by_size"] = (
            object_df["size_category"]
            .value_counts()
            .to_dict()
        )

        summary["bbox_area_ratio"] = {
            "mean": float(
                object_df["area_ratio"].mean()
            ),
            "median": float(
                object_df["area_ratio"].median()
            ),
            "min": float(
                object_df["area_ratio"].min()
            ),
            "max": float(
                object_df["area_ratio"].max()
            ),
        }

    return summary


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    config_info,
    image_records,
    object_records,
    invalid_records,
    overlap_records,
    duplicate_records,
):

    image_df = pd.DataFrame(image_records)
    object_df = pd.DataFrame(object_records)
    invalid_df = pd.DataFrame(invalid_records)
    overlap_df = pd.DataFrame(overlap_records)
    duplicate_df = pd.DataFrame(duplicate_records)

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    if not object_df.empty:

        class_distribution = (
            object_df
            .groupby(["split", "class_id", "class_name"])
            .size()
            .reset_index(name="object_count")
        )

        image_class_distribution = (
            object_df
            .groupby(
                ["split", "class_id", "class_name"]
            )["image"]
            .nunique()
            .reset_index(
                name="image_count"
            )
        )

        class_distribution = class_distribution.merge(
            image_class_distribution,
            on=[
                "split",
                "class_id",
                "class_name",
            ],
            how="left",
        )

    else:

        class_distribution = pd.DataFrame()

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    image_df.to_csv(
        OUTPUT_DIR / "image_statistics.csv",
        index=False,
    )

    object_df.to_csv(
        OUTPUT_DIR / "bbox_statistics.csv",
        index=False,
    )

    class_distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False,
    )

    invalid_df.to_csv(
        OUTPUT_DIR / "invalid_annotations.csv",
        index=False,
    )

    overlap_df.to_csv(
        OUTPUT_DIR / "overlap_statistics.csv",
        index=False,
    )

    duplicate_df.to_csv(
        OUTPUT_DIR / "duplicate_candidates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = create_summary(
        config_info,
        image_records,
        object_records,
        invalid_records,
        overlap_records,
        duplicate_records,
    )

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return summary


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_visualizations(
    image_records,
    object_records,
):

    image_df = pd.DataFrame(image_records)
    object_df = pd.DataFrame(object_records)

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    if not object_df.empty:

        class_counts = (
            object_df["class_name"]
            .value_counts()
            .sort_values()
        )

        plt.figure(figsize=(10, 6))

        class_counts.plot(
            kind="barh"
        )

        plt.title(
            "Cassava Object Distribution by Class"
        )

        plt.xlabel("Number of Objects")
        plt.ylabel("Class")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "01_class_distribution.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # OBJECTS PER IMAGE
    # --------------------------------------------------------

    if not image_df.empty:

        plt.figure(figsize=(10, 6))

        image_df["objects"].hist(
            bins=20
        )

        plt.title(
            "Objects per Image"
        )

        plt.xlabel(
            "Number of Objects"
        )

        plt.ylabel(
            "Number of Images"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "02_objects_per_image.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # BBOX AREA
    # --------------------------------------------------------

    if not object_df.empty:

        plt.figure(figsize=(10, 6))

        object_df["area_ratio"].hist(
            bins=40
        )

        plt.title(
            "Relative Bounding Box Area"
        )

        plt.xlabel(
            "Bounding Box Area / Image Area"
        )

        plt.ylabel(
            "Number of Objects"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "03_bbox_area_distribution.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # SIZE CATEGORY
    # --------------------------------------------------------

    if not object_df.empty:

        size_order = [
            "very_small",
            "small",
            "medium",
            "large",
            "very_large",
        ]

        size_counts = (
            object_df["size_category"]
            .value_counts()
            .reindex(
                size_order,
                fill_value=0,
            )
        )

        plt.figure(figsize=(10, 6))

        size_counts.plot(
            kind="bar"
        )

        plt.title(
            "Bounding Box Size Categories"
        )

        plt.xlabel(
            "Size Category"
        )

        plt.ylabel(
            "Number of Objects"
        )

        plt.xticks(
            rotation=0
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "04_bbox_size_category.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # ASPECT RATIO
    # --------------------------------------------------------

    if not object_df.empty:

        plt.figure(figsize=(10, 6))

        object_df["aspect_ratio"].clip(
            upper=10
        ).hist(
            bins=40
        )

        plt.title(
            "Bounding Box Aspect Ratio"
        )

        plt.xlabel(
            "Width / Height"
        )

        plt.ylabel(
            "Number of Objects"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "05_bbox_aspect_ratio.png",
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------
    # OBJECT POSITION
    # --------------------------------------------------------

    if not object_df.empty:

        plt.figure(figsize=(8, 8))

        plt.scatter(
            object_df["x_center"],
            object_df["y_center"],
            alpha=0.3,
            s=10,
        )

        plt.xlim(0, 1)
        plt.ylim(1, 0)

        plt.title(
            "Bounding Box Center Distribution"
        )

        plt.xlabel(
            "Normalized X Center"
        )

        plt.ylabel(
            "Normalized Y Center"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "06_object_position.png",
            dpi=200,
        )

        plt.close()


# ============================================================
# VISUAL SAMPLE SELECTION
# ============================================================

def create_visual_sample_list(
    image_records,
    object_records,
    sample_per_category=10,
):

    random.seed(RANDOM_SEED)

    image_df = pd.DataFrame(image_records)
    object_df = pd.DataFrame(object_records)

    if image_df.empty:
        return

    samples = []

    # --------------------------------------------------------
    # BY SIZE
    # --------------------------------------------------------

    if not object_df.empty:

        for category in SIZE_BINS:

            matching_images = (
                object_df[
                    object_df["size_category"] == category
                ]["image"]
                .unique()
                .tolist()
            )

            selected = random.sample(
                matching_images,
                min(
                    sample_per_category,
                    len(matching_images),
                ),
            )

            for image in selected:

                samples.append({
                    "category": f"size_{category}",
                    "image": image,
                })

    # --------------------------------------------------------
    # MULTI OBJECT
    # --------------------------------------------------------

    multi_images = (
        image_df[
            image_df["multi_object"]
        ]["image"]
        .tolist()
    )

    selected = random.sample(
        multi_images,
        min(
            sample_per_category,
            len(multi_images),
        ),
    )

    for image in selected:

        samples.append({
            "category": "multi_object",
            "image": image,
        })

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    pd.DataFrame(samples).drop_duplicates().to_csv(
        OUTPUT_DIR / "visual_sample_list.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CASSAVA YOLO DATASET AUDIT")
    print("=" * 70)

    ensure_output_dir()

    (
        config_info,
        image_records,
        object_records,
        invalid_records,
        hash_records,
    ) = scan_dataset()

    print("\nAnalyzing overlap...")

    overlap_records = analyze_overlap(
        image_records,
        object_records,
    )

    print("Analyzing exact duplicates...")

    duplicate_records = analyze_duplicates(
        hash_records
    )

    print("Saving reports...")

    summary = save_reports(
        config_info,
        image_records,
        object_records,
        invalid_records,
        overlap_records,
        duplicate_records,
    )

    print("Creating visualizations...")

    create_visualizations(
        image_records,
        object_records,
    )

    print("Creating visual sample list...")

    create_visual_sample_list(
        image_records,
        object_records,
    )

    print("\n" + "=" * 70)
    print("AUDIT SELESAI")
    print("=" * 70)

    print(
        f"Images        : {summary['total_images']}"
    )

    print(
        f"Objects       : {summary['total_objects']}"
    )

    print(
        f"Invalid label : {summary['invalid_annotations']}"
    )

    print(
        f"Multi-object  : {summary['multi_object_images']}"
    )

    print(
        f"Overlap pairs : {summary['overlap_pairs']}"
    )

    print(
        f"Duplicates    : {summary['exact_duplicate_images']}"
    )

    print(
        f"\nReport tersedia di:"
        f"\n{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
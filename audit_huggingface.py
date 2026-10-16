from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import re
import sys
import time
from urllib.parse import quote

import requests


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "edcelbogs/cassava-7-dataset"
REVISION = "main"

OUTPUT_DIR = Path("cassava_hf_audit_report")

# Full-image tolerance.
# Bbox will be considered "full-image" if:
# width  >= this threshold
# height >= this threshold
# center is close to (0.5, 0.5)
FULL_IMAGE_SIZE_THRESHOLD = 0.98
FULL_IMAGE_CENTER_TOLERANCE = 0.02

# Object size thresholds based on bbox area / image area.
SIZE_BINS = [
    ("very_small", 0.00, 0.01),
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.20),
    ("large", 0.20, 0.50),
    ("very_large", 0.50, 1.01),
]

REQUEST_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_DELAY = 2


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def safe_get(url, timeout=REQUEST_TIMEOUT):
    """
    GET remote file with retry.
    """
    last_error = None

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Cassava-HuggingFace-Audit/1.0"
                },
            )

            response.raise_for_status()
            return response

        except Exception as exc:
            last_error = exc

            print(
                f"[WARNING] Request failed "
                f"(attempt {attempt}/{RETRY_COUNT}): {exc}"
            )

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Failed to retrieve URL after {RETRY_COUNT} attempts:\n{url}\n"
        f"Last error: {last_error}"
    )


def download_text_file(repo_path):
    """
    Read a single text file directly from Hugging Face.

    IMPORTANT:
    This downloads ONLY the requested label/yaml/text file,
    not the whole dataset.
    """

    encoded_path = quote(repo_path, safe="/")

    url = (
        f"https://huggingface.co/datasets/"
        f"{REPO_ID}/resolve/{REVISION}/{encoded_path}"
        f"?download=true"
    )

    response = safe_get(url)

    return response.text


def find_data_yaml(files):
    candidates = [
        f for f in files
        if Path(f).name.lower() in ("data.yaml", "data.yml")
    ]

    if not candidates:
        return None

    # Prefer root-level data.yaml
    root_candidates = [
        f for f in candidates
        if "/" not in f.strip("/")
    ]

    if root_candidates:
        return root_candidates[0]

    return candidates[0]


def parse_data_yaml_simple(text):
    """
    Lightweight YAML parser.

    We intentionally avoid requiring PyYAML.
    This handles the common YOLO data.yaml structure.
    """

    result = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        # Remove comments
        if "#" in value:
            value = value.split("#", 1)[0].strip()

        # Remove quotes
        value = value.strip("\"'")

        result[key] = value

    return result


def parse_names_from_yaml(text):
    """
    Attempt to parse YOLO names field.

    Supports examples:

    names:
      0: CBB
      1: CBSD

    or

    names: [CBB, CBSD, CGM, CMD, HEALTHY]
    """

    names = {}

    lines = text.splitlines()

    in_names_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("names:"):
            in_names_block = True

            value = stripped[len("names:"):].strip()

            # Inline list
            if value.startswith("[") and value.endswith("]"):
                content = value[1:-1]

                parts = [
                    p.strip().strip("\"'")
                    for p in content.split(",")
                ]

                for idx, name in enumerate(parts):
                    if name:
                        names[idx] = name

                return names

            continue

        if in_names_block:
            if not line.startswith(" ") and not line.startswith("\t"):
                break

            match = re.match(
                r"\s*(\d+)\s*:\s*(.+)",
                line
            )

            if match:
                class_id = int(match.group(1))
                class_name = match.group(2).strip().strip("\"'")
                names[class_id] = class_name

    return names


def parse_yolo_line(line):
    """
    YOLO format:

    class_id x_center y_center width height

    Returns tuple or None.
    """

    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        return (
            class_id,
            x_center,
            y_center,
            width,
            height,
        )

    except ValueError:
        return None


def is_full_image_bbox(x, y, w, h):
    """
    Detect whether bbox effectively covers the whole image.
    """

    return (
        w >= FULL_IMAGE_SIZE_THRESHOLD
        and
        h >= FULL_IMAGE_SIZE_THRESHOLD
        and
        abs(x - 0.5) <= FULL_IMAGE_CENTER_TOLERANCE
        and
        abs(y - 0.5) <= FULL_IMAGE_CENTER_TOLERANCE
    )


def classify_size(area):
    for name, low, high in SIZE_BINS:
        if low <= area < high:
            return name

    return "unknown"


def bbox_to_xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return x1, y1, x2, y2


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def write_csv(path, rows, fieldnames):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# STEP 1 — GET REPOSITORY FILE LIST
# ============================================================

def get_repository_files():
    print_header("STEP 1 — READING HUGGING FACE REPOSITORY")

    api_url = (
        "https://huggingface.co/api/datasets/"
        f"{REPO_ID}/tree/{REVISION}"
        "?recursive=true&expand=false"
    )

    response = safe_get(api_url)

    data = response.json()

    files = []

    for item in data:
        if item.get("type") == "file":
            path = item.get("path")

            if path:
                files.append(path)

    print(f"Repository : {REPO_ID}")
    print(f"Revision   : {REVISION}")
    print(f"Files found: {len(files)}")

    return sorted(files)


# ============================================================
# STEP 2 — CLASSIFY FILES
# ============================================================

def classify_repository_files(files):

    print_header("STEP 2 — CLASSIFYING REPOSITORY FILES")

    images = []
    labels = []
    yamls = []
    other = []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    for path in files:

        suffix = Path(path).suffix.lower()

        if suffix in image_extensions:
            images.append(path)

        elif suffix == ".txt":
            labels.append(path)

        elif suffix in {".yaml", ".yml"}:
            yamls.append(path)

        else:
            other.append(path)

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")
    print(f"YAML   : {len(yamls)}")
    print(f"Other  : {len(other)}")

    return images, labels, yamls, other


# ============================================================
# STEP 3 — IDENTIFY SPLITS
# ============================================================

def identify_split(path):

    normalized = path.lower().replace("\\", "/")

    if "/train/" in f"/{normalized}":
        return "train"

    if "/valid/" in f"/{normalized}":
        return "valid"

    if "/val/" in f"/{normalized}":
        return "valid"

    if "/validation/" in f"/{normalized}":
        return "valid"

    if "/test/" in f"/{normalized}":
        return "test"

    return "unknown"


# ============================================================
# STEP 4 — LOAD LABELS
# ============================================================

def analyze_labels(label_files, class_names):

    print_header("STEP 3 — ANALYZING YOLO LABELS")

    bbox_rows = []
    image_rows = []

    class_counts = Counter()
    split_class_counts = defaultdict(Counter)

    split_image_counts = Counter()
    split_object_counts = Counter()

    full_image_count = 0
    non_full_image_count = 0

    invalid_count = 0
    label_file_errors = 0

    overlap_pair_count = 0
    overlap_significant_count = 0

    bbox_index = 0

    for index, label_path in enumerate(label_files, start=1):

        if index % 100 == 0 or index == 1:
            print(
                f"Processing labels: "
                f"{index}/{len(label_files)}",
                end="\r"
            )

        try:
            text = download_text_file(label_path)

        except Exception as exc:
            label_file_errors += 1

            print(
                f"\n[ERROR] Could not read: "
                f"{label_path}\n{exc}"
            )

            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        split = identify_split(label_path)

        image_name = Path(label_path).stem

        object_count = 0
        valid_object_count = 0
        invalid_object_count = 0

        boxes = []

        for line_number, line in enumerate(lines, start=1):

            object_count += 1

            parsed = parse_yolo_line(line)

            if parsed is None:
                invalid_count += 1
                invalid_object_count += 1
                continue

            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            ) = parsed

            valid = True
            validation_errors = []

            if not all(
                math.isfinite(v)
                for v in [
                    x_center,
                    y_center,
                    width,
                    height,
                ]
            ):
                valid = False
                validation_errors.append("non_finite")

            if not (0 <= x_center <= 1):
                valid = False
                validation_errors.append("x_center_out_of_range")

            if not (0 <= y_center <= 1):
                valid = False
                validation_errors.append("y_center_out_of_range")

            if not (0 < width <= 1):
                valid = False
                validation_errors.append("width_out_of_range")

            if not (0 < height <= 1):
                valid = False
                validation_errors.append("height_out_of_range")

            if class_id < 0:
                valid = False
                validation_errors.append("negative_class")

            if not valid:
                invalid_count += 1
                invalid_object_count += 1

            else:
                valid_object_count += 1

            area_ratio = width * height

            aspect_ratio = (
                width / height
                if height > 0
                else float("nan")
            )

            size_category = classify_size(area_ratio)

            full_image = is_full_image_bbox(
                x_center,
                y_center,
                width,
                height,
            )

            if full_image:
                full_image_count += 1
            else:
                non_full_image_count += 1

            class_name = class_names.get(
                class_id,
                f"class_{class_id}"
            )

            class_counts[class_id] += 1
            split_class_counts[split][class_id] += 1

            split_object_counts[split] += 1

            xyxy = bbox_to_xyxy(
                x_center,
                y_center,
                width,
                height,
            )

            boxes.append(
                (
                    xyxy,
                    class_id
                )
            )

            bbox_rows.append({
                "split": split,
                "label_file": label_path,
                "image_name": image_name,
                "line_number": line_number,
                "class_id": class_id,
                "class_name": class_name,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area_ratio": area_ratio,
                "aspect_ratio": aspect_ratio,
                "size_category": size_category,
                "is_full_image_bbox": full_image,
                "validation_status": (
                    "valid"
                    if valid
                    else "invalid"
                ),
                "validation_errors": ";".join(
                    validation_errors
                ),
            })

            bbox_index += 1

        split_image_counts[split] += 1

        image_rows.append({
            "split": split,
            "label_file": label_path,
            "image_name": image_name,
            "object_count": object_count,
            "valid_object_count": valid_object_count,
            "invalid_object_count": invalid_object_count,
            "is_multi_object": valid_object_count >= 2,
        })

        # Pairwise overlap
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):

                box_a, class_a = boxes[i]
                box_b, class_b = boxes[j]

                iou = calculate_iou(
                    box_a,
                    box_b
                )

                if iou > 0:
                    overlap_pair_count += 1

                if iou > 0.5:
                    overlap_significant_count += 1

    print()
    print(f"Total bbox           : {bbox_index}")
    print(f"Invalid annotations  : {invalid_count}")
    print(f"Full-image bbox      : {full_image_count}")
    print(f"Non-full-image bbox  : {non_full_image_count}")
    print(f"Overlap pairs        : {overlap_pair_count}")
    print(
        f"Significant overlap : "
        f"{overlap_significant_count}"
    )

    return {
        "bbox_rows": bbox_rows,
        "image_rows": image_rows,
        "class_counts": class_counts,
        "split_class_counts": split_class_counts,
        "split_image_counts": split_image_counts,
        "split_object_counts": split_object_counts,
        "full_image_count": full_image_count,
        "non_full_image_count": non_full_image_count,
        "invalid_count": invalid_count,
        "label_file_errors": label_file_errors,
        "overlap_pair_count": overlap_pair_count,
        "overlap_significant_count": overlap_significant_count,
    }


# ============================================================
# STEP 5 — CHECK IMAGE/LABEL PAIRING
# ============================================================

def check_image_label_pairing(images, labels):

    print_header("STEP 4 — CHECKING IMAGE/LABEL PAIRING")

    image_stems = defaultdict(list)
    label_stems = defaultdict(list)

    for path in images:
        stem = Path(path).stem.lower()
        image_stems[stem].append(path)

    for path in labels:
        stem = Path(path).stem.lower()
        label_stems[stem].append(path)

    missing_labels = []
    orphan_labels = []

    for stem, paths in image_stems.items():

        if stem not in label_stems:
            missing_labels.extend(paths)

    for stem, paths in label_stems.items():

        if stem not in image_stems:
            orphan_labels.extend(paths)

    print(
        f"Images without labels : "
        f"{len(missing_labels)}"
    )

    print(
        f"Labels without images : "
        f"{len(orphan_labels)}"
    )

    return {
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
    }


# ============================================================
# STEP 6 — GENERATE REPORT
# ============================================================

def generate_report(
    files,
    images,
    labels,
    yamls,
    data_yaml,
    yaml_text,
    analysis,
    pairing,
):

    print_header("STEP 5 — GENERATING REPORTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    bbox_rows = analysis["bbox_rows"]
    image_rows = analysis["image_rows"]

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_bbox = len(bbox_rows)

    full_image_pct = (
        analysis["full_image_count"]
        / total_bbox * 100
        if total_bbox
        else 0
    )

    non_full_pct = (
        analysis["non_full_image_count"]
        / total_bbox * 100
        if total_bbox
        else 0
    )

    multi_object_images = sum(
        1
        for row in image_rows
        if row["is_multi_object"]
    )

    total_images_with_labels = len(image_rows)

    multi_object_pct = (
        multi_object_images
        / total_images_with_labels * 100
        if total_images_with_labels
        else 0
    )

    # Determine verdict

    if total_bbox == 0:

        verdict = "NO_BBOX_DATA"

    elif full_image_pct >= 95:

        verdict = (
            "FATAL_FULL_IMAGE_ANNOTATION"
        )

    elif full_image_pct >= 80:

        verdict = (
            "MOSTLY_FULL_IMAGE_ANNOTATION"
        )

    elif (
        full_image_pct < 50
        and
        analysis["non_full_image_count"] > 0
    ):

        verdict = (
            "POTENTIALLY_VALID_OBJECT_DETECTION"
        )

    else:

        verdict = (
            "NEEDS_MANUAL_REVIEW"
        )

    summary = {
        "repository": REPO_ID,
        "revision": REVISION,

        "total_repository_files": len(files),
        "total_images": len(images),
        "total_label_files": len(labels),
        "total_yaml_files": len(yamls),

        "data_yaml": data_yaml,

        "total_bounding_boxes": total_bbox,

        "full_image_bbox_count":
            analysis["full_image_count"],

        "full_image_bbox_percentage":
            round(full_image_pct, 4),

        "non_full_image_bbox_count":
            analysis["non_full_image_count"],

        "non_full_image_bbox_percentage":
            round(non_full_pct, 4),

        "invalid_annotation_count":
            analysis["invalid_count"],

        "label_file_read_errors":
            analysis["label_file_errors"],

        "images_without_labels":
            len(pairing["missing_labels"]),

        "labels_without_images":
            len(pairing["orphan_labels"]),

        "multi_object_image_count":
            multi_object_images,

        "multi_object_image_percentage":
            round(multi_object_pct, 4),

        "overlap_pair_count":
            analysis["overlap_pair_count"],

        "significant_overlap_pair_count":
            analysis["overlap_significant_count"],

        "verdict": verdict,

        "interpretation": (
            "The dataset appears to contain "
            "lesion-level/object-level annotations."
            if verdict
            == "POTENTIALLY_VALID_OBJECT_DETECTION"
            else
            "The dataset requires further inspection."
        ),
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE REPOSITORY FILE LIST
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "repository_files.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for path in files:
            f.write(path + "\n")

    # --------------------------------------------------------
    # SAVE YAML
    # --------------------------------------------------------

    if yaml_text:

        with open(
            OUTPUT_DIR / "data.yaml.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(yaml_text)

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    class_rows = []

    for class_id, count in sorted(
        analysis["class_counts"].items()
    ):

        class_name = "unknown"

        if yaml_text:
            names = parse_names_from_yaml(
                yaml_text
            )

            class_name = names.get(
                class_id,
                f"class_{class_id}"
            )

        percentage = (
            count / total_bbox * 100
            if total_bbox
            else 0
        )

        class_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "object_count": count,
            "percentage": round(
                percentage,
                4
            ),
        })

    write_csv(
        OUTPUT_DIR / "class_distribution.csv",
        class_rows,
        [
            "class_id",
            "class_name",
            "object_count",
            "percentage",
        ]
    )

    # --------------------------------------------------------
    # BBOX STATISTICS
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIR / "bbox_statistics.csv",
        bbox_rows,
        [
            "split",
            "label_file",
            "image_name",
            "line_number",
            "class_id",
            "class_name",
            "x_center",
            "y_center",
            "width",
            "height",
            "area_ratio",
            "aspect_ratio",
            "size_category",
            "is_full_image_bbox",
            "validation_status",
            "validation_errors",
        ]
    )

    # --------------------------------------------------------
    # IMAGE STATISTICS
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIR / "image_statistics.csv",
        image_rows,
        [
            "split",
            "label_file",
            "image_name",
            "object_count",
            "valid_object_count",
            "invalid_object_count",
            "is_multi_object",
        ]
    )

    # --------------------------------------------------------
    # PAIRING PROBLEMS
    # --------------------------------------------------------

    pairing_rows = []

    for path in pairing["missing_labels"]:

        pairing_rows.append({
            "type": "image_without_label",
            "path": path,
        })

    for path in pairing["orphan_labels"]:

        pairing_rows.append({
            "type": "label_without_image",
            "path": path,
        })

    write_csv(
        OUTPUT_DIR / "pairing_problems.csv",
        pairing_rows,
        [
            "type",
            "path",
        ]
    )

    # --------------------------------------------------------
    # HUMAN-READABLE VERDICT
    # --------------------------------------------------------

    verdict_text = []

    verdict_text.append(
        "CASSAVA HUGGING FACE DATASET AUDIT"
    )
    verdict_text.append("=" * 60)
    verdict_text.append("")
    verdict_text.append(
        f"Repository : {REPO_ID}"
    )
    verdict_text.append(
        f"Revision   : {REVISION}"
    )
    verdict_text.append("")
    verdict_text.append(
        f"Images            : {len(images)}"
    )
    verdict_text.append(
        f"Label files       : {len(labels)}"
    )
    verdict_text.append(
        f"Bounding boxes    : {total_bbox}"
    )
    verdict_text.append("")
    verdict_text.append(
        f"Full-image bbox   : "
        f"{analysis['full_image_count']} "
        f"({full_image_pct:.2f}%)"
    )
    verdict_text.append(
        f"Non-full bbox     : "
        f"{analysis['non_full_image_count']} "
        f"({non_full_pct:.2f}%)"
    )
    verdict_text.append("")
    verdict_text.append(
        f"Multi-object imgs : "
        f"{multi_object_images} "
        f"({multi_object_pct:.2f}%)"
    )
    verdict_text.append(
        f"Overlap pairs     : "
        f"{analysis['overlap_pair_count']}"
    )
    verdict_text.append(
        f"Significant IoU>0.5: "
        f"{analysis['overlap_significant_count']}"
    )
    verdict_text.append("")
    verdict_text.append(
        f"Invalid bbox      : "
        f"{analysis['invalid_count']}"
    )
    verdict_text.append(
        f"Missing labels    : "
        f"{len(pairing['missing_labels'])}"
    )
    verdict_text.append(
        f"Orphan labels     : "
        f"{len(pairing['orphan_labels'])}"
    )
    verdict_text.append("")
    verdict_text.append(
        "VERDICT:"
    )
    verdict_text.append(
        verdict
    )
    verdict_text.append("")

    if verdict == "FATAL_FULL_IMAGE_ANNOTATION":

        verdict_text.extend([
            "WARNING:",
            "Most bounding boxes cover essentially",
            "the entire image.",
            "",
            "This strongly suggests that the dataset",
            "is classification data converted into",
            "YOLO format rather than true lesion-level",
            "object detection annotations.",
            "",
            "DO NOT TRAIN YOLO FOR LESION DETECTION",
            "before manually validating the annotations.",
        ])

    elif verdict == "POTENTIALLY_VALID_OBJECT_DETECTION":

        verdict_text.extend([
            "PROMISING:",
            "A substantial proportion of bounding boxes",
            "are smaller than the entire image.",
            "",
            "The dataset may support genuine object",
            "detection research.",
            "",
            "NEXT STEP:",
            "Perform visual inspection on representative",
            "images before accepting the dataset.",
        ])

    else:

        verdict_text.extend([
            "CAUTION:",
            "The automatic audit cannot establish the",
            "semantic quality of the annotations.",
            "",
            "Visual inspection is required.",
        ])

    with open(
        OUTPUT_DIR / "AUDIT_VERDICT.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(verdict_text)
        )

    # --------------------------------------------------------
    # PRINT FINAL RESULT
    # --------------------------------------------------------

    print_header("FINAL RESULT")

    print(
        f"Images             : {len(images)}"
    )

    print(
        f"Bounding boxes     : {total_bbox}"
    )

    print(
        f"Full-image bbox    : "
        f"{analysis['full_image_count']} "
        f"({full_image_pct:.2f}%)"
    )

    print(
        f"Non-full bbox      : "
        f"{analysis['non_full_image_count']} "
        f"({non_full_pct:.2f}%)"
    )

    print(
        f"Multi-object image : "
        f"{multi_object_images} "
        f"({multi_object_pct:.2f}%)"
    )

    print(
        f"Invalid annotation : "
        f"{analysis['invalid_count']}"
    )

    print(
        f"Overlap pairs      : "
        f"{analysis['overlap_pair_count']}"
    )

    print()
    print(
        "VERDICT:"
    )

    print(verdict)

    print()
    print(
        f"Reports saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("CASSAVA HUGGING FACE DATASET AUDITOR")
    print("=" * 72)
    print()
    print(
        "Repository:"
    )
    print(
        f"https://huggingface.co/datasets/{REPO_ID}"
    )
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This script DOES NOT download the entire dataset."
    )
    print(
        "It reads repository metadata and individual"
    )
    print(
        "YOLO label files remotely."
    )
    print()

    try:

        # ----------------------------------------------------
        # Repository files
        # ----------------------------------------------------

        files = get_repository_files()

        # ----------------------------------------------------
        # File classification
        # ----------------------------------------------------

        (
            images,
            labels,
            yamls,
            other
        ) = classify_repository_files(files)

        # ----------------------------------------------------
        # YAML
        # ----------------------------------------------------

        data_yaml = find_data_yaml(files)

        yaml_text = None

        if data_yaml:

            print_header(
                "READING DATA.YAML"
            )

            print(
                f"Found: {data_yaml}"
            )

            try:

                yaml_text = download_text_file(
                    data_yaml
                )

                print()
                print(yaml_text[:3000])

            except Exception as exc:

                print(
                    f"[WARNING] "
                    f"Could not read data.yaml: {exc}"
                )

        else:

            print(
                "[WARNING] data.yaml not found."
            )

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------

        class_names = {}

        if yaml_text:

            class_names = parse_names_from_yaml(
                yaml_text
            )

        if class_names:

            print()
            print("Detected classes:")

            for class_id, name in sorted(
                class_names.items()
            ):

                print(
                    f"  {class_id}: {name}"
                )

        # ----------------------------------------------------
        # Image / label pairing
        # ----------------------------------------------------

        pairing = check_image_label_pairing(
            images,
            labels
        )

        # ----------------------------------------------------
        # Analyze labels
        # ----------------------------------------------------

        analysis = analyze_labels(
            labels,
            class_names
        )

        # ----------------------------------------------------
        # Generate report
        # ----------------------------------------------------

        generate_report(
            files=files,
            images=images,
            labels=labels,
            yamls=yamls,
            data_yaml=data_yaml,
            yaml_text=yaml_text,
            analysis=analysis,
            pairing=pairing,
        )

    except KeyboardInterrupt:

        print()
        print(
            "Audit cancelled by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 72)
        print("AUDIT FAILED")
        print("=" * 72)
        print()
        print(
            f"Error: {exc}"
        )
        print()
        print(
            "If this error persists, copy the complete"
        )
        print(
            "error message and send it to me."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
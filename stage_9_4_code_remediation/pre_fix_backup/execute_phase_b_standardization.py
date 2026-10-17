import os
import csv
import glob
import hashlib
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

SRC_ROOT = Path("cassava-v7-research")
DST_ROOT = Path("cassava-v7-standardized")
VIS_DIR = Path("stage_10_data_audit/visualizations")
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]

CLASS_COLORS = {
    0: (0, 0, 255),    # CBB: Red
    1: (255, 128, 0),  # CBSD: Orange/Blue
    2: (0, 255, 0),    # CGM: Green
    3: (255, 0, 255),  # CMD: Magenta
    4: (0, 255, 255),  # HEALTHY: Yellow
}

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def md5_file(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def convert_polygon_to_bbox(coords):
    """Converts a flat list of normalized polygon coordinates to (xc, yc, w, h)."""
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))

    x_min = max(0.0, min(1.0, x_min))
    x_max = max(0.0, min(1.0, x_max))
    y_min = max(0.0, min(1.0, y_min))
    y_max = max(0.0, min(1.0, y_max))

    w = max(1e-6, x_max - x_min)
    h = max(1e-6, y_max - y_min)
    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    return xc, yc, w, h

def parse_raw_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cid = int(parts[0])
    raw_coords = [float(x) for x in parts[1:]]
    if len(raw_coords) == 4:
        # Bbox
        xc, yc, w, h = raw_coords
        return {"type": "bbox", "class_id": cid, "xc": xc, "yc": yc, "w": w, "h": h, "raw": raw_coords}
    else:
        # Polygon
        xc, yc, w, h = convert_polygon_to_bbox(raw_coords)
        return {"type": "polygon", "class_id": cid, "xc": xc, "yc": yc, "w": w, "h": h, "raw": raw_coords}

def draw_overlay(img_path, raw_lines, std_lines, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    H, W = img.shape[:2]

    # Canvas Before
    canvas_before = img.copy()
    cv2.putText(canvas_before, "BEFORE (Raw Label: Polygon + BBox)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for l in raw_lines:
        parsed = parse_raw_label_line(l)
        if not parsed:
            continue
        cid = parsed["class_id"]
        cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
        color = CLASS_COLORS.get(cid, (255, 255, 255))

        if parsed["type"] == "polygon":
            pts = np.array(parsed["raw"], dtype=np.float32).reshape(-1, 2)
            pts_px = np.int32(pts * [W, H])
            cv2.polylines(canvas_before, [pts_px], isClosed=True, color=color, thickness=2)
            # Label
            cv2.putText(canvas_before, f"Poly:{cname}", (pts_px[0][0], max(15, pts_px[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            xc, yc, w, h = parsed["xc"], parsed["yc"], parsed["w"], parsed["h"]
            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)
            cv2.rectangle(canvas_before, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas_before, f"Box:{cname}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Canvas After
    canvas_after = img.copy()
    cv2.putText(canvas_after, "AFTER (Standardized YOLO BBox)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for l in std_lines:
        parts = l.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        xc, yc, w, h = [float(x) for x in parts[1:]]
        cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
        color = CLASS_COLORS.get(cid, (255, 255, 255))

        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)
        cv2.rectangle(canvas_after, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas_after, f"{cname}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Combine side-by-side
    combined = np.hstack([canvas_before, canvas_after])
    cv2.imwrite(str(out_path), combined)

def standardize():
    splits = ["train", "valid", "test"]
    manifest_rows = []
    checksum_rows = []

    # Target visualization samples
    vis_samples = {
        "train": [
            "253146039_jpg.rf.9e814fd03d5d2b739164a536d0d5b2a5", # Mixed sample 1
            "image_1012_jpg.rf.d53a95ef6a505eaeef7a6119c5ac01de", # Mixed sample 4
            "image_947_jpg.rf.cb175d6bae37e56344bbe71dbaea1e5d",  # Mixed sample 10
            "1000015157_jpg.rf.3484b29e2abaad936b27d5d90cfd2d78", # Pure polygon control
        ],
        "valid": [
            "image_1000_jpg.rf.6cfb821bfe27d81187278c669937c9ea", # Mixed valid 1
            "image_918_jpg.rf.a20c88d5772e34bff9f7fc5d858dd355",  # Mixed valid 2
        ],
        "test": [
            "image_996_jpg.rf.5ff1d580dd6268900bde22c1c321a2aa",  # Mixed test 1
        ]
    }

    print("=" * 80)
    print("EXECUTING STANDARDIZATION TO cassava-v7-standardized")
    print("=" * 80)

    for s in splits:
        src_img_dir = SRC_ROOT / s / "images"
        src_lbl_dir = SRC_ROOT / s / "labels"
        dst_img_dir = DST_ROOT / s / "images"
        dst_lbl_dir = DST_ROOT / s / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        lbl_files = sorted(list(src_lbl_dir.glob("*.txt")))
        img_files = sorted(list(src_img_dir.glob("*.jpg")))
        print(f"\nProcessing {s.upper()}: {len(img_files)} images, {len(lbl_files)} labels...")

        # 1. Hardlink images if not already existing
        for im in img_files:
            dst_im = dst_img_dir / im.name
            if not dst_im.exists():
                try:
                    os.link(str(im), str(dst_im))
                except FileExistsError:
                    pass

        # 2. Process and standardize labels
        total_poly_converted = 0
        total_bbox_preserved = 0
        mixed_count = 0

        for lf in lbl_files:
            stem = lf.stem
            raw_text = lf.read_text(encoding="utf-8").strip()
            raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

            std_lines = []
            file_poly = 0
            file_bbox = 0

            for l in raw_lines:
                parsed = parse_raw_label_line(l)
                if not parsed:
                    continue
                if parsed["type"] == "polygon":
                    file_poly += 1
                else:
                    file_bbox += 1
                # Format standardized YOLO detection line
                std_line = f"{parsed['class_id']} {parsed['xc']:.6f} {parsed['yc']:.6f} {parsed['w']:.6f} {parsed['h']:.6f}"
                std_lines.append(std_line)

            is_mixed = (file_poly > 0 and file_bbox > 0)
            if is_mixed:
                mixed_count += 1
            total_poly_converted += file_poly
            total_bbox_preserved += file_bbox

            # Write clean standardized label
            dst_lf = dst_lbl_dir / lf.name
            dst_text = "\n".join(std_lines) + ("\n" if std_lines else "")
            dst_lf.write_text(dst_text, encoding="utf-8")

            # Checksums
            lbl_sha = sha256_file(dst_lf)
            lbl_md5 = md5_file(dst_lf)

            manifest_rows.append({
                "split": s,
                "file_name": lf.name,
                "objects_before": len(raw_lines),
                "objects_after": len(std_lines),
                "objects_match": (len(raw_lines) == len(std_lines)),
                "was_mixed_format": is_mixed,
                "polygon_rows_converted": file_poly,
                "bbox_rows_preserved": file_bbox,
                "is_empty_background": (len(std_lines) == 0),
                "sha256": lbl_sha,
                "md5": lbl_md5,
            })

            # Check if sample for visualization
            if stem in vis_samples.get(s, []):
                matching_im = src_img_dir / f"{stem}.jpg"
                if matching_im.exists():
                    vis_out = VIS_DIR / f"{s}_{stem}_comparison.png"
                    draw_overlay(matching_im, raw_lines, std_lines, vis_out)
                    print(f"  [VISUALIZATION] Saved overlay comparison: {vis_out.name}")

        print(f"Finished {s.upper()}:")
        print(f"  Total polygon rows converted to bbox : {total_poly_converted}")
        print(f"  Total bbox rows preserved           : {total_bbox_preserved}")
        print(f"  Mixed-format files resolved          : {mixed_count}")
        print(f"  Total standardized label files       : {len(lbl_files)}")

    # Write manifest CSV
    manifest_csv = Path("stage_10_data_audit/label_conversion_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split", "file_name", "objects_before", "objects_after", "objects_match",
            "was_mixed_format", "polygon_rows_converted", "bbox_rows_preserved",
            "is_empty_background", "sha256", "md5"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
    print(f"\nSaved {manifest_csv} with {len(manifest_rows)} records.")

    # Write data.yaml for cassava-v7-standardized
    yaml_content = f"""path: {DST_ROOT.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['CBB', 'CBSD', 'CGM', 'CMD', 'HEALTHY']
"""
    (DST_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"Saved {DST_ROOT / 'data.yaml'}")

    # Generate dataset checksums summary
    for s in splits:
        img_hashes = [sha256_file(p) for p in (DST_ROOT / s / "images").glob("*.jpg")]
        lbl_hashes = [sha256_file(p) for p in (DST_ROOT / s / "labels").glob("*.txt")]
        combined_img_hash = hashlib.sha256("".join(sorted(img_hashes)).encode("utf-8")).hexdigest()
        combined_lbl_hash = hashlib.sha256("".join(sorted(lbl_hashes)).encode("utf-8")).hexdigest()
        checksum_rows.append({
            "split": s,
            "total_images": len(img_hashes),
            "total_labels": len(lbl_hashes),
            "images_aggregate_sha256": combined_img_hash,
            "labels_aggregate_sha256": combined_lbl_hash,
        })

    checksum_csv = Path("stage_10_data_audit/dataset_checksums.csv")
    with open(checksum_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "total_images", "total_labels", "images_aggregate_sha256", "labels_aggregate_sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in checksum_rows:
            writer.writerow(r)
    print(f"Saved {checksum_csv}")

if __name__ == "__main__":
    standardize()

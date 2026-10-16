import os
import re
import json
import math
import csv
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse

import requests

try:
    from remotezip import RemoteZip
except ImportError:
    import subprocess
    import sys
    print("[INFO] Installing remotezip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "remotezip"])
    from remotezip import RemoteZip


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_URL = (
    "https://universe.roboflow.com/"
    "cassava-disease/cassava-zpvrj/dataset/7"
)

def _get_api_key():
    key = os.getenv("ROBOFLOW_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

ROBOFLOW_API_KEY = _get_api_key()

OUTPUT_DIR = Path("roboflow_cassava_v7_audit")

TIMEOUT = 60

# Bounding-box thresholds
FULL_CENTER_TOL = 0.02
FULL_SIZE_THRESHOLD = 0.98

SIZE_THRESHOLDS = {
    "very_small": 0.01,
    "small": 0.05,
    "medium": 0.20,
    "large": 0.50,
}

IOU_THRESHOLDS = [0.10, 0.30, 0.50]


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
})


# ============================================================
# UTILITY
# ============================================================

def safe_get(url, params=None):
    try:
        response = session.get(url, params=params, timeout=TIMEOUT)
        return response
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {url}")
        print(f"        {exc}")
        return None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_dataset_url(url):
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    result = {
        "host": parsed.netloc,
        "workspace": "cassava-disease",
        "project": "cassava-zpvrj",
        "version": 7,
    }
    try:
        dataset_index = parts.index("dataset")
        result["workspace"] = parts[dataset_index - 2]
        result["project"] = parts[dataset_index - 1]
        result["version"] = int(parts[dataset_index + 1])
    except Exception:
        pass
    return result


def is_full_image_bbox(xc, yc, width, height):
    center_ok = (
        abs(xc - 0.5) <= FULL_CENTER_TOL
        and abs(yc - 0.5) <= FULL_CENTER_TOL
    )
    size_ok = (
        width >= FULL_SIZE_THRESHOLD
        and height >= FULL_SIZE_THRESHOLD
    )
    return center_ok and size_ok


def classify_object_size(rel_area):
    if rel_area < SIZE_THRESHOLDS["very_small"]:
        return "very_small"
    if rel_area < SIZE_THRESHOLDS["small"]:
        return "small"
    if rel_area < SIZE_THRESHOLDS["medium"]:
        return "medium"
    if rel_area < SIZE_THRESHOLDS["large"]:
        return "large"
    return "very_large"


def bbox_xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ============================================================
# AUDITOR ENGINE
# ============================================================

class RoboflowRemoteAuditor:
    def __init__(self, workspace, project, version, api_key):
        self.workspace = workspace
        self.project = project
        self.version = version
        self.api_key = api_key

        self.metadata = {}
        self.categories = {}
        self.split_summaries = {}

        self.all_bbox_rows = []
        self.all_image_rows = []
        self.all_areas = []

        self.class_counts = Counter()
        self.split_class_counts = defaultdict(Counter)
        self.split_image_counts = Counter()
        self.split_object_counts = Counter()

        self.size_counts = Counter()
        self.total_objects = 0
        self.full_image_count = 0
        self.invalid_count = 0
        self.multi_object_images = 0

        self.overlap_pairs = Counter()

        self.base_images = defaultdict(set)
        self.all_base_images = defaultdict(list)

    def fetch_project_metadata(self):
        print("[1/5] Mengambil metadata proyek dari Roboflow REST API...")
        url = f"https://api.roboflow.com/{self.workspace}/{self.project}/{self.version}?api_key={self.api_key}"
        resp = safe_get(url)
        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"Gagal mengakses Roboflow API. Status code: {resp.status_code if resp else 'None'}")

        data = resp.json()
        self.metadata = {
            "workspace": data.get("workspace", {}),
            "project": data.get("project", {}),
            "version": data.get("version", {}),
        }
        print(f"      Proyek: {self.workspace}/{self.project} (Versi {self.version})")
        print(f"      Total Gambar (Roboflow): {data.get('version', {}).get('images', 'N/A')}")
        print(f"      Splits: {data.get('version', {}).get('splits', {})}")
        return self.metadata

    def get_export_url(self):
        print("[2/5] Meminta link ekspor COCO JSON...")
        url = f"https://api.roboflow.com/{self.workspace}/{self.project}/{self.version}/coco?api_key={self.api_key}"
        resp = safe_get(url)
        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"Gagal meminta ekspor COCO dari Roboflow API: {resp.text if resp else 'None'}")

        export_data = resp.json()
        export_link = export_data.get("export", {}).get("link")
        if not export_link:
            raise RuntimeError("Link ekspor tidak ditemukan di respons Roboflow.")

        # Follow redirects to resolve direct cloud storage URL
        print("      Menyelesaikan direct Google Cloud Storage URL...")
        head_resp = requests.head(export_link, allow_redirects=True, timeout=TIMEOUT)
        direct_url = head_resp.url
        print(f"      Ukuran ZIP: {export_data.get('export', {}).get('size', 0):.2f} MB")
        return direct_url

    def audit_remote_zip(self, zip_url):
        print("[3/5] Membuka remote ZIP via HTTP Range Requests (tanpa download gambar)...")
        with RemoteZip(zip_url) as z:
            namelist = z.namelist()
            print(f"      Total file di remote ZIP: {len(namelist)} file")

            json_files = [f for f in namelist if f.endswith(".json")]
            print(f"      File anotasi JSON ditemukan: {json_files}")

            splits = ["train", "valid", "test"]
            for split in splits:
                target_json = f"{split}/_annotations.coco.json"
                if target_json not in namelist:
                    print(f"[WARN] {target_json} tidak ditemukan di ZIP.")
                    continue

                print(f"[4/5] Membaca dan menganalisis anotasi: {target_json}...")
                raw_bytes = z.read(target_json)
                print(f"      Ukuran payload JSON terkompresi yang diunduh: {len(raw_bytes) / (1024*1024):.2f} MB")

                coco_data = json.loads(raw_bytes.decode("utf-8"))
                self._process_coco_split(split, coco_data)

    def _process_coco_split(self, split, coco_data):
        # Parse categories
        cat_map = {}
        for cat in coco_data.get("categories", []):
            cid = cat.get("id")
            cname = cat.get("name")
            cat_map[cid] = cname
            if cid not in self.categories:
                self.categories[cid] = cname

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])

        print(f"      -> {split.upper()}: {len(images)} gambar, {len(annotations)} anotasi")

        # Map images by id
        img_by_id = {}
        for img in images:
            img_id = img["id"]
            img_by_id[img_id] = img
            fname = img.get("file_name", "")
            # Extract base name before .rf.<hash>
            match = re.match(r"^(.*?)(?:\.rf\.[a-f0-9]+)?\.[a-zA-Z]+$", fname)
            base = match.group(1) if match else fname
            self.base_images[split].add(base)
            self.all_base_images[base].append((split, fname))

        # Group annotations by image
        annos_by_img = defaultdict(list)
        for ann in annotations:
            annos_by_img[ann["image_id"]].append(ann)

        self.split_image_counts[split] += len(images)

        for img in images:
            img_id = img["id"]
            img_w = float(img.get("width", 640))
            img_h = float(img.get("height", 640))
            img_area = img_w * img_h
            fname = img.get("file_name", "")

            img_annos = annos_by_img.get(img_id, [])
            obj_count = len(img_annos)
            if obj_count >= 2:
                self.multi_object_images += 1

            self.all_image_rows.append({
                "split": split,
                "image_id": img_id,
                "file_name": fname,
                "width": img_w,
                "height": img_h,
                "object_count": obj_count,
                "is_multi_object": obj_count >= 2,
            })

            # Process individual bounding boxes
            boxes_for_iou = []
            for ann in img_annos:
                self.total_objects += 1
                self.split_object_counts[split] += 1

                cid = ann.get("category_id")
                cname = cat_map.get(cid, f"class_{cid}")
                self.class_counts[cname] += 1
                self.split_class_counts[split][cname] += 1

                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    self.invalid_count += 1
                    continue

                bx, by, bw, bh = [float(v) for v in bbox]

                # Validation
                if bw <= 0 or bh <= 0 or not math.isfinite(bx) or not math.isfinite(by):
                    self.invalid_count += 1
                    continue

                # Relative metrics
                rel_w = bw / img_w
                rel_h = bh / img_h
                rel_xc = (bx + bw / 2.0) / img_w
                rel_yc = (by + bh / 2.0) / img_h
                rel_area = (bw * bh) / img_area

                self.all_areas.append(rel_area)

                full_img = is_full_image_bbox(rel_xc, rel_yc, rel_w, rel_h)
                if full_img:
                    self.full_image_count += 1

                size_cat = classify_object_size(rel_area)
                self.size_counts[size_cat] += 1

                # Save bbox info
                self.all_bbox_rows.append({
                    "split": split,
                    "image_id": img_id,
                    "file_name": fname,
                    "class_id": cid,
                    "class_name": cname,
                    "x": bx,
                    "y": by,
                    "w": bw,
                    "h": bh,
                    "rel_xc": round(rel_xc, 5),
                    "rel_yc": round(rel_yc, 5),
                    "rel_w": round(rel_w, 5),
                    "rel_h": round(rel_h, 5),
                    "rel_area": round(rel_area, 5),
                    "size_category": size_cat,
                    "is_full_image": full_img,
                })

                boxes_for_iou.append(bbox_xywh_to_xyxy(bx, by, bw, bh))

            # Pairwise overlap inside same image
            for i in range(len(boxes_for_iou)):
                for j in range(i + 1, len(boxes_for_iou)):
                    iou = calculate_iou(boxes_for_iou[i], boxes_for_iou[j])
                    for th in IOU_THRESHOLDS:
                        if iou >= th:
                            self.overlap_pairs[f"iou_ge_{th:.2f}"] += 1

    @property
    def total_images(self):
        return sum(self.split_image_counts.values())

    def generate_report(self):
        print("[5/5] Menyusun laporan audit komprehensif...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        tot_ann = self.total_objects
        tot_img = self.total_images

        full_img_pct = (self.full_image_count / tot_ann * 100) if tot_ann else 0
        multi_obj_pct = (self.multi_object_images / tot_img * 100) if tot_img else 0

        # Verdict logic
        if tot_ann == 0:
            verdict = "NO_BBOX_DATA"
            verdict_desc = "Tidak ada bounding box yang ditemukan."
        elif full_img_pct >= 90:
            verdict = "FATAL_FULL_IMAGE_ANNOTATION"
            verdict_desc = f"{full_img_pct:.2f}% bounding box menutupi seluruh gambar (klasifikasi semu)."
        elif full_img_pct < 20 and (self.size_counts["small"] + self.size_counts["medium"]) / tot_ann >= 0.30:
            verdict = "POTENTIALLY_VALID_OBJECT_DETECTION"
            verdict_desc = "Dataset memiliki karakteristik deteksi objek riil dengan variasi ukuran dan multi-object."
        else:
            verdict = "NEEDS_MANUAL_REVIEW"
            verdict_desc = "Karakteristik objek memerlukan inspeksi visual lanjutan."

        summary = {
            "dataset_url": DATASET_URL,
            "workspace": self.workspace,
            "project": self.project,
            "version": self.version,
            "verdict": verdict,
            "verdict_description": verdict_desc,
            "images": {
                "total": tot_img,
                "by_split": dict(self.split_image_counts),
                "unique_base_images": len(self.all_base_images),
            },
            "objects": {
                "total": tot_ann,
                "by_split": dict(self.split_object_counts),
                "full_image_count": self.full_image_count,
                "full_image_percentage": round(full_img_pct, 2),
                "invalid_annotations": self.invalid_count,
            },
            "multi_object": {
                "multi_object_images": self.multi_object_images,
                "multi_object_percentage": round(multi_obj_pct, 2),
                "mean_objects_per_image": round(tot_ann / tot_img, 3) if tot_img else 0,
            },
            "size_distribution": {
                k: {
                    "count": self.size_counts[k],
                    "percentage": round(self.size_counts[k] / tot_ann * 100, 2) if tot_ann else 0,
                }
                for k in ["very_small", "small", "medium", "large", "very_large"]
            },
            "area_statistics": {
                "mean_relative_area": round(sum(self.all_areas) / len(self.all_areas), 5) if self.all_areas else 0,
                "min_relative_area": round(min(self.all_areas), 5) if self.all_areas else 0,
                "max_relative_area": round(max(self.all_areas), 5) if self.all_areas else 0,
            },
            "classes": {
                cls: {
                    "total_objects": self.class_counts[cls],
                    "percentage": round(self.class_counts[cls] / tot_ann * 100, 2) if tot_ann else 0,
                    "by_split": {
                        split: self.split_class_counts[split][cls]
                        for split in ["train", "valid", "test"]
                    },
                }
                for cls in sorted(self.class_counts.keys())
            },
            "overlap_pairs": dict(self.overlap_pairs),
        }

        save_json(OUTPUT_DIR / "summary.json", summary)
        save_json(OUTPUT_DIR / "metadata.json", self.metadata)

        # Class distribution CSV
        class_rows = []
        for cls in sorted(self.class_counts.keys()):
            cnt = self.class_counts[cls]
            class_rows.append({
                "class_name": cls,
                "total_objects": cnt,
                "percentage": round(cnt / tot_ann * 100, 2) if tot_ann else 0,
                "train": self.split_class_counts["train"][cls],
                "valid": self.split_class_counts["valid"][cls],
                "test": self.split_class_counts["test"][cls],
            })
        save_csv(
            OUTPUT_DIR / "class_distribution.csv",
            class_rows,
            ["class_name", "total_objects", "percentage", "train", "valid", "test"],
        )

        # Size distribution CSV
        size_rows = []
        for sz in ["very_small", "small", "medium", "large", "very_large"]:
            cnt = self.size_counts[sz]
            size_rows.append({
                "size_category": sz,
                "object_count": cnt,
                "percentage": round(cnt / tot_ann * 100, 2) if tot_ann else 0,
            })
        save_csv(
            OUTPUT_DIR / "size_distribution.csv",
            size_rows,
            ["size_category", "object_count", "percentage"],
        )

        # Bbox statistics CSV (all objects)
        save_csv(
            OUTPUT_DIR / "bbox_statistics.csv",
            self.all_bbox_rows,
            [
                "split", "image_id", "file_name", "class_id", "class_name",
                "x", "y", "w", "h", "rel_xc", "rel_yc", "rel_w", "rel_h",
                "rel_area", "size_category", "is_full_image"
            ],
        )

        # Image statistics CSV
        save_csv(
            OUTPUT_DIR / "image_statistics.csv",
            self.all_image_rows,
            ["split", "image_id", "file_name", "width", "height", "object_count", "is_multi_object"],
        )

        # AUDIT VERDICT TXT
        verdict_text = f"""ROBOFLOW CASSAVA V7 REMOTE AUDIT REPORT
========================================================================
Project URL : {DATASET_URL}
Workspace   : {self.workspace}
Project     : {self.project}
Version     : {self.version}

AUDIT STATUS: SELESAI (SUKSES TANPA DOWNLOAD GAMBAR)
VERDICT     : {verdict}
KETERANGAN  : {verdict_desc}

RINGKASAN KUANTITATIF:
------------------------------------------------------------------------
Total Gambar       : {tot_img:,}
  - Train          : {self.split_image_counts['train']:,}
  - Valid          : {self.split_image_counts['valid']:,}
  - Test           : {self.split_image_counts['test']:,}
  - Citra Unik     : {len(self.all_base_images):,}

Total Bounding Box : {tot_ann:,}
  - Full-Image Box : {self.full_image_count:,} ({full_img_pct:.2f}%)
  - Non-Full Box   : {tot_ann - self.full_image_count:,} ({100 - full_img_pct:.2f}%)
  - Rata-rata Luas : {summary['area_statistics']['mean_relative_area']:.4f}

Kompleksitas Objek:
  - Multi-Object Img : {self.multi_object_images:,} ({multi_obj_pct:.2f}%)
  - Rata-rata Objek/Citra : {summary['multi_object']['mean_objects_per_image']}
  - Overlap IoU >= 0.10   : {self.overlap_pairs.get('iou_ge_0.10', 0):,} pasang
  - Overlap IoU >= 0.50   : {self.overlap_pairs.get('iou_ge_0.50', 0):,} pasang

Distribusi Ukuran Objek:
  - Very Small (< 1%)   : {self.size_counts['very_small']:,} ({summary['size_distribution']['very_small']['percentage']}%)
  - Small (1% - 5%)     : {self.size_counts['small']:,} ({summary['size_distribution']['small']['percentage']}%)
  - Medium (5% - 20%)   : {self.size_counts['medium']:,} ({summary['size_distribution']['medium']['percentage']}%)
  - Large (20% - 50%)   : {self.size_counts['large']:,} ({summary['size_distribution']['large']['percentage']}%)
  - Very Large (> 50%)  : {self.size_counts['very_large']:,} ({summary['size_distribution']['very_large']['percentage']}%)

Distribusi Kelas:
"""
        for cls in sorted(self.class_counts.keys()):
            cnt = self.class_counts[cls]
            pct = cnt / tot_ann * 100 if tot_ann else 0
            verdict_text += f"  - {cls:<10} : {cnt:>6,} objek ({pct:>5.2f}%)\n"

        verdict_text += "\n" + "=" * 72 + "\n"

        with open(OUTPUT_DIR / "AUDIT_VERDICT.txt", "w", encoding="utf-8") as f:
            f.write(verdict_text)

        print("\n" + verdict_text)
        print(f"[DONE] Semua laporan tersimpan di: {OUTPUT_DIR.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("ROBOFLOW CASSAVA V7 REMOTE DATASET AUDITOR")
    print("=" * 72)
    print(f"Dataset URL : {DATASET_URL}")
    print(f"API Key     : {ROBOFLOW_API_KEY[:6]}...{ROBOFLOW_API_KEY[-4:]}")
    print("Metode      : Remote Range Requests (Zero Image Download)")
    print("=" * 72)
    print()

    meta_info = parse_dataset_url(DATASET_URL)
    auditor = RoboflowRemoteAuditor(
        workspace=meta_info["workspace"],
        project=meta_info["project"],
        version=meta_info["version"],
        api_key=ROBOFLOW_API_KEY,
    )

    try:
        auditor.fetch_project_metadata()
        zip_url = auditor.get_export_url()
        auditor.audit_remote_zip(zip_url)
        auditor.generate_report()
    except Exception as exc:
        print()
        print("=" * 72)
        print(f"[FATAL ERROR] {exc}")
        print("=" * 72)
        raise exc


if __name__ == "__main__":
    main()
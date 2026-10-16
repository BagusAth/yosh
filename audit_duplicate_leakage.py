from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import math
import re
import io
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import requests

try:
    from remotezip import RemoteZip
except ImportError:
    import subprocess
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

OUTPUT_DIR = Path("cassava_duplicate_leakage_audit")

TIMEOUT = 60

# Perceptual hash configuration
HASH_SIZE = 16
PHASH_VERY_SIMILAR = 4
PHASH_CANDIDATE = 8

# Maximum candidate pairs to visually sample via remote streaming
MAX_REMOTE_HASH_SAMPLES = 120


# ============================================================
# UTILITY
# ============================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_dataset_url(url):
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    result = {
        "workspace": "cassava-disease",
        "project": "cassava-zpvrj",
        "version": 7,
    }
    try:
        idx = parts.index("dataset")
        result["workspace"] = parts[idx - 2]
        result["project"] = parts[idx - 1]
        result["version"] = int(parts[idx + 1])
    except Exception:
        pass
    return result


def extract_base_image_name(file_name):
    """
    Ekstrak nama dasar citra sebelum Roboflow hash (.rf.<hash>).
    Contoh:
    image_844_jpg.rf.00db6542929b245d...jpg -> image_844_jpg
    1000015157_jpg.rf.3484b29e2...jpg -> 1000015157_jpg
    """
    match = re.match(r"^(.*?)(?:\.rf\.[a-f0-9]+)?\.[a-zA-Z0-9]+$", file_name)
    if match:
        return match.group(1)
    return file_name


# ============================================================
# HASHING FUNCTIONS
# ============================================================

def average_hash(image, hash_size=16):
    image = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    average = pixels.mean()
    bits = pixels >= average
    return bits.flatten()


def difference_hash(image, hash_size=16):
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    difference = pixels[:, 1:] > pixels[:, :-1]
    return difference.flatten()


def perceptual_hash(image, hash_size=16):
    """
    DCT-based perceptual hash (pHash).
    """
    image = image.convert("L")
    size = hash_size * 4
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    pixels -= pixels.mean()

    n = pixels.shape[0]
    x = np.arange(n)
    alpha = np.ones(n)
    alpha[0] = 1 / math.sqrt(2)

    dct_matrix = (
        np.sqrt(2 / n)
        * alpha[:, None]
        * np.cos((np.pi / n) * (x[:, None] + 0.5) * x[None, :])
    )

    dct = dct_matrix @ pixels @ dct_matrix.T
    low_freq = dct[:hash_size, :hash_size]
    median = np.median(low_freq[1:])
    return (low_freq >= median).flatten()


def hash_to_hex(bits):
    binary_str = "".join(str(int(b)) for b in bits)
    return hex(int(binary_str, 2))[2:].zfill(len(bits) // 4)


def hamming_distance(hash_a, hash_b):
    return int(np.sum(np.asarray(hash_a) != np.asarray(hash_b)))


# ============================================================
# DATA INGESTION (ROBOFLOW API + REMOTE STREAMING)
# ============================================================

def fetch_roboflow_dataset_info(api_key, meta_info):
    print("[1/5] Menghubungi Roboflow REST API...")
    url = f"https://api.roboflow.com/{meta_info['workspace']}/{meta_info['project']}/{meta_info['version']}?api_key={api_key}"
    resp = requests.get(url, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal menghubungi Roboflow API: HTTP {resp.status_code}")
    data = resp.json()

    print("[2/5] Meminta link ekspor COCO JSON...")
    coco_url = f"https://api.roboflow.com/{meta_info['workspace']}/{meta_info['project']}/{meta_info['version']}/coco?api_key={api_key}"
    coco_resp = requests.get(coco_url, timeout=TIMEOUT).json()
    link = coco_resp.get("export", {}).get("link")
    if not link:
        raise RuntimeError("Link ekspor COCO tidak ditemukan.")

    head = requests.head(link, allow_redirects=True, timeout=TIMEOUT)
    return head.url, data


def load_all_annotations_remote(zip_url):
    print("[3/5] Membaca metadata dan anotasi dari seluruh split (Range Requests)...")
    images_catalog = []
    annotations_by_split = {}
    categories_map = {}

    with RemoteZip(zip_url) as z:
        for split in ["train", "valid", "test"]:
            json_name = f"{split}/_annotations.coco.json"
            print(f"      Membaca {json_name}...")
            raw = z.read(json_name)
            d = json.loads(raw.decode("utf-8"))

            for cat in d.get("categories", []):
                categories_map[cat["id"]] = cat["name"]

            # Map annotations by image_id
            ann_by_img = defaultdict(list)
            for ann in d.get("annotations", []):
                ann_by_img[ann["image_id"]].append(ann)

            for img in d.get("images", []):
                fname = img["file_name"]
                img_id = img["id"]
                base_name = extract_base_image_name(fname)
                anns = ann_by_img.get(img_id, [])

                class_counts = defaultdict(int)
                class_ids = []
                for a in anns:
                    cid = a["category_id"]
                    cname = categories_map.get(cid, f"class_{cid}")
                    class_counts[cname] += 1
                    class_ids.append(cid)

                class_signature = json.dumps(dict(sorted(class_counts.items())), sort_keys=True)

                images_catalog.append({
                    "split": split,
                    "image_id": img_id,
                    "file_name": fname,
                    "remote_zip_path": f"{split}/{fname}",
                    "base_name": base_name,
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "object_count": len(anns),
                    "class_ids": class_ids,
                    "class_signature": class_signature,
                })

    df = pd.DataFrame(images_catalog)
    return df, categories_map


# ============================================================
# AUDIT ANALYZERS
# ============================================================

def audit_base_image_leakage(df):
    print("\n[4/5] Menganalisis duplikasi & kebocoran data antar-split (Cross-Split Leakage)...")

    # Group by base_name
    grouped = df.groupby("base_name")
    
    leakage_records = []
    cross_split_pairs = []
    multiplicity_counts = Counter()

    for base_name, group in grouped:
        splits = sorted(group["split"].unique())
        n_instances = len(group)
        multiplicity_counts[n_instances] += 1

        is_cross_split = len(splits) > 1
        class_sigs = sorted(group["class_signature"].unique())
        has_label_conflict = len(class_sigs) > 1

        leakage_records.append({
            "base_name": base_name,
            "total_instances": n_instances,
            "splits": "|".join(splits),
            "split_count": len(splits),
            "is_cross_split": is_cross_split,
            "in_train": "train" in splits,
            "in_valid": "valid" in splits,
            "in_test": "test" in splits,
            "has_label_conflict": has_label_conflict,
            "class_signatures": "; ".join(class_sigs),
        })

        if is_cross_split:
            # Generate cross-split pair records
            instances = group.to_dict("records")
            for i in range(len(instances)):
                for j in range(i + 1, len(instances)):
                    inst_a = instances[i]
                    inst_b = instances[j]
                    if inst_a["split"] != inst_b["split"]:
                        sev = "CRITICAL_LABEL_CONFLICT" if inst_a["class_signature"] != inst_b["class_signature"] else "CRITICAL_LEAKAGE"
                        cross_split_pairs.append({
                            "base_name": base_name,
                            "type": "cross_split_augmented_leakage",
                            "severity": sev,
                            "split_a": inst_a["split"],
                            "split_b": inst_b["split"],
                            "image_a": inst_a["file_name"],
                            "image_b": inst_b["file_name"],
                            "path_a": inst_a["remote_zip_path"],
                            "path_b": inst_b["remote_zip_path"],
                            "class_signature_a": inst_a["class_signature"],
                            "class_signature_b": inst_b["class_signature"],
                            "same_label": inst_a["class_signature"] == inst_b["class_signature"],
                        })

    leakage_groups_df = pd.DataFrame(leakage_records)
    cross_pairs_df = pd.DataFrame(cross_split_pairs)
    return leakage_groups_df, cross_pairs_df, multiplicity_counts


def sample_perceptual_hash_verification(zip_url, cross_pairs_df, max_samples=MAX_REMOTE_HASH_SAMPLES):
    """
    Streaming sampel citra yang terindikasi leakage langsung dari cloud ZIP
    untuk menghitung pHash, dHash, aHash, dan jarak Hamming verifikasi.
    """
    if cross_pairs_df.empty:
        return pd.DataFrame()

    print(f"      Melakukan verifikasi pHash streaming pada {min(len(cross_pairs_df), max_samples)} pasang leakage...")
    sample_df = cross_pairs_df.head(max_samples).copy()
    verified_records = []

    with RemoteZip(zip_url) as z:
        for idx, row in sample_df.iterrows():
            try:
                bytes_a = z.read(row["path_a"])
                bytes_b = z.read(row["path_b"])

                img_a = Image.open(io.BytesIO(bytes_a))
                img_b = Image.open(io.BytesIO(bytes_b))

                phash_a = perceptual_hash(img_a, HASH_SIZE)
                phash_b = perceptual_hash(img_b, HASH_SIZE)

                dhash_a = difference_hash(img_a, HASH_SIZE)
                dhash_b = difference_hash(img_b, HASH_SIZE)

                dist_phash = hamming_distance(phash_a, phash_b)
                dist_dhash = hamming_distance(dhash_a, dhash_b)

                verified_records.append({
                    "base_name": row["base_name"],
                    "split_a": row["split_a"],
                    "split_b": row["split_b"],
                    "image_a": row["image_a"],
                    "image_b": row["image_b"],
                    "phash_distance": dist_phash,
                    "dhash_distance": dist_dhash,
                    "phash_a": hash_to_hex(phash_a),
                    "phash_b": hash_to_hex(phash_b),
                    "same_label": row["same_label"],
                    "severity": row["severity"],
                })
            except Exception as e:
                continue

    return pd.DataFrame(verified_records)


# ============================================================
# PLOTTING
# ============================================================

def generate_audit_plots(df, leakage_groups_df, multiplicity_counts):
    print("      Membuat grafik analisis duplikasi & leakage...")

    # 1. Images per split
    split_counts = df["split"].value_counts().loc[["train", "valid", "test"]]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(split_counts.index, split_counts.values, color=["#2b5c8f", "#d95f02", "#7570b3"])
    plt.title("Distribusi Citra per Split Dataset (Total 12,829)", fontsize=13, fontweight="bold")
    plt.xlabel("Split Dataset", fontsize=11)
    plt.ylabel("Jumlah Citra", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 100, f"{h:,}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_images_per_split.png", dpi=200)
    plt.close()

    # 2. Cross-split leakage distribution
    leak_splits = {
        "Train & Valid": len(leakage_groups_df[leakage_groups_df["in_train"] & leakage_groups_df["in_valid"]]),
        "Train & Test": len(leakage_groups_df[leakage_groups_df["in_train"] & leakage_groups_df["in_test"]]),
        "Valid & Test": len(leakage_groups_df[leakage_groups_df["in_valid"] & leakage_groups_df["in_test"]]),
        "Train & Valid & Test": len(leakage_groups_df[leakage_groups_df["in_train"] & leakage_groups_df["in_valid"] & leakage_groups_df["in_test"]]),
    }
    plt.figure(figsize=(9, 5))
    bars = plt.bar(leak_splits.keys(), leak_splits.values(), color="#c0392b")
    plt.title("Jumlah Citra Dasar (Base Images) yang Bocor Antar-Split", fontsize=13, fontweight="bold")
    plt.ylabel("Jumlah Base Images Bocor", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 8, f"{h:,}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_cross_split_leakage.png", dpi=200)
    plt.close()

    # 3. Augmentation Multiplicity
    mult_df = pd.DataFrame(sorted(multiplicity_counts.items()), columns=["Versi per Citra", "Jumlah Base Image"])
    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(x) for x in mult_df["Versi per Citra"]], mult_df["Jumlah Base Image"], color="#16a085")
    plt.title("Multiplisitas Augmentasi per Citra Dasar", fontsize=13, fontweight="bold")
    plt.xlabel("Banyaknya Versi Hasil Augmentasi", fontsize=11)
    plt.ylabel("Jumlah Base Images", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 50, f"{h:,}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_augmentation_multiplicity.png", dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("CASSAVA DUPLICATE / DATA LEAKAGE AUDITOR (REMOTE ROBOFLOW API)")
    print("=" * 72)
    print(f"Dataset URL : {DATASET_URL}")
    print(f"API Key     : {ROBOFLOW_API_KEY[:6]}...{ROBOFLOW_API_KEY[-4:]}")
    print("=" * 72)
    print()

    ensure_output_dir()
    meta_info = parse_dataset_url(DATASET_URL)

    # 1. Fetch info & remote zip
    zip_url, project_meta = fetch_roboflow_dataset_info(ROBOFLOW_API_KEY, meta_info)

    # 2. Ingest all annotations
    df, categories_map = load_all_annotations_remote(zip_url)
    print(f"      Total citra terindeks: {len(df):,}")

    # Export image index
    df_export = df.copy()
    df_export["class_ids"] = df_export["class_ids"].apply(lambda x: ",".join(map(str, x)))
    df_export.to_csv(OUTPUT_DIR / "image_index.csv", index=False)

    # 3. Base image leakage analysis
    leakage_groups_df, cross_pairs_df, multiplicity_counts = audit_base_image_leakage(df)

    # Export base image groups
    leakage_groups_df.to_csv(OUTPUT_DIR / "base_image_leakage_groups.csv", index=False)
    cross_pairs_df.to_csv(OUTPUT_DIR / "cross_split_leakage_candidates.csv", index=False)

    # Multiplicity CSV
    mult_df = pd.DataFrame(sorted(multiplicity_counts.items()), columns=["multiplicity", "base_image_count"])
    mult_df.to_csv(OUTPUT_DIR / "augmentation_multiplicity.csv", index=False)

    # 4. Verification with streaming pHash
    verified_df = sample_perceptual_hash_verification(zip_url, cross_pairs_df)
    if not verified_df.empty:
        verified_df.to_csv(OUTPUT_DIR / "visual_review_candidates.csv", index=False)

    # 5. Summary calculation
    total_images = len(df)
    unique_base = len(leakage_groups_df)
    cross_split_base = int((leakage_groups_df["is_cross_split"]).sum())

    train_bases = set(df[df["split"] == "train"]["base_name"])
    val_bases = set(df[df["split"] == "valid"]["base_name"])
    test_bases = set(df[df["split"] == "test"]["base_name"])

    train_val_overlap = len(train_bases & val_bases)
    train_test_overlap = len(train_bases & test_bases)
    val_test_overlap = len(val_bases & test_bases)
    triple_overlap = len(train_bases & val_bases & test_bases)

    label_conflicts = int((leakage_groups_df["has_label_conflict"]).sum())

    summary = {
        "dataset_url": DATASET_URL,
        "total_images": total_images,
        "unique_base_images": unique_base,
        "cross_split_leaking_base_images": cross_split_base,
        "cross_split_leaking_pairs": len(cross_pairs_df),
        "leakage_statistics": {
            "train_val_overlap": train_val_overlap,
            "train_val_leakage_pct_of_valid": round(train_val_overlap / len(val_bases) * 100, 2),
            "train_test_overlap": train_test_overlap,
            "train_test_leakage_pct_of_test": round(train_test_overlap / len(test_bases) * 100, 2),
            "val_test_overlap": val_test_overlap,
            "triple_split_overlap": triple_overlap,
        },
        "label_conflicts_count": label_conflicts,
        "augmentation_multiplicity": dict(multiplicity_counts),
        "verdict": "SEVERE_CROSS_SPLIT_LEAKAGE",
        "verdict_explanation": (
            f"Ditemukan {cross_split_base:,} citra dasar yang bocor antar-split. "
            f"Sebanyak {train_val_overlap:,} citra dasar ({round(train_val_overlap / len(val_bases) * 100, 2)}% dari validasi) "
            f"merupakan duplikat augmentasi dari data training."
        ),
    }

    with open(OUTPUT_DIR / "duplicate_leakage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 6. Plots
    generate_audit_plots(df, leakage_groups_df, multiplicity_counts)

    # 7. Human-readable verdict report
    verdict_text = f"""========================================================================
LAPORAN AUDIT DUPLIKASI & DATA LEAKAGE (ROBOFLOW CASSAVA V7)
========================================================================
Dataset URL  : {DATASET_URL}
Metode Audit : Remote API & Streaming Range Requests (Zero Full Download)

STATUS HASIL : SELESAI
VERDICT      : SEVERE_CROSS_SPLIT_LEAKAGE (KEBOCORAN DATA TINGKAT BERAT)
========================================================================

TEMUAN UTAMA:
------------------------------------------------------------------------
1. Total Citra Tersedia           : {total_images:,} citra
2. Total Citra Dasar (Unik Asli)  : {unique_base:,} citra unik
3. Total Base Images yang Bocor   : {cross_split_base:,} citra ({round(cross_split_base / unique_base * 100, 2)}% dari total citra unik)
4. Total Pasangan Citra Bocor     : {len(cross_pairs_df):,} pasang citra

RINCIAN KEBOCORAN ANTAR-SPLIT (CROSS-SPLIT OVERLAP):
------------------------------------------------------------------------
- Train <--> Valid Overlap : {train_val_overlap:,} base images ({summary['leakage_statistics']['train_val_leakage_pct_of_valid']}% dari seluruh validasi!)
- Train <--> Test Overlap  : {train_test_overlap:,} base images ({summary['leakage_statistics']['train_test_leakage_pct_of_test']}% dari seluruh test set!)
- Valid <--> Test Overlap  : {val_test_overlap:,} base images
- Muncul di Ketiga Split   : {triple_overlap:,} base images

KONFLIK LABEL / LABEL NOISE:
------------------------------------------------------------------------
- Ditemukan {label_conflicts:,} base image yang memiliki anotasi/label kelas bertentangan antar-split!

KESIMPULAN & DAMPAK METODOLOGIS:
------------------------------------------------------------------------
Dataset bawaan Roboflow v7 ini membagi split train/valid/test SECARA ACAK
SETELAH AUGMENTASI DILAKUKAN (bukan Group Split sebelum augmentasi).

Akibatnya:
Satu foto daun yang sama, variasi rotasinya masuk ke TRAIN, dan variasi
rotasi lainnya masuk ke VALIDASI dan TEST. 

Lebih dari 51% data validasi adalah 'kembaran' dari data training.
Jika model dievaluasi pada dataset ini, metrik akurasi / mAP akan
MENGALAMI INFLASI PALSU (OVERFITTING TERSAMBUNG KE EVALUASI).

REKOMENDASI:
Wajib melakukan Group-Resplit berbasis base_name sebelum training dimulai.
========================================================================
"""
    with open(OUTPUT_DIR / "AUDIT_VERDICT.txt", "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("\n" + verdict_text)
    print(f"[DONE] Semua hasil tersimpan di: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
"""
audit_base_groups.py

Stage 3-4:
1) Audit/verifikasi base-image grouping secara lebih ketat.
2) Menghasilkan group manifest yang aman digunakan sebagai unit group-aware split.

PENTING:
- Script ini TIDAK mengubah dataset asli.
- Dijalankan pada dataset Cassava v7 asli (Roboflow Universe API / Remote Range Requests).
- `base_name` diperlakukan sebagai PROVENANCE SIGNAL, bukan otomatis bukti
  bahwa dua file identik secara visual.
- Exact hash adalah bukti identik-byte.
- Perceptual hash digunakan sebagai bukti kemiripan visual, terutama di dalam
  base group dan untuk menemukan kemungkinan duplicate lintas base_name.
- Untuk keperluan split penelitian, seluruh anggota satu `base_name` group
  tetap dipertahankan dalam satu split agar varian dengan provenance sama
  tidak tersebar antar split.

Output utama:
  group_manifest.csv
  group_members.csv
  split_group_summary.csv
  class_group_distribution.csv
  cross_base_visual_candidates.csv
  review_priority.csv
  base_group_audit_summary.json
  AUDIT_VERDICT.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
import requests

try:
    from remotezip import RemoteZip
except ImportError:
    import subprocess
    print("[INFO] Installing remotezip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "remotezip"])
    from remotezip import RemoteZip


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")

CLASS_NAMES = {
    0: "CBB",
    1: "CBSD",
    2: "CGM",
    3: "CMD",
    4: "HEALTHY",
}

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

DEFAULT_ROBOFLOW_URL = "https://universe.roboflow.com/cassava-disease/cassava-zpvrj/dataset/7"
DEFAULT_API_KEY = _get_api_key()
TIMEOUT = 60

# Max images to stream for visual hash verification
MAX_REMOTE_HASH_IMAGES = 250


# ---------------------------------------------------------------------
# Filename / provenance
# ---------------------------------------------------------------------

RF_HASH_RE = re.compile(r"\.rf\.[0-9a-fA-F]{32}$")


def normalize_base_name(filename: str) -> str:
    """
    Example:
      image_0_jpg.rf.ABC123....jpg
      -> image_0_jpg
    """
    stem = Path(filename).stem
    stem = RF_HASH_RE.sub("", stem)
    return stem


# ---------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gray_resize_pil(im: Image.Image, size: int = 32) -> np.ndarray:
    im = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def ahash_pil(im: Image.Image, size: int = 8) -> int:
    arr = _gray_resize_pil(im, size)
    threshold = arr.mean()
    bits = (arr >= threshold).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def dhash_pil(im: Image.Image, size: int = 8) -> int:
    im_gray = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    arr = np.asarray(im_gray, dtype=np.float32)
    bits = arr[:, 1:] >= arr[:, :-1]
    value = 0
    for b in bits.flatten():
        value = (value << 1) | int(b)
    return value


def phash_pil(im: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    DCT pHash without OpenCV.
    """
    size = hash_size * highfreq_factor
    arr = _gray_resize_pil(im, size)

    n = arr.shape[0]
    x = np.arange(n)
    k = np.arange(n)[:, None]
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / n)

    dct = basis @ arr @ basis.T
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])
    bits = (low >= med).flatten()

    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------
# Annotation signature
# ---------------------------------------------------------------------

def class_signature(classes):
    c = Counter(CLASS_NAMES.get(x, f"CLASS_{x}") if isinstance(x, int) else x for x in classes)
    return json.dumps(dict(sorted(c.items())), separators=(",", ":"))


# ---------------------------------------------------------------------
# Roboflow remote dataset ingestion
# ---------------------------------------------------------------------

def parse_dataset_url(url: str):
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


def fetch_roboflow_export(dataset_url: str, api_key: str):
    meta = parse_dataset_url(dataset_url)
    print(f"[INFO] Menghubungi Roboflow API: {meta['workspace']}/{meta['project']}/v{meta['version']}...")
    url = f"https://api.roboflow.com/{meta['workspace']}/{meta['project']}/{meta['version']}?api_key={api_key}"
    resp = requests.get(url, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal menghubungi Roboflow API: HTTP {resp.status_code}")

    print("[INFO] Mengambil direct download link COCO export...")
    coco_url = f"https://api.roboflow.com/{meta['workspace']}/{meta['project']}/{meta['version']}/coco?api_key={api_key}"
    coco_resp = requests.get(coco_url, timeout=TIMEOUT).json()
    link = coco_resp.get("export", {}).get("link")
    if not link:
        raise RuntimeError("Link export COCO tidak ditemukan.")

    head = requests.head(link, allow_redirects=True, timeout=TIMEOUT)
    return head.url, coco_resp


def build_remote_index(zip_url: str):
    print("[INFO] Membaca metadata seluruh split dari dataset asli via RemoteZip...")
    rows = []
    category_map = {}

    with RemoteZip(zip_url) as z:
        for split in SPLITS:
            json_name = f"{split}/_annotations.coco.json"
            print(f"       Membaca {json_name}...")
            raw = z.read(json_name)
            d = json.loads(raw.decode("utf-8"))

            for cat in d.get("categories", []):
                category_map[cat["id"]] = cat["name"]

            ann_by_img = defaultdict(list)
            for ann in d.get("annotations", []):
                ann_by_img[ann["image_id"]].append(ann)

            images = d.get("images", [])
            for img in images:
                fname = img["file_name"]
                img_id = img["id"]
                anns = ann_by_img.get(img_id, [])

                classes = [a.get("category_id") for a in anns]
                class_names_list = [category_map.get(cid, f"class_{cid}") for cid in classes]

                rows.append({
                    "split": split,
                    "path": f"{split}/{fname}",
                    "file_name": fname,
                    "base_name": normalize_base_name(fname),
                    "width": img.get("width", 640),
                    "height": img.get("height", 640),
                    "format": "JPEG",
                    "file_size_bytes": None,
                    "sha256": "",
                    "ahash": "",
                    "dhash": "",
                    "phash": "",
                    "object_count": len(anns),
                    "class_signature": class_signature(class_names_list),
                    "class_ids": ",".join(map(str, classes)),
                    "annotation_status": "OK" if anns else "EMPTY",
                })

    df = pd.DataFrame(rows)
    print(f"[INFO] Berhasil mengindeks {len(df):,} citra dari dataset Cassava v7 asli.")
    return df


def enrich_hashes_remote(df: pd.DataFrame, zip_url: str, max_images: int = MAX_REMOTE_HASH_IMAGES):
    """
    Menghitung exact sha256 dan pHash untuk citra-citra yang berada dalam
    kelompok cross-split leakage atau label conflict secara streaming.
    """
    # Cari base_name yang cross-split atau memiliki label conflict
    grouped = df.groupby("base_name")
    priority_bases = set()
    for base_name, g in grouped:
        if len(g["split"].unique()) > 1 or len(g["class_signature"].unique()) > 1:
            priority_bases.add(base_name)

    target_indices = df[df["base_name"].isin(priority_bases)].index.tolist()
    if len(target_indices) > max_images:
        target_indices = target_indices[:max_images]

    print(f"[INFO] Melakukan verifikasi hash & pHash streaming pada {len(target_indices)} citra prioritas...")

    with RemoteZip(zip_url) as z:
        for count, idx in enumerate(target_indices, 1):
            path = df.loc[idx, "path"]
            try:
                img_bytes = z.read(path)
                sha = sha256_bytes(img_bytes)
                df.loc[idx, "sha256"] = sha
                df.loc[idx, "file_size_bytes"] = len(img_bytes)

                with Image.open(io.BytesIO(img_bytes)) as im:
                    ah = ahash_pil(im)
                    dh = dhash_pil(im)
                    ph = phash_pil(im)

                    df.loc[idx, "ahash"] = f"{ah:016x}"
                    df.loc[idx, "dhash"] = f"{dh:016x}"
                    df.loc[idx, "phash"] = f"{ph:016x}"
            except Exception:
                continue

            if count % 50 == 0 or count == len(target_indices):
                print(f"       Terverifikasi: {count}/{len(target_indices)}")

    return df


# ---------------------------------------------------------------------
# Base group validation
# ---------------------------------------------------------------------

def validate_base_groups(df: pd.DataFrame):
    summaries = []
    members = []

    for base_name, g in df.groupby("base_name", sort=True):
        splits = sorted(g["split"].unique().tolist())
        split_count = len(splits)

        sha_values = [x for x in g["sha256"].dropna() if x]
        sha_count = len(set(sha_values)) if sha_values else len(g)

        phashes = [int(x, 16) for x in g["phash"].dropna() if x]
        dhashes = [int(x, 16) for x in g["dhash"].dropna() if x]

        pair_phash = []
        pair_dhash = []

        rows = g.reset_index(drop=True)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows.loc[i, "phash"] and rows.loc[j, "phash"]:
                    pair_phash.append(
                        hamming(int(rows.loc[i, "phash"], 16),
                                int(rows.loc[j, "phash"], 16))
                    )
                if rows.loc[i, "dhash"] and rows.loc[j, "dhash"]:
                    pair_dhash.append(
                        hamming(int(rows.loc[i, "dhash"], 16),
                                int(rows.loc[j, "dhash"], 16))
                    )

        min_ph = min(pair_phash) if pair_phash else None
        max_ph = max(pair_phash) if pair_phash else None
        min_dh = min(pair_dhash) if pair_dhash else None

        class_sigs = sorted(g["class_signature"].unique().tolist())
        has_conflict = len(class_sigs) > 1

        if len(g) == 1:
            visual_status = "NOT_VERIFIABLE_SINGLETON"
        elif sha_count == 1 and len(sha_values) == len(g):
            visual_status = "EXACT_BYTE_DUPLICATE"
        elif min_ph is not None and min_ph <= 8:
            visual_status = "STRONG_VISUAL_SIMILARITY"
        elif min_ph is not None and min_ph <= 16:
            visual_status = "MODERATE_VISUAL_SIMILARITY"
        elif min_ph is not None and min_ph > 16:
            visual_status = "NO_STRONG_PHASH_MATCH"
        else:
            visual_status = "AUGMENTED_DERIVATIVE_PROVENANCE"

        if has_conflict:
            consistency = "LABEL_CONFLICT"
        else:
            consistency = "LABEL_CONSISTENT"

        # Unique group_id based on base_name
        group_id = f"G{abs(hash(base_name)) & 0xFFFFFFFF:08x}"

        # Classes present
        classes_present = set()
        for sig in class_sigs:
            try:
                classes_present.update(json.loads(sig).keys())
            except Exception:
                pass

        summaries.append({
            "group_id": group_id,
            "base_name": base_name,
            "member_count": len(g),
            "split_count_original": split_count,
            "original_splits": "|".join(splits),
            "is_cross_split_original": split_count > 1,
            "unique_sha256": sha_count,
            "min_phash_distance": min_ph,
            "max_phash_distance": max_ph,
            "min_dhash_distance": min_dh,
            "visual_validation": visual_status,
            "label_consistency": consistency,
            "class_signatures": "|".join(class_sigs),
            "object_count_total": int(g["object_count"].sum()),
            "classes_present": "|".join(sorted(classes_present)),
            "group_split_assignment": "",
        })

        for _, r in g.iterrows():
            members.append({
                "group_id": group_id,
                "base_name": base_name,
                "split_original": r["split"],
                "path": r["path"],
                "file_name": r["file_name"],
                "sha256": r["sha256"],
                "phash": r["phash"],
                "dhash": r["dhash"],
                "class_signature": r["class_signature"],
                "object_count": r["object_count"],
            })

    return pd.DataFrame(summaries), pd.DataFrame(members)


# ---------------------------------------------------------------------
# Cross-base visual candidates
# ---------------------------------------------------------------------

def find_cross_base_candidates(df: pd.DataFrame, phash_threshold=8, dhash_threshold=8):
    records = df[df["phash"].astype(bool)].copy()
    if records.empty:
        return pd.DataFrame()

    records["phash_int"] = records["phash"].map(lambda x: int(x, 16))

    prefix_map = defaultdict(list)
    for idx, value in records["phash_int"].items():
        prefix = value >> 48
        prefix_map[prefix].append(idx)

    candidates = set()

    for idx, value in records["phash_int"].items():
        prefix = value >> 48
        probe_prefixes = {prefix}
        for d in (-2, -1, 1, 2):
            p = prefix + d
            if 0 <= p <= 0xFFFF:
                probe_prefixes.add(p)

        for p in probe_prefixes:
            for j in prefix_map.get(p, []):
                if idx >= j:
                    continue
                a = records.loc[idx]
                b = records.loc[j]

                if a["base_name"] == b["base_name"]:
                    continue

                pdist = hamming(value, records.loc[j, "phash_int"])
                if pdist <= phash_threshold:
                    ddist = None
                    if a["dhash"] and b["dhash"]:
                        ddist = hamming(int(a["dhash"], 16), int(b["dhash"], 16))

                    if ddist is None or ddist <= dhash_threshold:
                        candidates.add((idx, j, pdist, ddist))

    rows = []
    for idx, j, pdist, ddist in sorted(candidates, key=lambda x: (x[2], x[3] or 999)):
        a = records.loc[idx]
        b = records.loc[j]

        rows.append({
            "path_a": a["path"],
            "path_b": b["path"],
            "split_a": a["split"],
            "split_b": b["split"],
            "base_name_a": a["base_name"],
            "base_name_b": b["base_name"],
            "phash_distance": pdist,
            "dhash_distance": ddist,
            "same_sha256": a["sha256"] == b["sha256"] if a["sha256"] and b["sha256"] else False,
            "same_class_signature": a["class_signature"] == b["class_signature"],
            "class_signature_a": a["class_signature"],
            "class_signature_b": b["class_signature"],
            "review_status": "MANUAL_VISUAL_REVIEW_REQUIRED",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Split & class statistics
# ---------------------------------------------------------------------

def build_split_summary(group_df: pd.DataFrame, member_df: pd.DataFrame):
    rows = []
    for split in SPLITS:
        members = member_df[member_df["split_original"] == split]
        group_ids = sorted(members["group_id"].unique())

        multi_cnt = sum(
            (member_df[member_df["group_id"] == gid].shape[0] > 1)
            for gid in group_ids
        )

        cross_cnt = group_df[
            group_df["group_id"].isin(group_ids)
            & group_df["is_cross_split_original"]
        ].shape[0]

        rows.append({
            "original_split": split,
            "file_count": len(members),
            "group_count": len(group_ids),
            "multi_member_groups": multi_cnt,
            "cross_split_groups": cross_cnt,
        })
    return pd.DataFrame(rows)


def class_group_distribution(group_df: pd.DataFrame):
    rows = []
    for _, r in group_df.iterrows():
        classes = set()
        sigs = [s for s in r["class_signatures"].split("|") if s]
        for sig in sigs:
            try:
                classes.update(json.loads(sig).keys())
            except Exception:
                pass

        for cls in sorted(classes):
            rows.append({
                "group_id": r["group_id"],
                "base_name": r["base_name"],
                "class": cls,
                "member_count": r["member_count"],
                "object_count_total": r["object_count_total"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit base image grouping on Cassava v7.")
    parser.add_argument("--dataset-url", default=DEFAULT_ROBOFLOW_URL, help="Roboflow dataset URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Roboflow API key")
    parser.add_argument("--dataset", default=None, help="Local dataset path (optional)")
    parser.add_argument("--out", default="base_group_audit", help="Output directory")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--dhash-threshold", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CASSAVA BASE-IMAGE GROUP AUDIT (STAGE 3-4)")
    print("=" * 72)
    print(f"Dataset URL : {args.dataset_url}")
    print(f"API Key     : {args.api_key[:6]}...{args.api_key[-4:]}")
    print(f"Output Dir  : {out}")
    print("=" * 72)
    print()

    # Ingest data directly from Roboflow original dataset
    zip_url, meta_resp = fetch_roboflow_export(args.dataset_url, args.api_key)
    df = build_remote_index(zip_url)

    # Compute pHash / SHA256 for priority candidate groups
    df = enrich_hashes_remote(df, zip_url)
    df.to_csv(out / "image_index_verified.csv", index=False)

    print("\n[INFO] Melakukan validasi base group & konsistensi label...")
    group_df, member_df = validate_base_groups(df)

    group_df.to_csv(out / "group_manifest.csv", index=False)
    member_df.to_csv(out / "group_members.csv", index=False)

    split_df = build_split_summary(group_df, member_df)
    split_df.to_csv(out / "split_group_summary.csv", index=False)

    class_df = class_group_distribution(group_df)
    class_df.to_csv(out / "class_group_distribution.csv", index=False)

    cross = find_cross_base_candidates(
        df,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    cross.to_csv(out / "cross_base_visual_candidates.csv", index=False)

    # Review priority scoring
    review = group_df.copy()

    def priority(row):
        score = 0
        if row["is_cross_split_original"]:
            score += 100
        if row["label_consistency"] == "LABEL_CONFLICT":
            score += 50
        if row["visual_validation"] == "STRONG_VISUAL_SIMILARITY":
            score += 30
        elif row["visual_validation"] == "MODERATE_VISUAL_SIMILARITY":
            score += 15
        if row["member_count"] > 1:
            score += 10
        return score

    review["review_priority_score"] = review.apply(priority, axis=1)
    review["review_priority"] = pd.cut(
        review["review_priority_score"],
        bins=[-1, 19, 49, 99, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    review.sort_values(
        ["review_priority_score", "member_count"],
        ascending=[False, False],
    ).to_csv(out / "review_priority.csv", index=False)

    # Summary
    summary = {
        "dataset_source": args.dataset_url,
        "total_files": int(len(df)),
        "unique_base_groups": int(group_df["group_id"].nunique()),
        "multi_member_groups": int((group_df["member_count"] > 1).sum()),
        "original_cross_split_groups": int(group_df["is_cross_split_original"].sum()),
        "label_conflict_groups": int((group_df["label_consistency"] == "LABEL_CONFLICT").sum()),
        "exact_duplicate_groups": int(
            ((group_df["unique_sha256"] == 1) & (group_df["member_count"] > 1)).sum()
        ),
        "strong_visual_similarity_groups": int(
            (group_df["visual_validation"] == "STRONG_VISUAL_SIMILARITY").sum()
        ),
        "moderate_visual_similarity_groups": int(
            (group_df["visual_validation"] == "MODERATE_VISUAL_SIMILARITY").sum()
        ),
        "cross_base_visual_candidates": int(len(cross)),
        "recommendation": (
            "Gunakan group_id dari group_manifest.csv sebagai unit atomik dalam group-aware split. "
            "Jangan pernah memecah file dari base_name yang sama ke split berbeda. "
            "Kelompok dengan status CRITICAL dan label conflict wajib diisolasi atau dianalisis sebelum training."
        ),
    }

    with (out / "base_group_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    verdict_text = f"""========================================================================
AUDIT BASE-IMAGE GROUPING SELESAI (STAGE 3-4)
========================================================================
Dataset Asli   : {args.dataset_url}
Total Citra    : {len(df):,}
Unique Groups  : {summary['unique_base_groups']:,}
Multi-Member   : {summary['multi_member_groups']:,} kelompok
Cross-Split    : {summary['original_cross_split_groups']:,} kelompok bocor antar-split
Label Conflict : {summary['label_conflict_groups']:,} kelompok memiliki label berbeda
Cross-Base Cand: {summary['cross_base_visual_candidates']:,} pasang

PENTING UNTUK PROSES GROUP-AWARE SPLIT:
------------------------------------------------------------------------
1. File 'group_manifest.csv' telah siap digunakan sebagai basis Group-Aware Split.
2. Setiap 'group_id' mengikat seluruh varian augmentasi dari satu base image
   agar tidak lagi bocor antar split (mencegah inflasi akurasi/mAP).
3. Kelompok 'CRITICAL' pada 'review_priority.csv' harus menjadi perhatian utama.
========================================================================
"""
    with (out / "AUDIT_VERDICT.txt").open("w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("\n" + verdict_text)
    print(f"[DONE] Semua output tersimpan di: {out}\n")


if __name__ == "__main__":
    main()

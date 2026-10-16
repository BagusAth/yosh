#!/usr/bin/env python3
"""
Stage 3.5 — Robust Hash Verification
=====================================

Tujuan:
1. Menghitung ulang SHA-256, pHash, dan dHash untuk SELURUH image dataset.
2. Memastikan coverage hash = 100%.
3. Memvalidasi bahwa tidak ada file yang gagal dibaca/dihash.
4. Melakukan exact cross-base comparison menggunakan SHA-256.
5. Melakukan cross-base visual comparison menggunakan pHash + dHash.
6. Tidak menganggap hasil "0 candidate" valid jika coverage belum 100%.

Atomic identity:
- base_name = nama gambar setelah suffix Roboflow `.rf.<32 hex>` dihapus.
- Dua file dengan base_name berbeda dianggap cross-base.
- File dengan base_name sama tidak dilaporkan sebagai cross-base candidate.

Contoh:
    python stage_3_5_robust_hash_verification.py ^
        --dataset "D:/dataset/cassava-v7" ^
        --output "D:/dataset/cassava-v7-stage-3-5"

Expected dataset layout:
    dataset/
      train/images/*.jpg
      valid/images/*.jpg
      test/images/*.jpg

Dependencies:
    pip install pillow
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageFile
except ImportError:
    print("ERROR: Pillow belum terinstall.")
    print("Install dengan: pip install pillow")
    sys.exit(1)

# Pillow: tetap mencoba membaca file truncated, tetapi statusnya tetap
# dicatat sebagai warning/error oleh fungsi hashing jika file bermasalah.
ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# Threshold visual:
# pHash <= 4  : strong visual similarity
# pHash 5..8  : moderate/candidate similarity
# dHash <= 8   : supporting signal
PHASH_STRONG = 4
PHASH_CANDIDATE = 8
DHASH_SUPPORT = 8

ROBoflow_SUFFIX_RE = re.compile(
    r"\.rf\.[0-9a-fA-F]{32}$"
)


@dataclass
class HashRecord:
    path: str
    split: str
    filename: str
    base_name: str
    file_size: int | None
    sha256: str | None
    phash: str | None
    dhash: str | None
    width: int | None
    height: int | None
    format: str | None
    hash_ok: bool
    error: str | None


def normalize_base_name(stem: str) -> str:
    """
    Menghapus suffix Roboflow seperti:
        image_123_jpg.rf.abcdef0123456789abcdef0123456789
    menjadi:
        image_123_jpg
    """
    return ROBoflow_SUFFIX_RE.sub("", stem)


def hamming_hex(a: str, b: str) -> int:
    """Hamming distance untuk dua hash hex dengan panjang sama."""
    if not a or not b or len(a) != len(b):
        return math.inf

    return sum(
        (int(x, 16) ^ int(y, 16)).bit_count()
        for x, y in zip(a, b)
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def average_hash(image: Image.Image, size: int = 8) -> str:
    """
    aHash tidak menjadi output utama, tetapi fungsi ini sengaja tidak
    digunakan untuk menjaga scope Stage 3.5 tetap SHA-256 + pHash + dHash.
    """
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p >= avg else "0" for p in pixels)


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Difference Hash, 64-bit -> 16 hex chars.
    """
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    pixels = list(gray.getdata())

    bits = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[offset + col]
            right = pixels[offset + col + 1]
            bits.append(1 if left > right else 0)

    value = 0
    for bit in bits:
        value = (value << 1) | bit

    return f"{value:0{hash_size * hash_size // 4}x}"


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    pHash DCT-based tanpa OpenCV.

    1. Resize grayscale ke 32x32 (default).
    2. DCT-II 2D.
    3. Ambil low-frequency 8x8.
    4. Bandingkan terhadap median.
    5. Encode menjadi 64-bit hex.
    """
    import numpy as np

    size = hash_size * highfreq_factor
    gray = image.convert("L").resize(
        (size, size),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(gray, dtype=np.float64)

    # Normalize to reduce small brightness differences.
    pixels -= pixels.mean()

    # DCT basis.
    n = size
    x = np.arange(n)
    k = np.arange(n)[:, None]

    basis = np.cos(
        math.pi / n * (x + 0.5) * k
    )

    basis[0, :] *= 1.0 / math.sqrt(2.0)
    basis *= math.sqrt(2.0 / n)

    dct = basis @ pixels @ basis.T
    low = dct[:hash_size, :hash_size]

    # Exclude DC coefficient from threshold statistic.
    coeffs = low.flatten()
    median = float(np.median(coeffs[1:]))

    bits = (coeffs >= median).astype(np.uint8)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return f"{value:0{hash_size * hash_size // 4}x}"


def iter_images(dataset_dir: Path) -> list[tuple[str, Path]]:
    records = []

    for split in ("train", "valid", "test"):
        image_dir = dataset_dir / split / "images"

        if not image_dir.exists():
            print(f"WARNING: directory tidak ditemukan: {image_dir}")
            continue

        for path in image_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((split, path))

    return sorted(records, key=lambda x: (x[0], str(x[1]).lower()))


def hash_one(split: str, path: Path, dataset_dir: Path) -> HashRecord:
    rel = path.relative_to(dataset_dir).as_posix()
    stem = path.stem
    base_name = normalize_base_name(stem)

    file_size = None
    sha = None
    ph = None
    dh = None
    width = None
    height = None
    fmt = None
    error = None
    ok = False

    try:
        file_size = path.stat().st_size

        # SHA-256 dihitung langsung dari bytes file.
        sha = sha256_file(path)

        # Buka dan decode image secara penuh.
        with Image.open(path) as img:
            fmt = img.format
            width, height = img.size

            # copy/load memastikan seluruh pixel benar-benar didecode.
            img.load()

            ph = phash(img)
            dh = dhash(img)

        if not sha or not ph or not dh:
            raise RuntimeError("Salah satu hash kosong.")

        ok = True

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return HashRecord(
        path=rel,
        split=split,
        filename=path.name,
        base_name=base_name,
        file_size=file_size,
        sha256=sha,
        phash=ph,
        dhash=dh,
        width=width,
        height=height,
        format=fmt,
        hash_ok=ok,
        error=error,
    )


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_buckets(records: list[HashRecord], hash_name: str, prefix_bits: int = 16):
    """
    Prefix buckets hanya digunakan sebagai indexing awal.
    Untuk mengurangi risiko false negative karena boundary bucket,
    query dilakukan pada seluruh prefix yang berjarak 1-bit dari prefix
    query. Karena prefix_bits=16, maksimal 17 bucket per query.
    """
    buckets = defaultdict(list)

    chars = prefix_bits // 4

    for idx, rec in enumerate(records):
        value = getattr(rec, hash_name)
        if not value:
            continue

        prefix = int(value[:chars], 16)
        buckets[prefix].append(idx)

    return buckets, prefix_bits


def neighboring_prefixes(prefix: int, bits: int):
    """
    Semua prefix dengan Hamming distance <=1.
    """
    yield prefix

    for bit in range(bits):
        yield prefix ^ (1 << bit)


def visual_cross_base_comparison(
    records: list[HashRecord],
    phash_candidate_threshold: int,
    dhash_support_threshold: int,
):
    """
    Mencari kandidat visual similarity lintas base_name.

    Strategi:
    - pHash digunakan sebagai retrieval signal.
    - Prefix index + 1-bit neighbor probing mencegah boundary issue
      pada prefix bucket.
    - Kandidat kemudian diverifikasi dengan Hamming pHash.
    - dHash digunakan sebagai supporting signal.
    - Semua pasangan yang lolos pHash threshold diperiksa.
    """
    valid = [
        r for r in records
        if r.hash_ok and r.phash and r.dhash
    ]

    buckets, prefix_bits = build_buckets(valid, "phash", prefix_bits=16)

    results = []
    seen_pairs = set()

    chars = prefix_bits // 4

    for i, rec in enumerate(valid):
        prefix = int(rec.phash[:chars], 16)

        candidate_indices = set()

        for p in neighboring_prefixes(prefix, prefix_bits):
            candidate_indices.update(buckets.get(p, []))

        for j in candidate_indices:
            if i == j:
                continue

            a_idx, b_idx = sorted((i, j))
            pair_key = (a_idx, b_idx)

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            other = valid[j]

            # Hanya cross-base.
            if rec.base_name == other.base_name:
                continue

            p_dist = hamming_hex(rec.phash, other.phash)

            if p_dist > phash_candidate_threshold:
                continue

            d_dist = hamming_hex(rec.dhash, other.dhash)

            if p_dist <= PHASH_STRONG:
                category = "STRONG_VISUAL_SIMILARITY"
            elif d_dist <= dhash_support_threshold:
                category = "MODERATE_VISUAL_SIMILARITY"
            else:
                category = "PHASH_CANDIDATE_REVIEW"

            results.append({
                "category": category,
                "phash_distance": p_dist,
                "dhash_distance": d_dist,
                "path_a": rec.path,
                "path_b": other.path,
                "base_name_a": rec.base_name,
                "base_name_b": other.base_name,
                "split_a": rec.split,
                "split_b": other.split,
                "sha256_a": rec.sha256,
                "sha256_b": other.sha256,
                "same_sha256": rec.sha256 == other.sha256,
                "width_a": rec.width,
                "height_a": rec.height,
                "width_b": other.width,
                "height_b": other.height,
            })

    results.sort(
        key=lambda x: (
            x["phash_distance"],
            x["dhash_distance"],
            x["base_name_a"],
            x["base_name_b"],
        )
    )

    return results


def exact_cross_base_comparison(records: list[HashRecord]):
    groups = defaultdict(list)

    for rec in records:
        if rec.hash_ok and rec.sha256:
            groups[rec.sha256].append(rec)

    rows = []

    for sha, members in groups.items():
        base_names = sorted(set(r.base_name for r in members))

        if len(base_names) <= 1:
            continue

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a = members[i]
                b = members[j]

                if a.base_name == b.base_name:
                    continue

                rows.append({
                    "sha256": sha,
                    "path_a": a.path,
                    "path_b": b.path,
                    "base_name_a": a.base_name,
                    "base_name_b": b.base_name,
                    "split_a": a.split,
                    "split_b": b.split,
                })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3.5 — Robust Hash Verification"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Root dataset cassava-v7.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Folder output. Default: <dataset>/stage_3_5_hash_verification",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Jumlah worker hashing. Default max 8.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=PHASH_CANDIDATE,
        help="Maksimum Hamming distance pHash untuk kandidat visual. Default 8.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=DHASH_SUPPORT,
        help="Supporting dHash threshold. Default 8.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()

    if not dataset_dir.exists():
        print(f"ERROR: dataset tidak ditemukan: {dataset_dir}")
        sys.exit(1)

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else dataset_dir / "stage_3_5_hash_verification"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 3.5 — ROBUST HASH VERIFICATION")
    print("=" * 80)
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print()

    image_files = iter_images(dataset_dir)

    if not image_files:
        print("ERROR: tidak ada image yang ditemukan.")
        sys.exit(1)

    expected_total = len(image_files)
    print(f"Total image ditemukan: {expected_total:,}")
    print("Menghitung SHA-256 + pHash + dHash...")
    print()

    started = time.time()
    records = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(hash_one, split, path, dataset_dir): (split, path)
            for split, path in image_files
        }

        completed = 0

        for future in as_completed(futures):
            records.append(future.result())
            completed += 1

            if completed % 250 == 0 or completed == expected_total:
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed else 0
                print(
                    f"  [{completed:>6,}/{expected_total:,}] "
                    f"{completed / expected_total * 100:6.2f}% "
                    f"| {rate:,.1f} img/s"
                )

    # Restore deterministic ordering.
    records.sort(key=lambda r: r.path)

    # ------------------------------------------------------------------
    # 1. COVERAGE VALIDATION
    # ------------------------------------------------------------------
    failed = [r for r in records if not r.hash_ok]

    hash_coverage = (
        (len(records) - len(failed)) / len(records) * 100
        if records else 0
    )

    print()
    print("=" * 80)
    print("1. HASH COVERAGE VALIDATION")
    print("=" * 80)
    print(f"Expected images : {expected_total:,}")
    print(f"Processed       : {len(records):,}")
    print(f"Successful      : {len(records) - len(failed):,}")
    print(f"Failed          : {len(failed):,}")
    print(f"Coverage        : {hash_coverage:.4f}%")

    if failed:
        failed_rows = [asdict(r) for r in failed]

        write_csv(
            output_dir / "hash_failures.csv",
            failed_rows,
            list(asdict(failed[0]).keys()),
        )

        print()
        print("ERROR: HASH COVERAGE BELUM 100%.")
        print(
            "Cross-base visual comparison DIBATALKAN agar hasil 0 candidate "
            "tidak disalahartikan."
        )
        print(f"Detail failure: {output_dir / 'hash_failures.csv'}")
        sys.exit(2)

    # Hard assertion: Stage 3.5 tidak boleh lanjut jika coverage bukan 100%.
    assert len(records) == expected_total
    assert all(r.hash_ok for r in records)

    # ------------------------------------------------------------------
    # 2. HASH TABLE
    # ------------------------------------------------------------------
    hash_rows = [asdict(r) for r in records]

    fields = list(hash_rows[0].keys())

    write_csv(
        output_dir / "robust_hash_index.csv",
        hash_rows,
        fields,
    )

    # ------------------------------------------------------------------
    # 3. BASIC STATISTICS
    # ------------------------------------------------------------------
    base_groups = defaultdict(list)

    for rec in records:
        base_groups[rec.base_name].append(rec)

    exact_hash_groups = defaultdict(list)
    for rec in records:
        exact_hash_groups[rec.sha256].append(rec)

    exact_same_file_groups = [
        members
        for members in exact_hash_groups.values()
        if len(members) > 1
    ]

    cross_base_exact = exact_cross_base_comparison(records)

    # ------------------------------------------------------------------
    # 4. CROSS-BASE VISUAL COMPARISON
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("2. CROSS-BASE VISUAL COMPARISON")
    print("=" * 80)
    print(
        f"pHash candidate threshold : <= {args.phash_threshold} bits"
    )
    print(
        f"dHash support threshold   : <= {args.dhash_threshold} bits"
    )
    print(
        "Catatan: kandidat visual bukan otomatis bukti bahwa dua gambar "
        "berasal dari source yang sama."
    )
    print("Menjalankan comparison...")

    visual_candidates = visual_cross_base_comparison(
        records,
        phash_candidate_threshold=args.phash_threshold,
        dhash_support_threshold=args.dhash_threshold,
    )

    write_csv(
        output_dir / "cross_base_visual_candidates.csv",
        visual_candidates,
        [
            "category",
            "phash_distance",
            "dhash_distance",
            "path_a",
            "path_b",
            "base_name_a",
            "base_name_b",
            "split_a",
            "split_b",
            "sha256_a",
            "sha256_b",
            "same_sha256",
            "width_a",
            "height_a",
            "width_b",
            "height_b",
        ],
    )

    write_csv(
        output_dir / "cross_base_exact_duplicates.csv",
        cross_base_exact,
        [
            "sha256",
            "path_a",
            "path_b",
            "base_name_a",
            "base_name_b",
            "split_a",
            "split_b",
        ],
    )

    # ------------------------------------------------------------------
    # 5. SUMMARY
    # ------------------------------------------------------------------
    category_counts = defaultdict(int)
    for row in visual_candidates:
        category_counts[row["category"]] += 1

    cross_base_pairs = {
        tuple(sorted((r["base_name_a"], r["base_name_b"])))
        for r in visual_candidates
    }

    summary = {
        "stage": "3.5 — Robust Hash Verification",
        "dataset": str(dataset_dir),
        "expected_images": expected_total,
        "processed_images": len(records),
        "successful_hashes": len(records) - len(failed),
        "failed_hashes": len(failed),
        "hash_coverage_percent": round(hash_coverage, 6),
        "coverage_100_percent": len(failed) == 0,

        "unique_base_groups": len(base_groups),
        "multi_member_base_groups": sum(
            1 for members in base_groups.values() if len(members) > 1
        ),

        "unique_sha256": len(exact_hash_groups),
        "exact_duplicate_groups": len(exact_same_file_groups),
        "cross_base_exact_duplicate_pairs": len(cross_base_exact),

        "cross_base_visual_candidate_pairs": len(visual_candidates),
        "cross_base_visual_base_pairs": len(cross_base_pairs),

        "visual_categories": dict(category_counts),

        "thresholds": {
            "phash_candidate": args.phash_threshold,
            "phash_strong": PHASH_STRONG,
            "dhash_support": args.dhash_threshold,
        },

        "interpretation": {
            "hash_coverage": (
                "100% coverage achieved. Cross-base results are valid "
                "for the implemented hash-based screening."
                if len(failed) == 0
                else
                "Coverage incomplete. Cross-base results must not be "
                "used to conclude absence of duplicates."
            ),
            "visual_candidates": (
                "Candidates require review; pHash/dHash similarity alone "
                "does not prove common source or leakage."
            ),
        },

        "generated_files": [
            "robust_hash_index.csv",
            "cross_base_exact_duplicates.csv",
            "cross_base_visual_candidates.csv",
            "stage_3_5_summary.json",
        ],
    }

    with (output_dir / "stage_3_5_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6. HUMAN-READABLE VERDICT
    # ------------------------------------------------------------------
    verdict_lines = [
        "STAGE 3.5 — ROBUST HASH VERIFICATION VERDICT",
        "=" * 60,
        f"Total images                : {expected_total:,}",
        f"Successful hash             : {len(records):,}",
        f"Failed hash                 : {len(failed):,}",
        f"Hash coverage               : {hash_coverage:.4f}%",
        f"Unique base groups          : {len(base_groups):,}",
        f"Multi-member base groups    : "
        f"{sum(1 for x in base_groups.values() if len(x) > 1):,}",
        f"Unique SHA-256              : {len(exact_hash_groups):,}",
        f"Exact duplicate groups      : {len(exact_same_file_groups):,}",
        f"Cross-base exact pairs      : {len(cross_base_exact):,}",
        f"Cross-base visual pairs     : {len(visual_candidates):,}",
        f"Cross-base base-pairs       : {len(cross_base_pairs):,}",
        "",
        "VISUAL CANDIDATE BREAKDOWN",
        "-" * 60,
        f"Strong pHash                : "
        f"{category_counts.get('STRONG_VISUAL_SIMILARITY', 0):,}",
        f"Moderate pHash+dHash        : "
        f"{category_counts.get('MODERATE_VISUAL_SIMILARITY', 0):,}",
        f"pHash candidate review      : "
        f"{category_counts.get('PHASH_CANDIDATE_REVIEW', 0):,}",
        "",
    ]

    if len(failed) == 0:
        verdict_lines.extend([
            "VERDICT: PASS — hash coverage 100%.",
            "Cross-base comparison dapat digunakan sebagai screening.",
            "",
            "CATATAN:",
            "Visual candidate tetap perlu review sebelum dianggap leakage.",
            "Kandidat cross-base harus dipertimbangkan saat membentuk",
            "group-aware split penelitian.",
        ])
    else:
        verdict_lines.extend([
            "VERDICT: FAIL — hash coverage < 100%.",
            "Jangan menggunakan hasil cross-base untuk menyimpulkan",
            "bahwa dataset bebas dari duplicate/near-duplicate.",
        ])

    (output_dir / "STAGE_3_5_VERDICT.txt").write_text(
        "\n".join(verdict_lines) + "\n",
        encoding="utf-8",
    )

    elapsed = time.time() - started

    print()
    print("=" * 80)
    print("3. RESULT")
    print("=" * 80)
    print(f"Hash coverage             : {hash_coverage:.4f}%")
    print(f"Unique base groups        : {len(base_groups):,}")
    print(f"Exact cross-base pairs    : {len(cross_base_exact):,}")
    print(f"Visual cross-base pairs   : {len(visual_candidates):,}")
    print(
        f"  Strong                  : "
        f"{category_counts.get('STRONG_VISUAL_SIMILARITY', 0):,}"
    )
    print(
        f"  Moderate                : "
        f"{category_counts.get('MODERATE_VISUAL_SIMILARITY', 0):,}"
    )
    print(
        f"  Candidate review        : "
        f"{category_counts.get('PHASH_CANDIDATE_REVIEW', 0):,}"
    )
    print(f"Runtime                   : {elapsed / 60:.2f} minutes")
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Files utama:")
    print("  - robust_hash_index.csv")
    print("  - cross_base_exact_duplicates.csv")
    print("  - cross_base_visual_candidates.csv")
    print("  - stage_3_5_summary.json")
    print("  - STAGE_3_5_VERDICT.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()

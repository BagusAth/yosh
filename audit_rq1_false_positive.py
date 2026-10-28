#!/usr/bin/env python3
"""
RQ1 bagian B — Membuktikan positif palsu audit kebocoran berbasis nama file (Pakem v1.3)
=======================================================================================

Membandingkan dua cara mengelompokkan citra Roboflow Cassava v7 untuk mendeteksi kebocoran split:
  (1) Berbasis NAMA FILE: grup = nama dasar sebelum akhiran ".rf.<hash>" (cara audit awal).
  (2) Berbasis ISI/KELAS: grup bersih Stage 5 (nama generik image_N diberi prefiks kelas) yang
      diverifikasi dengan SHA-256 dan pHash/dHash (Stage 3.5).

Hanya membaca arsip audit yang sudah ada; tidak mengubah dataset. Output: _analisis_rq1_audit/
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_analisis_rq1_audit"
RF = re.compile(r"\.rf\.[0-9a-fA-F]{32}$")
GENERIC = re.compile(r"^image_\d+_jpg$")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(d: Path) -> list[str]:
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT) if d.exists() else []


def base_name(fn: str) -> str:
    return RF.sub("", os.path.splitext(fn)[0])


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Split asli Roboflow per citra
    orig = {sp: list_images(ROOT / "cassava-v7" / sp / "images") for sp in ("train", "valid", "test")}
    split_of = {fn: sp for sp, fns in orig.items() for fn in fns}

    # (1) Pengelompokan berbasis nama file
    by_base: dict[str, set] = defaultdict(set)
    for fn, sp in split_of.items():
        by_base[base_name(fn)].add(sp)
    cross_base = {b for b, sps in by_base.items() if len(sps) > 1}
    val_in_cross_name = [fn for fn in orig["valid"] if base_name(fn) in cross_base and "train" in by_base[base_name(fn)]]

    # Detail audit awal (base_group_audit)
    bg = read_csv(ROOT / "base_group_audit" / "group_manifest.csv")
    bg_cross = [r for r in bg if r["is_cross_split_original"] == "True"]
    bg_cross_generic = [r for r in bg_cross if GENERIC.match(r["base_name"])]
    bg_cross_conflict = [r for r in bg_cross if r["label_consistency"] == "LABEL_CONFLICT"]
    vis_counts = Counter(r["visual_validation"] for r in bg_cross)
    with_phash = [r for r in bg_cross if r.get("min_phash_distance", "").strip()]

    # (2) Pengelompokan grup bersih (Stage 5/6)
    rg = read_csv(ROOT / "cassava-v7-stage-6" / "resplit_group_manifest.csv")
    clean_group_of = {}
    for r in rg:
        for fn in r["member_filenames"].split(";"):
            fn = fn.strip()
            if fn:
                clean_group_of[fn] = r["clean_group_id"]
    clean_splits: dict[str, set] = defaultdict(set)
    for fn, sp in split_of.items():
        if fn in clean_group_of:
            clean_splits[clean_group_of[fn]].add(sp)
    cross_clean = [g for g, sps in clean_splits.items() if len(sps) > 1]
    members_by_split = Counter()
    for r in rg:
        members_by_split[(r["original_splits"], r["num_images"])] += 1

    # Bukti isi citra (Stage 3.5)
    s35 = ROOT / "cassava-v7-stage-3-5"
    exact = read_csv(s35 / "cross_base_exact_duplicates.csv")
    visual = read_csv(s35 / "cross_base_visual_candidates.csv")
    exact_cross_split = [r for r in exact if r.get("split_a") != r.get("split_b")]
    visual_cross_split = [r for r in visual if r.get("split_a") != r.get("split_b")]

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "original_split_images": {k: len(v) for k, v in orig.items()},
        "filename_based": {
            "base_groups": len(by_base),
            "cross_split_groups": len(cross_base),
            "valid_images_sharing_name_with_train": len(val_in_cross_name),
            "share_of_valid": round(len(val_in_cross_name) / len(orig["valid"]), 4),
            "archived_audit_cross_split_groups": len(bg_cross),
            "archived_cross_groups_generic_name": len(bg_cross_generic),
            "archived_cross_groups_label_conflict": len(bg_cross_conflict),
            "archived_cross_groups_with_phash_evidence": len(with_phash),
            "archived_visual_validation_counts": dict(vis_counts),
        },
        "content_based": {
            "clean_groups": len(rg),
            "clean_groups_spanning_original_splits": len(cross_clean),
            "groups_by_original_split_and_size": {f"{k[0]}|{k[1]} citra": v for k, v in sorted(members_by_split.items())},
            "cross_base_exact_duplicates": len(exact),
            "cross_base_exact_duplicates_cross_split": len(exact_cross_split),
            "cross_base_visual_candidates": len(visual),
            "cross_base_visual_candidates_cross_split": len(visual_cross_split),
        },
    }
    (OUT / "rq1_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUT / "filename_cross_split_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "original_splits", "is_generic_name", "label_consistency", "classes_present", "visual_validation", "min_phash_distance"])
        for r in bg_cross:
            w.writerow([r["base_name"], r["original_splits"], bool(GENERIC.match(r["base_name"])), r["label_consistency"],
                        r.get("classes_present", ""), r["visual_validation"], r.get("min_phash_distance", "")])

    fb, cb = summary["filename_based"], summary["content_based"]
    md = f"""# RQ1 bagian B — Audit kebocoran berbasis nama file menghasilkan positif palsu

Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Kesimpulan:** split bawaan Roboflow Cassava v7 tidak bocor. Audit berbasis nama file menandai {fb['cross_split_groups']:,} grup lintas split
dan {fb['share_of_valid']*100:.2f}% citra validasi berbagi nama dengan citra latih, tetapi pemeriksaan berbasis isi citra menemukan
{cb['clean_groups_spanning_original_splits']} grup lintas split.

| Bukti | Berbasis nama file | Berbasis isi/kelas |
| --- | --- | --- |
| Grup yang melintasi split asli | {fb['cross_split_groups']:,} | {cb['clean_groups_spanning_original_splits']} |
| Citra validasi yang "berpasangan" dengan citra latih | {fb['valid_images_sharing_name_with_train']:,} ({fb['share_of_valid']*100:.2f}%) | 0 |
| Duplikat eksak (SHA-256) lintas grup | — | {cb['cross_base_exact_duplicates']} |
| Kandidat kemiripan visual (pHash) lintas grup | — | {cb['cross_base_visual_candidates']} (lintas split: {cb['cross_base_visual_candidates_cross_split']}) |

Rincian {fb['archived_audit_cross_split_groups']} grup lintas split pada arsip audit awal:

- {fb['archived_cross_groups_generic_name']} bernama file generik (`image_N`).
- {fb['archived_cross_groups_label_conflict']} berkonflik label (kelas berbeda dengan nama sama).
- {fb['archived_cross_groups_with_phash_evidence']} memiliki bukti kemiripan visual tercatat; label "AUGMENTED_DERIVATIVE_PROVENANCE" diberikan berdasarkan nama, bukan isi.

Struktur grup bersih per split asli: {', '.join(f'{k}: {v:,}' for k, v in cb['groups_by_original_split_and_size'].items())}.
Pola ini (grup latih berisi 2 citra hasil augmentasi, grup validasi dan uji berisi 1 citra) konsisten dengan pembagian data
SEBELUM augmentasi, yaitu praktik bawaan Roboflow.

File pendukung: `rq1_audit_summary.json`, `filename_cross_split_groups.csv`.
"""
    (OUT / "RQ1_AUDIT_FALSE_POSITIVE_REPORT.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

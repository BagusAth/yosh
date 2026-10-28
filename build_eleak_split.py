#!/usr/bin/env python3
"""
E-LEAK (RQ1) — Membangun split asli Roboflow dari pool train+val bersih (Pakem v1.2)
====================================================================================

Kontrak:
1. Pool = citra split latih dan validasi BERSIH (cassava-v7-runtime-mirror/train + valid).
   Citra test terkunci (1.282 citra) TIDAK disentuh sama sekali.
2. Setiap citra pool dikembalikan ke split ASLI Roboflow v7 (cassava-v7/{train,valid,test}).
3. Train E-LEAK = anggota pool yang aslinya di train. Val E-LEAK = anggota pool yang aslinya di valid.
   Anggota pool yang aslinya di test tidak dipakai (dilaporkan jumlahnya).
4. Subset val "bocor"  = grup bersihnya punya anggota di train E-LEAK.
   Subset val "bersih" = tidak punya.
5. Hanya menulis file daftar (.txt), data.yaml, dan ringkasan ke folder output baru.
   Tidak ada citra atau label yang disalin atau diubah.

Pemakaian (dari folder SKRIPSI):
  python build_eleak_split.py --root . --win-root D:/UNDIP/SKRIPSI --out eleak_rq1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(d: Path) -> list[str]:
    if not d.exists():
        return []
    # os.listdir tanpa stat per file: jauh lebih cepat pada folder jaringan/terpasang
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)


def load_groups(manifest: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gid = row["clean_group_id"]
            for fn in str(row.get("member_filenames", "")).split(";"):
                fn = fn.strip()
                if fn:
                    groups[fn] = gid
    return groups


def class_counts(label_dir_by_file: dict[str, Path], files: list[str]) -> dict[str, int]:
    c: Counter = Counter()
    for fn in files:
        lbl = label_dir_by_file[fn] / (Path(fn).stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                c[CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"class_{cid}"] += 1
    return {k: int(c.get(k, 0)) for k in CLASS_NAMES}


def main() -> None:
    ap = argparse.ArgumentParser(description="E-LEAK: split asli Roboflow dari pool train+val bersih")
    ap.add_argument("--root", default=".", help="Folder SKRIPSI di sistem tempat skrip dijalankan")
    ap.add_argument("--win-root", default="D:/UNDIP/SKRIPSI", help="Path folder SKRIPSI di Windows (dipakai di file daftar)")
    ap.add_argument("--out", default="eleak_rq1", help="Nama folder output baru di dalam root")
    ap.add_argument("--group-manifest", default="cassava-v7-stage-6/resplit_group_manifest.csv")
    ap.add_argument("--skip-class-counts", action="store_true", help="Lewati hitung objek per kelas (membaca semua label; lambat di folder jaringan)")
    ap.add_argument("--overwrite", action="store_true", help="Izinkan menulis ulang folder output (hasil deterministik)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    win = args.win_root.rstrip("/")
    out = root / args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Folder output {out} sudah berisi file; pakai nama lain agar tidak menimpa hasil lama.")
    out.mkdir(parents=True, exist_ok=True)

    orig_split: dict[str, str] = {}
    for sp in ("train", "valid", "test"):
        for fn in list_images(root / "cassava-v7" / sp / "images"):
            orig_split[fn] = sp

    mirror = root / "cassava-v7-runtime-mirror"
    clean_split: dict[str, str] = {}
    label_dir: dict[str, Path] = {}
    for sp in ("train", "valid"):
        for fn in list_images(mirror / sp / "images"):
            clean_split[fn] = sp
            label_dir[fn] = mirror / sp / "labels"

    groups = load_groups(root / args.group_manifest)

    pool = sorted(clean_split)
    no_orig = [f for f in pool if f not in orig_split]
    no_group = [f for f in pool if f not in groups]
    e_train = [f for f in pool if orig_split.get(f) == "train"]
    e_val = [f for f in pool if orig_split.get(f) == "valid"]
    e_unused = [f for f in pool if orig_split.get(f) == "test"]

    train_groups = {groups[f] for f in e_train if f in groups}
    leaked = [f for f in e_val if groups.get(f) in train_groups]
    clean = [f for f in e_val if f in groups and groups[f] not in train_groups]
    val_ungrouped = [f for f in e_val if f not in groups]

    def win_path(fn: str) -> str:
        return f"{win}/cassava-v7-runtime-mirror/{clean_split[fn]}/images/{fn}"

    lists = {
        "eleak_train.txt": e_train,
        "eleak_val_all.txt": e_val,
        "eleak_val_leaked.txt": leaked,
        "eleak_val_clean.txt": clean,
    }
    for name, files in lists.items():
        (out / name).write_text("\n".join(win_path(f) for f in files) + "\n", encoding="utf-8")

    names_yaml = "[" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]"
    for yname, val_list in (
        ("data_eleak.yaml", "eleak_val_all.txt"),
        ("data_eleak_val_leaked.yaml", "eleak_val_leaked.txt"),
        ("data_eleak_val_clean.yaml", "eleak_val_clean.txt"),
    ):
        (out / yname).write_text(
            f"path: {win}/{args.out}\ntrain: eleak_train.txt\nval: {val_list}\n\nnc: {len(CLASS_NAMES)}\nnames: {names_yaml}\n",
            encoding="utf-8",
        )

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "pool_images_train_plus_val_clean": len(pool),
        "test_split_untouched": True,
        "eleak_train_images": len(e_train),
        "eleak_val_images": len(e_val),
        "eleak_val_leaked_images": len(leaked),
        "eleak_val_clean_images": len(clean),
        "eleak_val_without_group": len(val_ungrouped),
        "pool_originally_test_unused": len(e_unused),
        "pool_without_original_split": len(no_orig),
        "pool_without_group": len(no_group),
        "leaked_share_of_val": round(len(leaked) / len(e_val), 4) if e_val else None,
        "train_groups": len(train_groups),
        "class_objects": None if args.skip_class_counts else {
            "eleak_train": class_counts(label_dir, e_train),
            "eleak_val_leaked": class_counts(label_dir, leaked),
            "eleak_val_clean": class_counts(label_dir, clean),
        },
        "note": "Pasangan kemiripan visual lintas grup (4 kandidat pHash) diabaikan dalam penentuan bocor.",
        "group_manifest": args.group_manifest,
    }
    (out / "eleak_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

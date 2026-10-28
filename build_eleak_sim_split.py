#!/usr/bin/env python3
"""
E-LEAK simulasi (RQ1 bagian A, Pakem v1.3) — split acak per citra dari pool train+val bersih
===========================================================================================

Meniru praktik umum "membagi data per citra setelah augmentasi": dua versi rotasi dari foto yang sama
bisa jatuh di latih dan validasi.

Kontrak:
1. Pool = citra split latih + validasi BERSIH (cassava-v7-runtime-mirror). Test terkunci tidak disentuh.
2. Pool diacak per citra (seed 42) dengan ukuran sama dengan split bersih: 8.980 latih, 2.567 validasi.
3. Subset validasi simulasi:
   - leaked  : grupnya punya anggota lain di latih simulasi (kembaran rotasinya terlihat saat latih)
   - twinval : grup berisi 2 citra yang KEDUANYA di validasi (populasi sama dengan leaked, tanpa kebocoran)
   - single  : grup berisi 1 citra (sumbernya split valid/test asli Roboflow)
4. Pembanding split per grup: subset validasi bersih dari grup 2 citra (clean_val_twin), populasi sama dengan leaked/twinval.
5. Hanya menulis file daftar, data.yaml, dan ringkasan ke folder output baru.

Pemakaian (dari folder SKRIPSI):
  python build_eleak_sim_split.py --root . --win-root D:/UNDIP/SKRIPSI --out eleak_sim_rq1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(d: Path) -> list[str]:
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT) if d.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--win-root", default="D:/UNDIP/SKRIPSI")
    ap.add_argument("--out", default="eleak_sim_rq1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-manifest", default="cassava-v7-stage-6/resplit_group_manifest.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    win = args.win_root.rstrip("/")
    out = root / args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Folder output {out} sudah berisi file; pakai --overwrite (hasil deterministik) atau nama lain.")
    out.mkdir(parents=True, exist_ok=True)

    mirror = root / "cassava-v7-runtime-mirror"
    clean_split = {}
    for sp in ("train", "valid"):
        for fn in list_images(mirror / sp / "images"):
            clean_split[fn] = sp
    n_train_clean = sum(1 for s in clean_split.values() if s == "train")
    n_val_clean = sum(1 for s in clean_split.values() if s == "valid")

    group_of = {}
    with open(root / args.group_manifest, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for fn in row["member_filenames"].split(";"):
                fn = fn.strip()
                if fn:
                    group_of[fn] = row["clean_group_id"]
    pool = sorted(clean_split)
    missing_group = [f for f in pool if f not in group_of]
    if missing_group:
        raise SystemExit(f"{len(missing_group)} citra pool tidak punya grup, contoh: {missing_group[:3]}")
    members = defaultdict(list)
    for fn in pool:
        members[group_of[fn]].append(fn)
    gsize = {g: len(m) for g, m in members.items()}

    rng = random.Random(args.seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    sim_val = set(shuffled[:n_val_clean])
    sim_train = [f for f in pool if f not in sim_val]
    sim_val_sorted = sorted(sim_val)

    def other_in_train(fn: str) -> bool:
        return any(m != fn and m not in sim_val for m in members[group_of[fn]])

    leaked = [f for f in sim_val_sorted if gsize[group_of[f]] >= 2 and other_in_train(f)]
    twinval = [f for f in sim_val_sorted if gsize[group_of[f]] >= 2 and not other_in_train(f)]
    single = [f for f in sim_val_sorted if gsize[group_of[f]] == 1]
    clean_val_twin = sorted(f for f, s in clean_split.items() if s == "valid" and gsize[group_of[f]] >= 2)
    clean_val_twin_ok = all(all(clean_split[m] == "valid" for m in members[group_of[f]]) for f in clean_val_twin)

    def wp(fn: str) -> str:
        return f"{win}/cassava-v7-runtime-mirror/{clean_split[fn]}/images/{fn}"

    lists = {
        "sim_train.txt": sim_train,
        "sim_val_all.txt": sim_val_sorted,
        "sim_val_leaked.txt": leaked,
        "sim_val_twinval.txt": twinval,
        "sim_val_single.txt": single,
        "clean_val_twin.txt": clean_val_twin,
    }
    for name, files in lists.items():
        (out / name).write_text("\n".join(wp(f) for f in files) + "\n", encoding="utf-8")

    names_yaml = "[" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]"
    for yname, val_list in (
        ("data_eleak_sim.yaml", "sim_val_all.txt"),
        ("data_eleak_sim_val_leaked.yaml", "sim_val_leaked.txt"),
        ("data_eleak_sim_val_twinval.yaml", "sim_val_twinval.txt"),
        ("data_eleak_sim_val_single.yaml", "sim_val_single.txt"),
        ("data_clean_val_twin.yaml", "clean_val_twin.txt"),
    ):
        (out / yname).write_text(
            f"path: {win}/{args.out}\ntrain: sim_train.txt\nval: {val_list}\n\nnc: {len(CLASS_NAMES)}\nnames: {names_yaml}\n",
            encoding="utf-8",
        )

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "test_split_untouched": True,
        "pool_images": len(pool),
        "pool_groups": len(members),
        "pool_groups_2_images": sum(1 for v in gsize.values() if v >= 2),
        "sim_train_images": len(sim_train),
        "sim_val_images": len(sim_val_sorted),
        "sim_val_leaked": len(leaked),
        "sim_val_twinval": len(twinval),
        "sim_val_single": len(single),
        "sim_val_leaked_share": round(len(leaked) / len(sim_val_sorted), 4),
        "clean_split_sizes": {"train": n_train_clean, "valid": n_val_clean},
        "clean_val_twin_images": len(clean_val_twin),
        "clean_val_twin_all_members_in_val": clean_val_twin_ok,
    }
    (out / "eleak_sim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

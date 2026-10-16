import os
import glob
from pathlib import Path
from collections import Counter, defaultdict

splits = ["train", "valid", "test"]
dataset_root = Path("cassava-v7-research")

print("=" * 80)
print("AUDIT ALL LABEL FILES ACROSS SPLITS")
print("=" * 80)

split_stats = {}
mixed_files = defaultdict(list)
pure_polygon_files = defaultdict(list)
pure_bbox_files = defaultdict(list)
empty_files = defaultdict(list)

total_objects_before = Counter()
objects_by_class_before = defaultdict(Counter)

for s in splits:
    lbl_dir = dataset_root / s / "labels"
    img_dir = dataset_root / s / "images"
    
    lbl_files = sorted(list(lbl_dir.glob("*.txt")))
    img_files = sorted(list(img_dir.glob("*.jpg")))
    
    print(f"\n--- Split: {s} ---")
    print(f"Total image files: {len(img_files)}")
    print(f"Total label files: {len(lbl_files)}")

    cat_counts = Counter()

    for lf in lbl_files:
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        valid_lines = [l.strip() for l in lines if l.strip()]

        if not valid_lines:
            cat_counts["empty"] += 1
            empty_files[s].append(lf.name)
            continue

        has_bbox = False
        has_poly = False
        n_objs = 0

        for l in valid_lines:
            parts = l.split()
            if len(parts) >= 5:
                n_objs += 1
                cid = int(parts[0])
                total_objects_before[s] += 1
                objects_by_class_before[s][cid] += 1
                if len(parts) == 5:
                    has_bbox = True
                else:
                    has_poly = True

        if has_bbox and has_poly:
            cat_counts["mixed"] += 1
            mixed_files[s].append((lf.name, len(valid_lines)))
        elif has_poly and not has_bbox:
            cat_counts["pure_polygon"] += 1
            pure_polygon_files[s].append(lf.name)
        elif has_bbox and not has_poly:
            cat_counts["pure_bbox"] += 1
            pure_bbox_files[s].append(lf.name)

    print(f"Breakdown for {s}:")
    for cat, count in cat_counts.items():
        print(f"  {cat:15s}: {count:5d} ({count/len(lbl_files)*100:.2f}%)")
    print(f"Total GT objects in {s}: {total_objects_before[s]}")
    print(f"Objects by class in {s}: {dict(objects_by_class_before[s])}")

print("\n" + "=" * 80)
print(f"TOTAL MIXED FILES FOUND:")
for s in splits:
    print(f"  {s}: {len(mixed_files[s])} files")
    if mixed_files[s]:
        for fname, n in mixed_files[s][:5]:
            print(f"    - {fname} ({n} lines)")
        if len(mixed_files[s]) > 5:
            print(f"    ... and {len(mixed_files[s])-5} more")

import os
import sys
import json
import csv
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

def audit_dataset_cache_and_manifest():
    print("\n" + "=" * 80)
    print("AUDIT TASK 5 & 6: DATASET CACHE, MANIFEST & THE 13 MISSING SAMPLES AUDIT")
    print("=" * 80)

    train_img_dir = Path("cassava-v7-research/train/images")
    train_lbl_dir = Path("cassava-v7-research/train/labels")
    cache_path = Path("cassava-v7-research/train/labels.cache")
    manifest_path = Path("stage_8_2_difficulty/train_image_difficulty_manifest.csv")

    all_disk_images = sorted([p.name for p in train_img_dir.glob("*.jpg")])
    all_disk_labels = sorted([p.name for p in train_lbl_dir.glob("*.txt")])
    print(f"Total disk images in train/images: {len(all_disk_images)}")
    print(f"Total disk labels in train/labels: {len(all_disk_labels)}")

    manifest_rows = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_rows[row["image_id"]] = row
    print(f"Total rows in Stage 8.2 difficulty manifest: {len(manifest_rows)}")

    # Load Ultralytics labels.cache
    cached_images = set()
    corrupt_msgs = {}
    if cache_path.exists():
        cache_data = np.load(cache_path, allow_pickle=True).item()
        print(f"Cache keys: {list(cache_data.keys())}")
        print(f"Cache version: {cache_data.get('version')}")
        print(f"Cache hash: {cache_data.get('hash')}")
        print(f"Cache results tuple: {cache_data.get('results')}")
        
        # In Ultralytics cache, 'labels' is a dict or list
        if "labels" in cache_data:
            labels_entry = cache_data["labels"]
            if isinstance(labels_entry, dict):
                cached_images = {Path(p).name for p in labels_entry.keys()}
            elif isinstance(labels_entry, list):
                # check if each item has im_file
                for item in labels_entry:
                    if isinstance(item, dict) and "im_file" in item:
                        cached_images.add(Path(item["im_file"]).name)

        # Check msgs if present
        if "msgs" in cache_data:
            print(f"Cache msgs count: {len(cache_data['msgs'])}")
            for msg in cache_data["msgs"]:
                print(f"  Cache warning/error: {msg}")

    print(f"Total images recognized in labels.cache: {len(cached_images)}")

    diff_disk_manifest = set(all_disk_images) - set(manifest_rows.keys())
    diff_manifest_disk = set(manifest_rows.keys()) - set(all_disk_images)
    print(f"Difference Disk vs Manifest: disk-manifest={len(diff_disk_manifest)}, manifest-disk={len(diff_manifest_disk)}")

    diff_manifest_cache = set(manifest_rows.keys()) - cached_images
    print(f"Difference Manifest vs Cache (Potential 13 samples): {len(diff_manifest_cache)}")
    
    missing_13_details = []
    if diff_manifest_cache:
        print("\nIDENTIFIED SAMPLES MISSING IN CACHE / SAMPLER:")
        for idx, fname in enumerate(sorted(list(diff_manifest_cache))):
            lbl_file = train_lbl_dir / (Path(fname).stem + ".txt")
            lbl_content = ""
            if lbl_file.exists():
                lbl_content = lbl_file.read_text(encoding="utf-8").strip()
            row_info = manifest_rows.get(fname, {})
            print(f"  Sample {idx+1}: {fname}")
            print(f"    Label content: {repr(lbl_content[:100])}")
            print(f"    Manifest d_image: {row_info.get('d_image')}, p_sampling: {row_info.get('p_sampling')}, has_gt: {row_info.get('has_gt')}")
            missing_13_details.append({
                "index": idx + 1,
                "image_name": fname,
                "label_file": str(lbl_file),
                "label_raw": lbl_content,
                "d_image": row_info.get("d_image"),
                "p_sampling": row_info.get("p_sampling"),
                "has_gt": row_info.get("has_gt"),
            })

    return {
        "disk_count": len(all_disk_images),
        "manifest_count": len(manifest_rows),
        "cache_count": len(cached_images),
        "missing_count": len(diff_manifest_cache),
        "missing_samples": missing_13_details
    }


def audit_ap_labels_and_metrics():
    print("\n" + "=" * 80)
    print("AUDIT TASK 2 & 3: CANONICAL METRICS & AP PER-CLASS LABEL AUDIT")
    print("=" * 80)

    data_yaml = Path("cassava-v7-research/data.yaml")
    experiments = {
        "EXP-0": Path("stage_8_ablation/exp_0_baseline"),
        "EXP-1": Path("stage_8_ablation/exp_1_adaptive_sampling"),
        "EXP-2": Path("stage_8_ablation/exp_2_targeted_aug"),
        "EXP-3": Path("stage_8_ablation/exp_3_full_proposed"),
    }

    classes = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
    audit_results = {}

    for exp_id, exp_dir in experiments.items():
        best_pt = exp_dir / "weights" / "best.pt"
        summary_json = exp_dir / "metrics_summary.json"
        results_csv = exp_dir / "results.csv"

        print(f"\nEvaluating canonical checkpoint: {exp_id} ({best_pt})...")
        model = YOLO(str(best_pt))
        val_res = model.val(
            data=str(data_yaml),
            split="val",
            batch=16,
            imgsz=640,
            device="0",
            workers=0,
            plots=False,
            verbose=False,
        )

        p = float(val_res.box.mp)
        r = float(val_res.box.mr)
        map50 = float(val_res.box.map50)
        map50_95 = float(val_res.box.map)

        # AP50 per class (box.ap50)
        ap50_per_class = {}
        if hasattr(val_res.box, "ap50") and val_res.box.ap50 is not None:
            for i, c in enumerate(classes):
                if i < len(val_res.box.ap50):
                    ap50_per_class[c] = float(val_res.box.ap50[i])

        # AP50-95 per class (box.maps)
        ap50_95_per_class = {}
        if hasattr(val_res.box, "maps") and val_res.box.maps is not None:
            for i, c in enumerate(classes):
                if i < len(val_res.box.maps):
                    ap50_95_per_class[c] = float(val_res.box.maps[i])

        # Read what was recorded in metrics_summary.json
        recorded_dict = {}
        if summary_json.exists():
            with open(summary_json, "r") as f:
                sj = json.load(f)
            recorded_dict = sj.get("metrics_validation", {}).get("per_class_mAP50", {})

        print(f"[{exp_id}] Global: P={p:.4f}, R={r:.4f}, mAP@50={map50:.4f}, mAP@50-95={map50_95:.4f}")
        print("  Comparison of Recorded vs Actual ap50 vs actual ap50-95 (maps):")
        for c in classes:
            rec_val = recorded_dict.get(c, 0.0)
            act_ap50 = ap50_per_class.get(c, 0.0)
            act_maps = ap50_95_per_class.get(c, 0.0)
            print(f"    {c:7s}: Recorded in JSON={rec_val:.4f} | Actual AP@50={act_ap50:.4f} | Actual AP@50-95={act_maps:.4f}")

        audit_results[exp_id] = {
            "precision": p,
            "recall": r,
            "map50": map50,
            "map50_95": map50_95,
            "ap50_per_class": ap50_per_class,
            "ap50_95_per_class": ap50_95_per_class,
            "recorded_per_class": recorded_dict,
        }

    return audit_results


def audit_configurations():
    print("\n" + "=" * 80)
    print("AUDIT TASK 4 & 8: CONFIGURATIONS (ARGS.YAML) & STAGE 7 VS EXP-0")
    print("=" * 80)

    configs = {
        "Stage7": Path("cassava-v7-stage-7/runs/baseline_yolo11n/args.yaml"),
        "EXP-0": Path("stage_8_ablation/exp_0_baseline/args.yaml"),
        "EXP-1": Path("stage_8_ablation/exp_1_adaptive_sampling/args.yaml"),
        "EXP-2": Path("stage_8_ablation/exp_2_targeted_aug/args.yaml"),
        "EXP-3": Path("stage_8_ablation/exp_3_full_proposed/args.yaml"),
    }

    loaded_cfgs = {}
    for name, path in configs.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded_cfgs[name] = yaml.safe_load(f)

    # Key parameters to compare
    keys_to_compare = [
        "model", "data", "epochs", "batch", "imgsz", "device", "workers", "seed",
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
        "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste"
    ]

    comp_table = []
    for k in keys_to_compare:
        row = {"parameter": k}
        for name in loaded_cfgs:
            row[name] = loaded_cfgs[name].get(k, "N/A")
        comp_table.append(row)

    print(pd.DataFrame(comp_table).to_string(index=False))
    return comp_table, loaded_cfgs


def main():
    task5_res = audit_dataset_cache_and_manifest()
    task23_res = audit_ap_labels_and_metrics()
    task48_res, all_cfgs = audit_configurations()

    print("\nAudit analysis completed. Writing diagnostic JSON...")
    diag_out = {
        "task5_manifest_cache": task5_res,
        "task23_metrics": task23_res,
        "task48_configs": task48_res,
    }
    with open("stage_9_audit/audit_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag_out, f, indent=2)
    print("Saved to stage_9_audit/audit_diagnostics.json")


if __name__ == "__main__":
    main()

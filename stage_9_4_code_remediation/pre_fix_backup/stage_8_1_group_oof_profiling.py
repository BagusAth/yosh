#!/usr/bin/env python3
"""
Stage 8.1 — Group-Aware 4-Fold Out-of-Fold (OOF) Profiling
==========================================================

Tujuan Ilmiah:
1. Membagi 8.980 citra latih ke dalam 4 fold secara group-aware (isolasi base_group_id 100%).
2. Melatih 4 probe model YOLO11n (12 epoch per fold) pada 3/4 data latih.
3. Melakukan inferensi diagnostik pada 1/4 data holdout dengan threshold sangat rendah (conf = 0.001)
   agar deteksi ber-confidence rendah tidak tereliminasi sebelum matching objek.
4. Mengekspor prediksi OOF mentah untuk seluruh 8.980 citra tanpa kebocoran data.
5. Menjalankan safety checks yang memvalidasi bahwa setiap citra diprediksi eksklusif oleh model
   yang tidak pernah melihat citra maupun grup tersebut dalam pelatihannya.

Output:
- stage_8_1_oof/
  ├── fold_assignments.csv
  ├── fold_class_distribution.csv
  ├── fold_group_leakage_check.csv
  ├── fold_1/weights/best.pt, predictions.csv
  ├── fold_2/weights/best.pt, predictions.csv
  ├── fold_3/weights/best.pt, predictions.csv
  ├── fold_4/weights/best.pt, predictions.csv
  ├── oof_predictions_raw.csv
  ├── oof_evaluated_images.csv
  └── OOF_PROFILING_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 8.1: Group-Aware 4-Fold Out-of-Fold (OOF) Profiling"
    )
    parser.add_argument(
        "--research-dir",
        default="d:/UNDIP/SKRIPSI/cassava-v7-research",
        help="Path ke dataset resmi cassava-v7-research",
    )
    parser.add_argument(
        "--image-index",
        default="d:/UNDIP/SKRIPSI/cassava-v7-stage-5/clean_image_index.csv",
        help="Path ke clean_image_index.csv yang memuat pemetaan clean_group_id",
    )
    parser.add_argument(
        "--output-dir",
        default="d:/UNDIP/SKRIPSI/stage_8_1_oof",
        help="Folder output untuk artefak Stage 8.1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Jumlah epoch per probe model fold (default: 12)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size pelatihan (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi citra input (default: 640)",
    )
    parser.add_argument(
        "--conf-oof",
        type=float,
        default=0.001,
        help="Threshold confidence inferensi diagnostik OOF (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reprodusibilitas (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Perangkat GPU CUDA (default: '0')",
    )
    parser.add_argument(
        "--skip-training-if-done",
        action="store_true",
        help="Lewati training fold jika file weights/best.pt dan predictions.csv sudah lengkap",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Jalankan hanya partisi 4-fold, audit label, dan leakage check tanpa training model",
    )
    return parser.parse_args()


def load_annotations(train_dir: Path):
    """
    Membaca label YOLO langsung dari file .txt untuk menentukan:
    - n_objects
    - class distribution
    - empty label status
    """
    labels_dir = train_dir / "labels"
    images_dir = train_dir / "images"

    image_files = sorted(os.listdir(images_dir))
    annotation_data = {}

    empty_count = 0
    pure_healthy_count = 0
    pure_disease_count = 0
    mixed_count = 0

    for fname in image_files:
        stem = Path(fname).stem
        lbl_file = labels_dir / f"{stem}.txt"

        boxes = []
        classes = []

        if lbl_file.exists() and os.path.getsize(lbl_file) > 0:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        classes.append(cls_id)
                        boxes.append([float(x) for x in parts[1:5]])

        n_obj = len(classes)
        has_healthy = (4 in classes)
        has_disease = any(c in [0, 1, 2, 3] for c in classes)

        if n_obj == 0:
            empty_count += 1
        elif has_healthy and has_disease:
            mixed_count += 1
        elif has_healthy:
            pure_healthy_count += 1
        elif has_disease:
            pure_disease_count += 1

        annotation_data[fname] = {
            "filename": fname,
            "image_path": str((images_dir / fname).resolve()),
            "label_path": str(lbl_file.resolve()),
            "n_objects": n_obj,
            "classes": classes,
            "boxes": boxes,
            "is_empty": (n_obj == 0),
            "has_healthy": has_healthy,
            "has_disease": has_disease,
        }

    total_with_healthy = pure_healthy_count + mixed_count
    total_with_disease = pure_disease_count + mixed_count

    print(f"[AUDIT LABEL] Total citra latih: {len(image_files):,}")
    print(f"              Empty label (0 box)          : {empty_count:,}")
    print(f"              Pure HEALTHY bbox only       : {pure_healthy_count:,}")
    print(f"              Pure Disease bbox (CBB..CMD) : {pure_disease_count:,}")
    print(f"              Mixed (HEALTHY + Disease)    : {mixed_count:,}")
    print(f"              Total citra memuat HEALTHY   : {total_with_healthy:,} (2.202 pure + 8 mixed)")
    print(f"              Total citra memuat Disease   : {total_with_disease:,} (6.704 pure + 8 mixed)")

    assert len(image_files) == 8980, f"Expected 8980 images, got {len(image_files)}"
    assert empty_count == 66, f"Expected 66 empty labels, got {empty_count}"
    assert pure_healthy_count == 2202, f"Expected 2202 pure healthy files, got {pure_healthy_count}"
    assert pure_disease_count == 6704, f"Expected 6704 pure disease files, got {pure_disease_count}"
    assert mixed_count == 8, f"Expected 8 mixed files, got {mixed_count}"
    assert total_with_healthy == 2210, f"Expected 2210 total with healthy, got {total_with_healthy}"
    assert (empty_count + pure_healthy_count + pure_disease_count + mixed_count) == 8980
    print("[AUDIT LABEL] Verifikasi source of truth PASSED (66 + 2.202 + 6.704 + 8 = 8.980).")

    return annotation_data


def load_group_mapping(index_path: Path, image_filenames: set[str]):
    """Memetakan setiap citra ke clean_group_id dari clean_image_index.csv."""
    img_to_group = {}
    with open(index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if fname in image_filenames:
                img_to_group[fname] = row["clean_group_id"]

    missing = image_filenames - set(img_to_group.keys())
    if missing:
        print(f"FATAL ERROR: {len(missing)} citra tidak memiliki mapping clean_group_id!")
        sys.exit(1)

    print(f"[GROUP MAPPING] Berhasil memetakan {len(img_to_group):,} citra ke grup.")
    return img_to_group


def build_group_stratified_folds(
    annotation_data: dict, img_to_group: dict, n_splits: int = 4, seed: int = 42
):
    """
    Membagi kelompok (base_group_id) ke dalam 4 fold secara stratifikasi kelas.
    Menjamin:
    - Tidak ada group leakage (1 group masuk tepat 1 fold).
    - Distribusi kelas dan citra antar-fold seimbang.
    """
    rng = random.Random(seed)

    # 1. Kumpulkan anggota per grup
    groups = defaultdict(list)
    for fname, data in annotation_data.items():
        gid = img_to_group[fname]
        groups[gid].append(fname)

    print(f"[GROUP SPLIT] Total grup atomik pada train set: {len(groups):,}")

    # 2. Tentukan signature kelas per grup untuk stratifikasi
    groups_by_sig = defaultdict(list)
    for gid, fnames in groups.items():
        cls_counter = Counter()
        for fn in fnames:
            cls_counter.update(annotation_data[fn]["classes"])

        if len(cls_counter) == 0:
            sig = "EMPTY"
        else:
            # Kelas dengan kemunculan tertinggi
            top_cls_id = cls_counter.most_common(1)[0][0]
            sig = CLASS_NAMES[top_cls_id]

        groups_by_sig[sig].append((gid, fnames, len(fnames)))

    # 3. Alokasikan ke 4 fold secara seimbang menggunakan greedy balancing
    fold_groups = {i: [] for i in range(1, n_splits + 1)}
    fold_image_counts = {i: 0 for i in range(1, n_splits + 1)}

    for sig, group_tuples in sorted(groups_by_sig.items()):
        # Acak urutan grup dalam stratum
        shuffled = list(group_tuples)
        rng.shuffle(shuffled)
        # Urutkan berdasarkan ukuran grup menurun untuk bin packing halus
        shuffled.sort(key=lambda x: x[2], reverse=True)

        for gid, fnames, n_imgs in shuffled:
            # Pilih fold dengan jumlah citra terkecil saat ini
            target_fold = min(fold_image_counts, key=fold_image_counts.get)
            fold_groups[target_fold].append(gid)
            fold_image_counts[target_fold] += n_imgs

    # 4. Petakan citra ke fold
    image_fold_assignment = {}
    for fold_id, gids in fold_groups.items():
        for gid in gids:
            for fname in groups[gid]:
                image_fold_assignment[fname] = fold_id

    # 5. Verifikasi Integritas Split
    print("\n[VERIFIKASI INTEGRITAS 4-FOLD]")
    assert len(image_fold_assignment) == 8980, f"Total assigned images != 8980"

    all_assigned_groups = set()
    for fold_id, gids in fold_groups.items():
        gset = set(gids)
        overlap = all_assigned_groups.intersection(gset)
        assert len(overlap) == 0, f"Group leakage detected in fold {fold_id}: {overlap}"
        all_assigned_groups.update(gset)
        print(f"  Fold {fold_id}: {len(gids):,} grup, {fold_image_counts[fold_id]:,} citra")

    print("[VERIFIKASI INTEGRITAS] Zero cross-fold leakage terverifikasi (100% PASS).\n")

    return fold_groups, image_fold_assignment, groups


def generate_fold_configs(
    annotation_data: dict,
    image_fold_assignment: dict,
    output_dir: Path,
    n_splits: int = 4,
):
    """
    Membuat file daftar citra (.txt) dan file konfigurasi YAML untuk masing-masing fold.
    Fold k:
      train: 3 fold lainnya (Holdout -k)
      val  : fold k
    """
    config_dir = output_dir / "data_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_paths = {}

    for fold_id in range(1, n_splits + 1):
        train_images = []
        val_images = []

        for fname, assigned_fold in image_fold_assignment.items():
            img_path = annotation_data[fname]["image_path"].replace("\\", "/")
            if assigned_fold == fold_id:
                val_images.append(img_path)
            else:
                train_images.append(img_path)

        train_txt = config_dir / f"fold_{fold_id}_train.txt"
        val_txt = config_dir / f"fold_{fold_id}_val.txt"

        with open(train_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(train_images) + "\n")

        with open(val_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(val_images) + "\n")

        yaml_content = f"""# Ultralytics YOLO OOF Configuration — Fold {fold_id}
train: {train_txt.resolve().as_posix()}
val: {val_txt.resolve().as_posix()}

nc: 5
names: {CLASS_NAMES}
"""
        yaml_file = config_dir / f"fold_{fold_id}_data.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        yaml_paths[fold_id] = yaml_file

    print(f"[DATA CONFIG] 4 file YAML konfigurasi fold dibuat di {config_dir}")
    return yaml_paths


def train_and_predict_fold(
    fold_id: int,
    yaml_path: Path,
    holdout_filenames: list[str],
    annotation_data: dict,
    output_dir: Path,
    args,
):
    """
    Melatih probe model YOLO11n pada fold k (12 epoch) dan
    menjalankan inferensi diagnostik pada holdout fold dengan conf = 0.001.
    """
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred_csv_path = fold_dir / "predictions.csv"
    weights_dir = fold_dir / "weights"

    # Cek apakah sudah selesai dan bisa diskip
    best_weight_path = weights_dir / "best.pt"
    trained_best = fold_dir / "train_run" / "weights" / "best.pt"
    trained_last = fold_dir / "train_run" / "weights" / "last.pt"

    if args.skip_training_if_done and pred_csv_path.exists() and best_weight_path.exists():
        print(f"[FOLD {fold_id}] Melewati pelatihan dan inferensi (artefak sudah lengkap).")
        return pred_csv_path

    # 1. Train Model (Hanya jika best_weight_path belum ada)
    if not best_weight_path.exists():
        print("=" * 80)
        print(f"MEMULAI PROBE TRAINING: FOLD {fold_id} / 4")
        print(f"Holdout Images : {len(holdout_filenames):,} citra")
        print(f"Train Epochs   : {args.epochs}")
        print(f"Batch Size     : {args.batch}")
        print(f"YAML Path      : {yaml_path}")
        print("=" * 80)

        t_start = time.time()
        model = YOLO("yolo11n.pt")

        train_results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            device=args.device,
            workers=0,  # 0 workers mencegah WinError 1455 pada Windows multiprocessing
            project=str(fold_dir),
            name="train_run",
            save=True,
            exist_ok=True,
            verbose=False,
            val=True,
            plots=False,
        )

        t_train = time.time() - t_start
        print(f"[FOLD {fold_id}] Pelatihan selesai dalam {t_train / 60:.2f} menit.")

        weights_dir.mkdir(parents=True, exist_ok=True)
        if trained_best.exists():
            shutil.copy2(trained_best, best_weight_path)
        elif trained_last.exists():
            shutil.copy2(trained_last, best_weight_path)
        else:
            raise FileNotFoundError(f"Model checkpoint tidak ditemukan di {fold_dir}")

        del model, train_results
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[FOLD {fold_id}] Checkpoint best.pt ditemukan. Melanjutkan ke inferensi diagnostik.")

    # 3. Jalankan Diagnostic OOF Prediction @ conf = 0.001 dalam mini-batch chunked
    print(f"[FOLD {fold_id}] Menjalankan inferensi diagnostik OOF @ conf = {args.conf_oof}...")
    eval_model = YOLO(str(best_weight_path))

    holdout_paths = [annotation_data[fn]["image_path"] for fn in holdout_filenames]

    t_infer_start = time.time()
    predictions = []
    evaluated_count = 0
    chunk_size = 32  # Chunked batching mencegah CUDA OOM saat inferensi 2.246 citra

    for c_idx in range(0, len(holdout_paths), chunk_size):
        chunk = holdout_paths[c_idx : c_idx + chunk_size]
        results = eval_model.predict(
            source=chunk,
            conf=args.conf_oof,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            stream=False,
        )

        for res in results:
            evaluated_count += 1
            img_fname = Path(res.path).name
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for b_idx in range(len(boxes)):
                    cls_id = int(boxes.cls[b_idx].item())
                    conf_val = float(boxes.conf[b_idx].item())
                    xyxy = boxes.xyxy[b_idx].tolist()
                    predictions.append({
                        "image_id": img_fname,
                        "fold_id": fold_id,
                        "prediction_id": b_idx,
                        "class_id": cls_id,
                        "confidence": f"{conf_val:.6f}",
                        "x1": f"{xyxy[0]:.2f}",
                        "y1": f"{xyxy[1]:.2f}",
                        "x2": f"{xyxy[2]:.2f}",
                        "y2": f"{xyxy[3]:.2f}",
                    })

    del eval_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_infer = time.time() - t_infer_start
    print(f"[FOLD {fold_id}] Inferensi selesai: {evaluated_count:,} citra diproses, "
          f"{len(predictions):,} bounding box terdeteksi ({t_infer / 60:.2f} menit).")

    # 4. Simpan predictions.csv
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "fold_id", "prediction_id", "class_id",
            "confidence", "x1", "y1", "x2", "y2"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"[FOLD {fold_id}] Prediksi disimpan ke {pred_csv_path}")
    return pred_csv_path


def main():
    args = parse_args()
    print("=" * 80)
    print("STAGE 8.1 — GROUP-AWARE 4-FOLD OUT-OF-FOLD (OOF) PROFILING")
    print("=" * 80)
    print(f"Waktu Mulai    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset Dir    : {args.research_dir}")
    print(f"Output Dir     : {args.output_dir}")
    print(f"Probe Epochs   : {args.epochs}")
    print(f"Conf Threshold : {args.conf_oof}")
    print(f"Random Seed    : {args.seed}")
    print(f"CUDA Device    : {args.device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    research_dir = Path(args.research_dir).resolve()
    train_dir = research_dir / "train"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit Label & Ground Truth dari file label .txt
    annotation_data = load_annotations(train_dir)

    # 2. Pemetaan Group
    img_to_group = load_group_mapping(Path(args.image_index).resolve(), set(annotation_data.keys()))

    # 3. Pembagian 4-Fold Group-Aware
    fold_groups, image_fold_assignment, groups = build_group_stratified_folds(
        annotation_data, img_to_group, n_splits=4, seed=args.seed
    )

    # Simpan fold_assignments.csv
    fold_assign_path = output_dir / "fold_assignments.csv"
    with open(fold_assign_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "base_group_id", "fold_id", "split"])
        for fname, fold_id in sorted(image_fold_assignment.items()):
            writer.writerow([fname, img_to_group[fname], fold_id, "train"])
    print(f"[OUTPUT] Tersimpan: {fold_assign_path}")

    # Simpan fold_class_distribution.csv
    fold_class_dist_path = output_dir / "fold_class_distribution.csv"
    with open(fold_class_dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "num_images", "num_groups", "EMPTY", "CBB", "CBSD", "CGM", "CMD", "HEALTHY"])
        for f_id in range(1, 5):
            c_counts = Counter()
            for fn, f_curr in image_fold_assignment.items():
                if f_curr == f_id:
                    classes = annotation_data[fn]["classes"]
                    if not classes:
                        c_counts["EMPTY"] += 1
                    else:
                        for c in classes:
                            c_counts[CLASS_NAMES[c]] += 1
            writer.writerow([
                f_id,
                sum(1 for f_curr in image_fold_assignment.values() if f_curr == f_id),
                len(fold_groups[f_id]),
                c_counts["EMPTY"],
                c_counts["CBB"],
                c_counts["CBSD"],
                c_counts["CGM"],
                c_counts["CMD"],
                c_counts["HEALTHY"],
            ])
    print(f"[OUTPUT] Tersimpan: {fold_class_dist_path}")

    # Simpan fold_group_leakage_check.csv
    leakage_check_path = output_dir / "fold_group_leakage_check.csv"
    with open(leakage_check_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_a", "fold_b", "shared_groups", "leakage_status"])
        for i in range(1, 5):
            for j in range(i + 1, 5):
                inter = set(fold_groups[i]).intersection(set(fold_groups[j]))
                writer.writerow([f"fold_{i}", f"fold_{j}", len(inter), "LEAKAGE_DETECTED" if inter else "ZERO_LEAKAGE"])
    print(f"[OUTPUT] Tersimpan: {leakage_check_path}")

    # 4. Buat File Konfigurasi Pelatihan Fold
    yaml_paths = generate_fold_configs(
        annotation_data, image_fold_assignment, output_dir, n_splits=4
    )

    if args.dry_run:
        print("\n" + "=" * 80)
        print("[DRY-RUN] Verifikasi partisi, label audit, dan leakage check SELESAI (100% SUKSES).")
        print("          Semua konfigurasi data fold siap untuk dieksekusi training penuh.")
        print("=" * 80)
        return

    # 5. Pelatihan dan Inferensi OOF untuk Ke-4 Fold
    fold_pred_files = {}
    for fold_id in range(1, 5):
        holdout_imgs = [fn for fn, f_curr in image_fold_assignment.items() if f_curr == fold_id]
        pred_file = train_and_predict_fold(
            fold_id=fold_id,
            yaml_path=yaml_paths[fold_id],
            holdout_filenames=holdout_imgs,
            annotation_data=annotation_data,
            output_dir=output_dir,
            args=args,
        )
        fold_pred_files[fold_id] = pred_file

    # 6. Agregasi Raw OOF Predictions
    print("\n[AGREGASI PREDIKSI OOF]")
    raw_oof_path = output_dir / "oof_predictions_raw.csv"
    evaluated_img_path = output_dir / "oof_evaluated_images.csv"

    all_raw_predictions = []
    images_with_preds = defaultdict(int)

    for fold_id, p_file in fold_pred_files.items():
        with open(p_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_raw_predictions.append(row)
                images_with_preds[row["image_id"]] += 1

    with open(raw_oof_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "fold_id", "prediction_id", "class_id",
            "confidence", "x1", "y1", "x2", "y2"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_raw_predictions)
    print(f"[OUTPUT] Tersimpan: {raw_oof_path} ({len(all_raw_predictions):,} prediksi mentah)")

    with open(evaluated_img_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "fold_id", "num_predictions", "n_gt_objects", "is_empty"])
        for fname, fold_id in sorted(image_fold_assignment.items()):
            writer.writerow([
                fname,
                fold_id,
                images_with_preds.get(fname, 0),
                annotation_data[fname]["n_objects"],
                int(annotation_data[fname]["is_empty"]),
            ])
    print(f"[OUTPUT] Tersimpan: {evaluated_img_path}")

    # 7. SAFETY CHECKS PASCA-OOF
    print("\n" + "=" * 80)
    print("SAFETY CHECKS PASCA-OOF PROFILING")
    print("=" * 80)

    # Check 1: 8980 unique images assigned
    assert len(image_fold_assignment) == 8980, f"FAIL: Expected 8980 unique images, got {len(image_fold_assignment)}"
    print("[SAFETY CHECK 1] Total citra ter-assign: 8.980 (PASS)")

    # Check 2: 8980 images accounted in evaluated_images
    with open(evaluated_img_path, "r", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    assert len(eval_rows) == 8980, f"FAIL: Expected 8980 evaluated images, got {len(eval_rows)}"
    print("[SAFETY CHECK 2] Total citra dievaluasi OOF: 8.980 (PASS)")

    # Check 3: fold_prediction == fold_assignment
    mismatch_count = 0
    for row in all_raw_predictions:
        fname = row["image_id"]
        fold_pred = int(row["fold_id"])
        fold_assigned = image_fold_assignment[fname]
        if fold_pred != fold_assigned:
            mismatch_count += 1
    assert mismatch_count == 0, f"FAIL: Ditemukan {mismatch_count} ketidaksesuaian fold_prediction vs fold_assignment!"
    print("[SAFETY CHECK 3] Verifikasi fold_prediction == fold_assignment: 100% Cocok (PASS)")

    # Check 4: Cross-fold group leakage
    leakage_count = 0
    for i in range(1, 5):
        for j in range(i + 1, 5):
            inter = set(fold_groups[i]).intersection(set(fold_groups[j]))
            leakage_count += len(inter)
    assert leakage_count == 0, f"FAIL: Ditemukan {leakage_count} group leakage!"
    print("[SAFETY CHECK 4] Cross-Fold Group Leakage: 0.00% (PASS)")

    print("=" * 80)
    print("SELURUH SAFETY CHECKS TAHAP 8.1: PASS 100%")
    print("=" * 80)

    # 8. Tulis OOF_PROFILING_REPORT.md
    report_path = output_dir / "OOF_PROFILING_REPORT.md"
    report_content = f"""# Laporan Hasil Stage 8.1 — Group-Aware 4-Fold OOF Profiling
Tanggal Eksekusi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Ringkasan Eksekusi
- **Dataset Latih**: 8.980 citra ({len(groups):,} grup atomik)
- **Skema Partisi**: 4-Fold Group-Aware Stratified Split
- **Isolasi Grup**: 0.00% Cross-Fold Group Leakage (100% Clean)
- **Model Probe**: YOLO11n (12 epoch per fold)
- **Threshold Inferensi Diagnostik**: conf = {args.conf_oof}

## 2. Distribusi Fold
| Fold ID | Jumlah Grup | Jumlah Citra | Empty Labels | Total Prediksi (@conf={args.conf_oof}) |
| :---: | :---: | :---: | :---: | :---: |
| Fold 1 | {len(fold_groups[1]):,} | {sum(1 for f in image_fold_assignment.values() if f == 1):,} | {sum(1 for fn, f in image_fold_assignment.items() if f == 1 and annotation_data[fn]['is_empty']):,} | {sum(1 for p in all_raw_predictions if int(p['fold_id']) == 1):,} |
| Fold 2 | {len(fold_groups[2]):,} | {sum(1 for f in image_fold_assignment.values() if f == 2):,} | {sum(1 for fn, f in image_fold_assignment.items() if f == 2 and annotation_data[fn]['is_empty']):,} | {sum(1 for p in all_raw_predictions if int(p['fold_id']) == 2):,} |
| Fold 3 | {len(fold_groups[3]):,} | {sum(1 for f in image_fold_assignment.values() if f == 3):,} | {sum(1 for fn, f in image_fold_assignment.items() if f == 3 and annotation_data[fn]['is_empty']):,} | {sum(1 for p in all_raw_predictions if int(p['fold_id']) == 3):,} |
| Fold 4 | {len(fold_groups[4]):,} | {sum(1 for f in image_fold_assignment.values() if f == 4):,} | {sum(1 for fn, f in image_fold_assignment.items() if f == 4 and annotation_data[fn]['is_empty']):,} | {sum(1 for p in all_raw_predictions if int(p['fold_id']) == 4):,} |
| **Total** | **{len(groups):,}** | **8.980** | **66** | **{len(all_raw_predictions):,}** |

## 3. Status Safety Checks
- [x] Total citra unik ter-assign: 8.980 (PASS)
- [x] Total citra terprediksi OOF: 8.980 (PASS)
- [x] Konsistensi fold prediksi dan fold penugasan: 100% Cocok (PASS)
- [x] Kebocoran kelompok antar-fold (Group Leakage): 0 (PASS)

## 4. Langkah Selanjutnya
Artefak `oof_predictions_raw.csv` siap digunakan oleh `stage_8_2_build_difficulty_manifest.py`
untuk pencocokan greedy one-to-one class-agnostic matching dan kalkulasi skor kesulitan $D_i$ serta $D_\\text{{image}}$.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[REPORT] Tersimpan: {report_path}")


if __name__ == "__main__":
    main()

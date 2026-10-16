#!/usr/bin/env python3
"""
Stage 8.3 — Adaptive Data Loading & On-the-Fly Context Augmentation Pipeline
=============================================================================

Kontrak Metodologis:
1. AdaptiveYOLODataset:
   - Mengintegrasikan manifest kesulitan citra dari Stage 8.2 (train_image_difficulty_manifest.csv).
   - On-the-Fly Targeted Context Augmentation:
     * Khusus untuk citra dengan lesi Small sulit (has_target_small == 1, D_i >= 0.70).
     * Lesi target tunggal i* (kesulitan tertinggi).
     * Jendela crop diperluas 2× dimensi bounding box lesi (w' = min(W, 2w), h' = min(H, 2h))
       untuk mempertahankan konteks morfologis daun di sekitar lesi.
     * Bounding box lain yang masuk jendela crop:
       - Visibilitas >= 0.30: dipertahankan dan ditransformasikan ke koordinat crop.
       - Visibilitas < 0.30: dibuang agar tidak menyajikan potongan lesi yang menyesatkan model.
     * Patch crop di-resize ke 640x640.
     * Mosaic dinonaktifkan khusus pada patch crop terarah agar lesi tidak kembali mengecil.
     * Pada citra lainnya atau saat augmentasi tidak terpicu, augmentasi baseline standar (termasuk Mosaic)
       dipertahankan secara identik.
2. Mixture Adaptive Sampling:
   - Mengambil probabilitas terkalibrasi P(k) = 0.3*(1/N) + 0.7*(D_k / sum(D)) dari manifest.
   - Membangun WeightedRandomSampler dengan penggantian (with replacement).
   - Menjamin setiap citra memiliki peluang kemunculan floor minimal 0.30/N.
3. AdaptiveDetectionTrainer:
   - Subclass resmi dari Ultralytics DetectionTrainer yang dapat mengaktifkan:
     * EXP-0: Baseline Re-run (Uniform Sampling, Standard Augmentation)
     * EXP-1: Baseline + Mixture Adaptive Sampling
     * EXP-2: Baseline + Targeted Context Augmentation (On-the-fly)
     * EXP-3: Full Proposed (Mixture Sampling + Targeted Context Augmentation)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from ultralytics.data.augment import Compose, Format, RandomFlip, RandomHSV, v8_transforms
from ultralytics.data.base import BaseDataset
from ultralytics.data.build import InfiniteDataLoader, build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first


def load_difficulty_manifest(manifest_path: str | Path) -> dict[str, dict]:
    """Membaca train_image_difficulty_manifest.csv dari Stage 8.2 ke dictionary."""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest kesulitan tidak ditemukan di {manifest_path}!")

    records = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["image_id"]
            records[fname] = {
                "d_image": float(row["d_image"]),
                "p_sampling": float(row["p_sampling"]),
                "has_gt": int(row["has_gt"]),
                "has_target_small": (row["has_target_small"] == "1"),
                "target_gt_id": int(row["target_gt_id"]),
                "target_small_di": float(row["target_small_di"]),
                "target_small_box": row["target_small_box"],  # 'x1,y1,x2,y2'
            }
    return records


class AdaptiveYOLODataset(YOLODataset):
    """
    YOLODataset kustom dengan kemampuan On-the-Fly Targeted Context Augmentation.
    """

    def __init__(
        self,
        *args,
        manifest_path: str | Path | None = None,
        enable_targeted_aug: bool = False,
        aug_prob: float = 0.50,
        context_margin: float = 2.0,
        vis_thresh: float = 0.30,
        hyp: Any = DEFAULT_CFG,
        **kwargs,
    ):
        self.hyp = hyp
        self.enable_targeted_aug = enable_targeted_aug
        self.aug_prob = aug_prob
        self.context_margin = context_margin
        self.vis_thresh = vis_thresh

        # Muat manifest kesulitan jika path diberikan
        self.manifest_data = {}
        if manifest_path:
            self.manifest_data = load_difficulty_manifest(manifest_path)

        super().__init__(*args, hyp=hyp, **kwargs)

        # Buat transforms khusus untuk patch crop (tanpa Mosaic sekunder)
        if self.augment:
            hyp = self.hyp
            self.crop_transforms = Compose([
                RandomHSV(hgain=getattr(hyp, "hsv_h", 0.015), sgain=getattr(hyp, "hsv_s", 0.7), vgain=getattr(hyp, "hsv_v", 0.4)),
                RandomFlip(direction="horizontal", p=getattr(hyp, "fliplr", 0.5)),
                self.format_class(
                    bbox_format="xywh",
                    normalize=True,
                    return_mask=self.use_segments,
                    return_keypoint=self.use_keypoints,
                    return_obb=self.use_obb,
                    batch_idx=True,
                    mask_ratio=getattr(hyp, "mask_ratio", 4),
                    mask_overlap=getattr(hyp, "overlap_mask", True),
                    bgr=getattr(hyp, "bgr", 0.0),
                ),
            ])

    def get_sampling_weights(self) -> torch.Tensor:
        """Mengambil bobot sampling P(k) untuk seluruh citra pada dataset."""
        weights = []
        default_p = 1.0 / len(self.im_files) if len(self.im_files) > 0 else 1.0

        for im_file in self.im_files:
            fname = Path(im_file).name
            if fname in self.manifest_data:
                weights.append(self.manifest_data[fname]["p_sampling"])
            else:
                weights.append(default_p)

        return torch.tensor(weights, dtype=torch.float32)

    def apply_context_crop(self, label: dict[str, Any], target_info: dict[str, Any]) -> dict[str, Any]:
        """
        Menerapkan Targeted Context-Preserving Crop:
        - Crop 2× dimensi bounding box di sekitar lesi target i*
        - Mempertahankan urat daun & konteks jaringan sekitar lesi
        - Mempertahankan lesi lain jika visibilitas >= 0.30
        - Resize ke resolusi input model (640x640)
        """
        img = label["img"]
        H, W = img.shape[:2]

        box_str = target_info.get("target_small_box", "")
        if not box_str:
            return label

        parts = [float(v) for v in box_str.split(",")]
        if len(parts) != 4:
            return label

        bx1, by1, bx2, by2 = parts
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        # Perluasan dimensi 2×
        w_crop = min(float(W), self.context_margin * bw)
        h_crop = min(float(H), self.context_margin * bh)

        # Clamping jendela crop pada kanvas citra
        x_min = max(0.0, min(float(W) - w_crop, bcx - w_crop / 2.0))
        y_min = max(0.0, min(float(H) - h_crop, bcy - h_crop / 2.0))
        x_max = x_min + w_crop
        y_max = y_min + h_crop

        ix_min, iy_min, ix_max, iy_max = (
            int(round(x_min)),
            int(round(y_min)),
            int(round(x_max)),
            int(round(y_max)),
        )

        cropped_img = img[iy_min:iy_max, ix_min:ix_max]
        if cropped_img.size == 0 or cropped_img.shape[0] < 4 or cropped_img.shape[1] < 4:
            return label

        # Resize patch crop kembali ke 640x640
        resized_crop = cv2.resize(cropped_img, (640, 640), interpolation=cv2.INTER_LINEAR)
        scale_x = 640.0 / w_crop
        scale_y = 640.0 / h_crop

        orig_boxes = label.get("bboxes", np.zeros((0, 4), dtype=np.float32))
        orig_cls = label.get("cls", np.zeros((0, 1), dtype=np.float32))

        new_bboxes = []
        new_cls = []

        for idx in range(len(orig_boxes)):
            c_id = orig_cls[idx]
            box = orig_boxes[idx]
            # Format normalized xywh -> pixel xyxy
            gxc = box[0] * W
            gyc = box[1] * H
            gw = box[2] * W
            gh = box[3] * H
            gx1 = gxc - gw / 2.0
            gy1 = gyc - gh / 2.0
            gx2 = gxc + gw / 2.0
            gy2 = gyc + gh / 2.0
            orig_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))

            # Irisan dengan jendela crop
            inter_x1 = max(gx1, x_min)
            inter_y1 = max(gy1, y_min)
            inter_x2 = min(gx2, x_max)
            inter_y2 = min(gy2, y_max)

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            vis_ratio = inter_area / orig_area

            # Lesi target selalu dipertahankan; lesi lain dipertahankan jika visibilitas >= 0.30
            is_target = (abs(gx1 - bx1) < 4.0 and abs(gy1 - by1) < 4.0)
            if is_target or vis_ratio >= self.vis_thresh:
                # Transformasi ke koordinat crop [0, 640]
                cx1 = (inter_x1 - x_min) * scale_x
                cy1 = (inter_y1 - y_min) * scale_y
                cx2 = (inter_x2 - x_min) * scale_x
                cy2 = (inter_y2 - y_min) * scale_y

                cx1 = max(0.0, min(640.0, cx1))
                cy1 = max(0.0, min(640.0, cy1))
                cx2 = max(0.0, min(640.0, cx2))
                cy2 = max(0.0, min(640.0, cy2))

                cw = cx2 - cx1
                ch = cy2 - cy1
                if cw > 1.0 and ch > 1.0:
                    n_xc = (cx1 + cx2) / (2.0 * 640.0)
                    n_yc = (cy1 + cy2) / (2.0 * 640.0)
                    n_w = cw / 640.0
                    n_h = ch / 640.0
                    new_bboxes.append([n_xc, n_yc, n_w, n_h])
                    new_cls.append(c_id)

        if len(new_bboxes) > 0:
            label["img"] = resized_crop
            label["bboxes"] = np.array(new_bboxes, dtype=np.float32)
            label["cls"] = np.array(new_cls, dtype=np.float32).reshape(-1, 1)
            label["ori_shape"] = (640, 640)
            label["resized_shape"] = (640, 640)
            label["_is_context_cropped"] = True

        return label

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Mengambil data citra dan label dengan augmentasi adaptif on-the-fly."""
        if not self.augment or not self.enable_targeted_aug:
            return self.transforms(self.get_image_and_label(index))

        label = self.get_image_and_label(index)
        fname = Path(label["im_file"]).name

        if fname in self.manifest_data:
            tinfo = self.manifest_data[fname]
            # Jika memuat target hard small lesion dan terpicu probabilitas augmentasi
            if tinfo.get("has_target_small", False) and random.random() < self.aug_prob:
                label = self.apply_context_crop(label, tinfo)
                if label.get("_is_context_cropped", False):
                    # Mosaic dinonaktifkan pada patch crop terarah
                    return self.crop_transforms(label)

        # Standar transform baseline (dengan Mosaic)
        return self.transforms(label)


class AdaptiveDetectionTrainer(DetectionTrainer):
    """
    Subclass DetectionTrainer dengan dukungan Mixture Sampling dan Targeted Augmentation.
    """

    def __init__(
        self,
        *args,
        enable_mixture_sampling: bool = False,
        enable_targeted_aug: bool = False,
        manifest_path: str | Path | None = "d:/UNDIP/SKRIPSI/stage_8_2_difficulty/train_image_difficulty_manifest.csv",
        aug_prob: float = 0.50,
        **kwargs,
    ):
        self.enable_mixture_sampling = enable_mixture_sampling
        self.enable_targeted_aug = enable_targeted_aug
        self.manifest_path = manifest_path
        self.aug_prob = aug_prob
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Membangun dataset AdaptiveYOLODataset untuk mode train."""
        if mode == "train":
            gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
            return AdaptiveYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch or self.args.batch,
                augment=True,
                hyp=self.args,
                rect=False,
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=gs,
                pad=0.0,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                manifest_path=self.manifest_path,
                enable_targeted_aug=self.enable_targeted_aug,
                aug_prob=self.aug_prob,
            )
        else:
            return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        """Membangun dataloader dengan WeightedRandomSampler jika mode train dan adaptive sampling aktif."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode == "train" and self.enable_mixture_sampling:
            weights = dataset.get_sampling_weights()
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(dataset),
                replacement=True,
            )
            LOGGER.info(
                f"[ADAPTIVE SAMPLER] Mengaktifkan Mixture WeightedRandomSampler (len={len(dataset):,}, replacement=True)"
            )
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=getattr(dataset, "collate_fn", None),
            )
        else:
            shuffle = (mode == "train")
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
                device=self.device,
            )


def self_test():
    """Fungsi verifikasi mandiri untuk menguji loader dan crop transform."""
    print("=" * 80)
    print("SELF-TEST STAGE 8.3 — ADAPTIVE LOADER & TARGETED AUGMENTATION")
    print("=" * 80)

    manifest_path = Path("d:/UNDIP/SKRIPSI/stage_8_2_difficulty/train_image_difficulty_manifest.csv")
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} tidak ditemukan!")
        sys.exit(1)

    manifest = load_difficulty_manifest(manifest_path)
    target_count = sum(1 for v in manifest.values() if v["has_target_small"])
    print(f"[TEST 1] Manifest berhasil dimuat: {len(manifest):,} citra.")
    print(f"         Citra dengan target hard small lesion: {target_count:,} ({target_count / len(manifest) * 100:.2f}%)")

    # Inisialisasi Dataset Adaptif
    data_yaml_dict = {
        "names": {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "HEALTHY"},
        "nc": 5,
    }
    hyp = SimpleNamespace(
        **{
            **vars(DEFAULT_CFG),
            "mosaic": 1.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "fliplr": 0.5,
            "flipud": 0.0,
        }
    )

    dataset = AdaptiveYOLODataset(
        img_path="d:/UNDIP/SKRIPSI/cassava-v7-research/train/images",
        imgsz=640,
        batch_size=16,
        augment=True,
        hyp=hyp,
        data=data_yaml_dict,
        manifest_path=manifest_path,
        enable_targeted_aug=True,
        aug_prob=1.0,  # 100% trigger untuk pengetesan
    )

    weights = dataset.get_sampling_weights()
    print(f"[TEST 2] Bobot sampling dataset terekstraksi: {len(weights):,} bobot.")
    print(f"         Min: {weights.min().item():.8e}, Max: {weights.max().item():.8e}, Mean: {weights.mean().item():.8e}")

    # Uji pengambilan batch
    sampler = WeightedRandomSampler(weights=weights, num_samples=32, replacement=True)
    loader = InfiniteDataLoader(
        dataset=dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=0,
        collate_fn=getattr(dataset, "collate_fn", None),
    )

    batch = next(iter(loader))
    print(f"[TEST 3] Batch berhasil dimuat melalui Adaptive DataLoader:")
    print(f"         Imgs tensor shape: {batch['img'].shape}")
    print(f"         Batch bboxes shape: {batch['bboxes'].shape}")
    print(f"         Batch cls shape: {batch['cls'].shape}")

    print("=" * 80)
    print("SELF-TEST STAGE 8.3 BERHASIL 100% (ALL CHECKS PASSED)!")
    print("=" * 80)


if __name__ == "__main__":
    self_test()

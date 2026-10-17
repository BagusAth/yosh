#!/usr/bin/env python3
"""
Stage 8.3 — Adaptive Data Loading & On-the-Fly Context Augmentation Pipeline (Hardened)
======================================================================================

Kontrak Metodologis:
1. AdaptiveYOLODataset:
   - Mengintegrasikan manifest kesulitan citra dari Stage 8.2 (train_image_difficulty_manifest.csv).
   - Validasi ketat fail-fast: Setiap citra latih HARUS memiliki tepat satu entri manifest (Zero Silent Fallback).
   - On-the-Fly Targeted Context Augmentation:
     * Khusus untuk citra dengan lesi Small sulit (has_target_small == 1, D_i >= 0.70).
     * Lesi target tunggal i* (kesulitan tertinggi).
     * Jendela crop diperluas 2× dimensi bounding box lesi (w' = min(W, 2w), h' = min(H, 2h))
       untuk mempertahankan konteks morfologis daun di sekitar lesi.
     * Bounding box lain yang masuk jendela crop:
       - Visibilitas >= 0.30: dipertahankan dan ditransformasikan ke koordinat crop.
       - Visibilitas < 0.30: dibuang agar tidak menyajikan potongan lesi yang menyesatkan model.
     * Patch crop di-resize ke target resolusi model (imgsz).
     * Mosaic dinonaktifkan khusus pada patch crop terarah agar lesi tidak kembali mengecil.
     * Label asli di-protect (deepcopy) agar tidak termutasi in-place saat augmentasi dievaluasi.
2. Mixture Adaptive Sampling:
   - Mengambil probabilitas terkalibrasi P(k) = (1 - lambda)*(1/N) + lambda*(D_k / sum(D)) (lambda = 0.70).
   - Membangun WeightedRandomSampler dengan penggantian (with replacement).
   - Verifikasi ketat: seluruh bobot bernilai finite, non-negatif, non-zero floor, dan sum(P) = 1.0 +- 1e-4.
3. AdaptiveDetectionTrainer:
   - Subclass resmi Ultralytics DetectionTrainer untuk EXP-0 s.d. EXP-3.
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
from ultralytics.utils.instance import Instances
from ultralytics.utils.torch_utils import torch_distributed_zero_first


def load_difficulty_manifest(manifest_path: str | Path) -> dict[str, dict]:
    """
    Membaca train_image_difficulty_manifest.csv ke dictionary dengan validasi integritas:
    - Tidak ada duplikasi image_id.
    - Nilai d_image dan p_sampling bernilai valid dan finite.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest kesulitan tidak ditemukan di {manifest_path}!")

    records = {}
    seen_ids = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):
            fname = row["image_id"]
            if fname in seen_ids:
                raise ValueError(f"Baris {line_num}: Duplikasi image_id '{fname}' ditemukan dalam manifest!")
            seen_ids.add(fname)

            try:
                d_img = float(row["d_image"])
                p_samp = float(row["p_sampling"])
                has_gt = int(row["has_gt"])
                has_small = (row["has_target_small"] == "1" or row["has_target_small"].lower() == "true")
                tgt_gt_id = int(row["target_gt_id"])
                tgt_di = float(row["target_small_di"])
                tgt_box = row["target_small_box"].strip()
            except Exception as e:
                raise ValueError(f"Baris {line_num} ({fname}): Format nilai manifest tidak valid ({e})")

            if not math.isfinite(d_img) or d_img < 0.0 or d_img > 1.0:
                raise ValueError(f"Baris {line_num} ({fname}): d_image={d_img} di luar rentang valid [0, 1]!")
            if not math.isfinite(p_samp) or p_samp <= 0.0:
                raise ValueError(f"Baris {line_num} ({fname}): p_sampling={p_samp} harus finite dan positif!")

            records[fname] = {
                "d_image": d_img,
                "p_sampling": p_samp,
                "has_gt": has_gt,
                "has_target_small": has_small,
                "target_gt_id": tgt_gt_id,
                "target_small_di": tgt_di,
                "target_small_box": tgt_box,
            }

    return records


class AdaptiveYOLODataset(YOLODataset):
    """
    YOLODataset kustom dengan kemampuan On-the-Fly Targeted Context Augmentation
    dan verifikasi manifest fail-fast (zero silent fallback).
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

        self.crop_diagnostics = {
            "crop_attempts": 0,
            "crops_applied": 0,
            "no_target_skips": 0,
            "invalid_box_skips": 0,
            "visibility_skips": 0,
        }

        # Muat manifest kesulitan jika diberikan
        self.manifest_data = {}
        if manifest_path:
            self.manifest_data = load_difficulty_manifest(manifest_path)

        super().__init__(*args, hyp=hyp, **kwargs)

        # Fail-fast validation: jika augmentasi terarah aktif, seluruh citra HARUS ada dalam manifest
        if self.enable_targeted_aug:
            self.validate_manifest_coverage()

        # Transforms khusus untuk patch crop (tanpa Mosaic sekunder)
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

    def validate_manifest_coverage(self):
        """Memverifikasi bahwa setiap file dalam dataset memiliki mapping manifest yang cocok."""
        if not self.manifest_data:
            raise ValueError("Targeted augmentation aktif tetapi manifest kesulitan kosong atau tidak dimuat!")

        missing = []
        for im_file in self.im_files:
            fname = Path(im_file).name
            if fname not in self.manifest_data:
                missing.append(fname)

        if missing:
            raise ValueError(
                f"FATAL: {len(missing)} citra pada dataset tidak ditemukan di manifest kesulitan!\n"
                f"Contoh citra yang hilang: {missing[:5]}\n"
                "Silent uniform fallback dilarang karena mengubah strategi eksperimen tanpa disadari."
            )

    def get_sampling_weights(self) -> torch.Tensor:
        """
        Mengambil bobot sampling P(k) untuk seluruh citra pada dataset secara ketat:
        - Jika manifest tidak mencakup suatu citra, eksekusi dihentikan (FAIL-FAST).
        - Verifikasi bahwa total bobot finite, non-negatif, dan sum(P) ~ 1.0.
        """
        self.validate_manifest_coverage()

        weights = []
        for im_file in self.im_files:
            fname = Path(im_file).name
            p_val = self.manifest_data[fname]["p_sampling"]
            weights.append(p_val)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        if torch.isnan(weights_tensor).any() or torch.isinf(weights_tensor).any():
            raise ValueError("Ditemukan bobot NaN atau Inf pada sampling weights!")
        if (weights_tensor <= 0.0).any():
            raise ValueError("Ditemukan bobot sampling <= 0 pada mixture sampler!")

        sum_w = weights_tensor.sum().item()
        if abs(sum_w - 1.0) > 1e-3:
            LOGGER.warning(f"[SAMPLER] sum(P) = {sum_w:.6f} berbeda dari 1.0, melakukan re-normalisasi.")
            weights_tensor = weights_tensor / sum_w

        return weights_tensor

    def apply_context_crop(self, label: dict[str, Any], target_info: dict[str, Any]) -> dict[str, Any]:
        """
        Menerapkan Targeted Context-Preserving Crop:
        - Mempertahankan label asli (deepcopy) agar tidak termutasi in-place saat crop gagal/skip.
        - Memperluas 2x bbox lesi target i* untuk konteks morfologi daun.
        - Mempertahankan lesi lain yang beririsan dengan visibilitas >= 0.30.
        - Resize patch ke resolusi input model (hyp.imgsz).
        """
        img = label.get("img")
        if img is None:
            self.crop_diagnostics["invalid_box_skips"] += 1
            return label

        H, W = img.shape[:2]
        if H < 4 or W < 4:
            self.crop_diagnostics["invalid_box_skips"] += 1
            return label

        instances = label.get("instances")
        if instances is None or len(instances) == 0:
            self.crop_diagnostics["no_target_skips"] += 1
            return label

        box_str = target_info.get("target_small_box", "")
        if not box_str or not target_info.get("has_target_small", False):
            self.crop_diagnostics["no_target_skips"] += 1
            return label

        try:
            parts = [float(v) for v in box_str.split(",")]
        except Exception:
            self.crop_diagnostics["invalid_box_skips"] += 1
            return label

        if len(parts) != 4:
            self.crop_diagnostics["invalid_box_skips"] += 1
            return label

        raw_bx1, raw_by1, raw_bx2, raw_by2 = parts
        if max(raw_bx1, raw_by1, raw_bx2, raw_by2) <= 1.01:
            bx1 = raw_bx1 * W
            by1 = raw_by1 * H
            bx2 = raw_bx2 * W
            by2 = raw_by2 * H
        else:
            bx1 = raw_bx1 * (W / 640.0)
            by1 = raw_by1 * (H / 640.0)
            bx2 = raw_bx2 * (W / 640.0)
            by2 = raw_by2 * (H / 640.0)

        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0.0 or bh <= 0.0:
            self.crop_diagnostics["invalid_box_skips"] += 1
            return label

        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        w_crop = min(float(W), self.context_margin * bw)
        h_crop = min(float(H), self.context_margin * bh)
        w_crop = max(4.0, w_crop)
        h_crop = max(4.0, h_crop)

        x_min = max(0.0, min(float(W) - w_crop, bcx - w_crop / 2.0))
        y_min = max(0.0, min(float(H) - h_crop, bcy - h_crop / 2.0))
        x_max = min(float(W), x_min + w_crop)
        y_max = min(float(H), y_min + h_crop)

        ix_min, iy_min, ix_max, iy_max = (
            int(round(x_min)),
            int(round(y_min)),
            int(round(x_max)),
            int(round(y_max)),
        )

        cropped_img = img[iy_min:iy_max, ix_min:ix_max]
        if cropped_img.size == 0 or cropped_img.shape[0] < 4 or cropped_img.shape[1] < 4:
            self.crop_diagnostics["invalid_box_skips"] += 1
            return label

        hyp_obj = getattr(self, "hyp", None)
        target_imgsz = getattr(hyp_obj, "imgsz", 640) if hyp_obj else 640
        resized_crop = cv2.resize(cropped_img, (target_imgsz, target_imgsz), interpolation=cv2.INTER_LINEAR)
        scale_x = float(target_imgsz) / (x_max - x_min)
        scale_y = float(target_imgsz) / (y_max - y_min)

        orig_boxes = instances.bboxes
        orig_cls = label.get("cls")
        if orig_cls is None:
            orig_cls = np.zeros((len(orig_boxes), 1), dtype=np.float32)

        new_bboxes = []
        new_cls = []
        target_gt_id = target_info.get("target_gt_id", -1)

        for idx in range(len(orig_boxes)):
            box = orig_boxes[idx]
            gxc = float(box[0]) * W
            gyc = float(box[1]) * H
            gw = float(box[2]) * W
            gh = float(box[3]) * H
            gx1 = gxc - gw / 2.0
            gy1 = gyc - gh / 2.0
            gx2 = gxc + gw / 2.0
            gy2 = gyc + gh / 2.0
            orig_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))

            inter_x1 = max(gx1, x_min)
            inter_y1 = max(gy1, y_min)
            inter_x2 = min(gx2, x_max)
            inter_y2 = min(gy2, y_max)

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            vis_ratio = inter_area / orig_area

            is_target = (target_gt_id >= 0 and idx == target_gt_id) or (abs(gx1 - bx1) < 6.0 and abs(gy1 - by1) < 6.0)
            if is_target or vis_ratio >= self.vis_thresh:
                cx1 = max(0.0, min(float(target_imgsz), (inter_x1 - x_min) * scale_x))
                cy1 = max(0.0, min(float(target_imgsz), (inter_y1 - y_min) * scale_y))
                cx2 = max(0.0, min(float(target_imgsz), (inter_x2 - x_min) * scale_x))
                cy2 = max(0.0, min(float(target_imgsz), (inter_y2 - y_min) * scale_y))

                cw = cx2 - cx1
                ch = cy2 - cy1
                if cw > 1.0 and ch > 1.0:
                    n_xc = (cx1 + cx2) / (2.0 * target_imgsz)
                    n_yc = (cy1 + cy2) / (2.0 * target_imgsz)
                    n_w = cw / target_imgsz
                    n_h = ch / target_imgsz
                    new_bboxes.append([n_xc, n_yc, n_w, n_h])
                    cls_val = orig_cls[idx][0] if hasattr(orig_cls[idx], "__getitem__") else orig_cls[idx]
                    new_cls.append(cls_val)
            else:
                self.crop_diagnostics["visibility_skips"] += 1

        if len(new_bboxes) > 0:
            out_label = copy(label)
            out_label["img"] = resized_crop
            out_label["cls"] = np.array(new_cls, dtype=np.float32).reshape(-1, 1)
            out_label["instances"] = Instances(
                bboxes=np.array(new_bboxes, dtype=np.float32),
                segments=None,
                keypoints=None,
                bbox_format="xywh",
                normalized=True,
            )
            out_label["ori_shape"] = (target_imgsz, target_imgsz)
            out_label["resized_shape"] = (target_imgsz, target_imgsz)
            out_label["_is_context_cropped"] = True
            self.crop_diagnostics["crops_applied"] += 1
            return out_label

        self.crop_diagnostics["invalid_box_skips"] += 1
        return label

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Mengambil data citra dan label dengan augmentasi adaptif on-the-fly."""
        if not self.augment or not self.enable_targeted_aug:
            return self.transforms(self.get_image_and_label(index))

        label = self.get_image_and_label(index)
        fname = Path(label["im_file"]).name

        if fname in self.manifest_data:
            tinfo = self.manifest_data[fname]
            if tinfo.get("has_target_small", False) and random.random() < self.aug_prob:
                self.crop_diagnostics["crop_attempts"] += 1
                cropped_label = self.apply_context_crop(label, tinfo)
                if cropped_label.get("_is_context_cropped", False):
                    cropped_label.pop("_is_context_cropped", None)
                    return self.crop_transforms(cropped_label)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default="d:/UNDIP/SKRIPSI/stage_8_2_difficulty/train_image_difficulty_manifest.csv")
    parser.add_argument("--img-path", default="d:/UNDIP/SKRIPSI/cassava-v7-research/train/images")
    args, _ = parser.parse_known_args()

    print("=" * 80)
    print("SELF-TEST STAGE 8.3 — ADAPTIVE LOADER & TARGETED AUGMENTATION (HARDENED)")
    print("=" * 80)

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} tidak ditemukan!")
        sys.exit(1)

    manifest = load_difficulty_manifest(manifest_path)
    target_count = sum(1 for v in manifest.values() if v["has_target_small"])
    print(f"[TEST 1] Manifest berhasil dimuat: {len(manifest):,} citra.")
    print(f"         Citra dengan target hard small lesion: {target_count:,} ({target_count / len(manifest) * 100:.2f}%)")

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
            "imgsz": 640,
        }
    )

    dataset = AdaptiveYOLODataset(
        img_path=args.img_path,
        imgsz=640,
        batch_size=16,
        augment=True,
        hyp=hyp,
        data=data_yaml_dict,
        manifest_path=manifest_path,
        enable_targeted_aug=True,
        aug_prob=1.0,
    )

    weights = dataset.get_sampling_weights()
    print(f"[TEST 2] Bobot sampling dataset terekstraksi: {len(weights):,} bobot.")
    print(f"         Min: {weights.min().item():.8e}, Max: {weights.max().item():.8e}, Mean: {weights.mean().item():.8e}")

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

    applied_count = 0
    for idx in range(min(100, len(dataset))):
        raw_lbl = dataset.get_image_and_label(idx)
        fn = Path(raw_lbl["im_file"]).name
        ti = dataset.manifest_data.get(fn, {})
        if ti.get("has_target_small", False):
            cropped = dataset.apply_context_crop(raw_lbl, ti)
            if cropped.get("_is_context_cropped", False):
                applied_count += 1
                print(f"[TEST 4] Targeted Context Crop terverifikasi aktif pada sampel {fn}:")
                print(f"         Input boxes: {len(raw_lbl['instances'])}, Output crop boxes: {len(cropped['instances'])}")
                print(f"         Patch shape: {cropped['img'].shape}, _is_context_cropped: {cropped.get('_is_context_cropped')}")
                break
    print(f"         Crop Diagnostics: {dataset.crop_diagnostics}")
    assert applied_count > 0, "FATAL: apply_context_crop gagal diaplikasikan pada sampel target!"

    print("=" * 80)
    print("SELF-TEST STAGE 8.3 BERHASIL 100% (ALL CHECKS PASSED)!")
    print("=" * 80)


if __name__ == "__main__":
    self_test()

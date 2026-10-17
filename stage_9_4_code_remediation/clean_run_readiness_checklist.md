# Clean Run Readiness Checklist — Stage 9.4

Tanggal Audit: 2026-10-17  
Auditor: Senior ML Research Engineer, Computer Vision Engineer, and Independent Code Auditor  
Target Pipeline: Clean 4-Fold OOF Profiling & Clean Ablation Experiments (EXP-0 s.d. EXP-3)  

---

## 1. Status Dataset & Anotasi

- [x] **Homogenitas Format Anotasi**: Seluruh citra pada dataset bersih `cassava-v7-standardized` memiliki anotasi kanonis tepat 5 kolom YOLO (`class_id xc yc w h`).
- [x] **Integritas Koordinat**: Koordinat ternormalisasi dalam rentang valid $[0.0, 1.0]$, nilai finite, dan dimensi box positif ($w > 0, h > 0$).
- [x] **Jumlah Citra & Label Sesuai**:
  - Train: 8.980 citra & 8.980 label
  - Validation: 2.567 citra & 2.567 label
  - Test: 1.282 citra & 1.282 label
  - Total: 12.829 citra
- [x] **Pelestarian Dataset Historis**: Dataset `cassava-v7-research` dipertahankan tanpa perubahan fisik sebagai artefak komparasi historis.

---

## 2. Status OOF Profiling (Stage 8.1)

- [x] **Nested Group-Aware Holdout Isolation**:
  - Prosedur OOF telah dirombak menggunakan skema nested split (`nested_inner_val`).
  - Outer Holdout fold $k$ (1/4 data latih) **100% diisolasi** dari `fold_k_data.yaml`.
  - Checkpoint selection (`best.pt`) murni dipilih berdasarkan Inner-Validation (1/4 data latih dari sisa fold), dan dilatih pada Inner-Train (2/4 data latih).
  - Checkpoint tersebut digunakan secara eksklusif untuk inferensi diagnostik pada Outer Holdout $k$.
- [x] **Zero Cross-Fold Group Leakage**: Tidak ada grup atomik yang beririsan antara Outer Holdout, Inner-Val, dan Inner-Train (terverifikasi 0% leakage).
- [x] **Zero Holdout Label Snooping**: Pemilihan checkpoint probe model tidak lagi dipengaruhi oleh label fold holdout.
- [x] **Artifact Checksum Verification**: Menolak penggunaan ulang checkpoint lama jika checksum kode, data YAML, atau konfigurasi hyperparameter tidak identik.

---

## 3. Status Evaluasi & Difficulty Manifest (Stage 8.2)

- [x] **Modul Evaluasi Terpadu (SSOT)**: Seluruh metrik evaluasi diturunkan dari `research_utils/detection_metrics.py`.
- [x] **Global One-to-One Bipartite Matching**: Menggunakan pengurutan IoU kandidat menurun secara global (zero prediction reuse, zero ground-truth reuse).
- [x] **Formula Kesulitan Kanonis**:
  - Matched + Correct Class: $D_i = 0.40(1 - C_i) + 0.40(1 - \text{IoU}_i) + 0.20 S_i$
  - Matched + Wrong Class: $D_i = \min(1.0, 0.40(1 - C_i) + 0.40(1 - \text{IoU}_i) + 0.20 S_i + 0.35)$
  - Unmatched FN: $D_i = 0.80 + 0.20 S_i$
  - Citra berobjek: $D_\text{image} = 0.5 \max(D_i) + 0.5 \text{mean}(D_i)$
  - Citra kosong: $D_\text{image} = \max(C_\text{FP})$ jika ada false positive, $0.0$ jika bersih.
- [x] **Mixture Sampling Probability**: $P(k) = (1 - \lambda)/N + \lambda (D_k / \sum D)$ dengan $\lambda = 0.70$ eksplisit, floor uniform $> 0$, dan $\sum P = 1.0 \pm 10^{-5}$.
- [x] **Fail-Fast Manifest Coverage**: Validasi ketat memicu `ValueError` jika terdapat citra yang hilang atau duplikasi dalam manifest (zero silent fallback).

---

## 4. Status Dataloader & Augmentasi Adaptif (Stage 8.3)

- [x] **Targeted Context Augmentation Contract**:
  - Crop diperluas $2\times$ dimensi bounding box lesi target $i^*$ (kesulitan tertinggi, $D_i \ge 0.70$).
  - Clamping aman pada batas kanvas $[0, W] \times [0, H]$.
  - Lesi sekunder dipertahankan jika rasio visibilitas $\ge 0.30$; dibuang jika $< 0.30$.
  - Patch di-resize ke target resolusi input (`imgsz`).
  - Mosaic dinonaktifkan khusus pada patch crop terarah.
- [x] **Label Immutability**: Struktur label asli di-protect (deepcopy) dan tidak termutasi in-place saat augmentasi dievaluasi atau di-skip.

---

## 5. Status Eksperimen Ablasi & Pelestarian Historis (Stage 8.4)

- [x] **Isolasi Output Bersih**: Output default eksperimen bersih dialihkan ke direktori baru `stage_10_11_clean_experiments/`.
- [x] **Anti-Overwrite Guard**: Penulisan ke direktori historis `stage_8_ablation/` diblokir secara otomatis dengan `PermissionError`.
- [x] **Matriks Eksperimen Apple-to-Apple**:
  - EXP-0: Baseline Re-run (Uniform 1/N, Standard Aug)
  - EXP-1: Mixture Adaptive Sampling (P(k), Standard Aug)
  - EXP-2: Targeted Context Augmentation (Uniform 1/N, On-the-Fly Crop)
  - EXP-3: Full Proposed System (P(k) Sampling + On-the-Fly Crop)
- [x] **Pemisahan Kanonis Metrik Ultralytics**:
  - `box.ap50` untuk metrik AP@50 per kelas.
  - `box.maps` untuk metrik AP@50-95 per kelas.
  - Serialization mempertahankan presisi penuh `float` tanpa pembulatan destruktif.

---

## 6. Status Seleksi Model & Proteksi Test Set (Stage 8.5 & 8.6)

- [x] **Hierarki Seleksi Model Terkunci (Decision Rule)**:
  - Kriteria Utama: Validasi mAP@50-95 tertinggi.
  - Tie-break 1: Recall Small ($< 32^2$) tertinggi (hanya jika tie numerik selisih $\le 10^{-5}$).
  - Tie-break 2: Validasi mAP@50 tertinggi.
  - Tie-break 3: Total False Negative (`fn_missed`) terendah.
  - Margin arbitrer $0.5\%$ telah dihapus sepenuhnya.
- [x] **Zero Test Set Snooping**: Seleksi model murni 100% bergantung pada Validation Set (2.567 citra).
- [x] **Proteksi Locked Test Set (Stage 8.6)**: Eksekusi final benchmark pada test set diblokir oleh guard `--confirm-final-evaluation` untuk mencegah unconfirmed data snooping selama tahap perbaikan kode.

---

## 7. Status Regression Test (Stage 9.4)

- [x] **Test Suite Master**: `test_stage_9_4_regression.py` mencakup 13 test suite (26 unit/integration tests).
- [x] **Hasil Eksekusi Aktual**: 26 passed, 0 failed, 0 errors (100% PASS dalam 0.14 detik).
- [x] **Backward Compatibility**: `test_stage_9_3_regression.py` lulus 100% (16 tests passed).

---

## Keputusan Kesiapan (Gate Decision)

Berdasarkan pemenuhan seluruh kriteria di atas:
- Defect kritis OOF checkpoint leakage: **REMEDIATED**.
- Path dataset inkonsisten: **REMEDIATED (SSOT via data.yaml)**.
- Discrepancy evaluator & matching: **REMEDIATED via research_utils**.
- Seleksi model tidak konsisten: **REMEDIATED (Locked Protocol)**.
- Regression suite: **100% PASS**.

**Status Gerbang Akhir: `READY_FOR_CLEAN_OOF`**
*(Pipeline kode telah terverifikasi aman, terisolasi, dan siap untuk memulai OOF profiling bersih pada dataset terstandarisasi).*

# STAGE 9.7 — PREFLIGHT DRY-RUN & ZERO-MUTATION VERIFICATION SUMMARY

**Tanggal Eksekusi**: 2026-10-19 21:19:45  
**Baseline HEAD**: `6e92427d70a93c45f0d3f5cc0853d65753a48ea7`  
**Working Branch**: `stage-9-7-readiness-gate-hardening`  
**Eksekutor**: Senior Machine Learning Engineer, Computer Vision Research Engineer & Test Auditor  

---

## 1. Tujuan Preflight Dry-Run

Membuktikan secara empiris bahwa:
1. Skrip orkestrasi Stage 8.1 dan Stage 8.4 dapat membaca dataset kanonis `cassava-v7-standardized/data.yaml` dan memvalidasi seluruh preflight tanpa crash.
2. Tidak ada operasi penulisan ke direktori dataset (`cassava-v7-standardized` maupun `cassava-v7-research`).
3. Tidak ada penulisan atau penimpaan ke direktori artefak historis (`stage_8_1_oof`, `stage_8_2_difficulty`, `stage_8_ablation`, `stage_7_baseline`).
4. Tidak ada pembuatan file `.cache` baru di direktori dataset penelitian.
5. Seluruh output uji coba terisolasi di dalam `stage_9_7_readiness_remediation/dry_run_evidence/`.

---

## 2. Rincian Eksekusi Dry-Run

### 2.1. Stage 8.1 — Group-Aware 4-Fold OOF Profiling Dry-Run

```powershell
python stage_8_1_group_oof_profiling.py `
  --data-yaml cassava-v7-standardized/data.yaml `
  --image-index cassava-v7-stage-5/clean_image_index.csv `
  --output-dir stage_9_7_readiness_remediation/dry_run_evidence/oof_dry_run `
  --dry-run
```

- **Exit Code**: `0` (SUCCESS)
- **Ringkasan Output**:
  - Total citra latih aktual: 8.980 (Expected: 8.980, Cocok 100%)
  - Empty label (0 box / background): 66 citra
  - Pure HEALTHY bbox only: 2.202 citra
  - Pure Disease bbox (CBB..CMD): 6.704 citra
  - Mixed (HEALTHY + Disease): 8 citra
  - Pemetaan grup atomik: 5.546 grup
  - Partisi 4-Fold:
    - Fold 1: 1.388 grup, 2.246 citra
    - Fold 2: 1.387 grup, 2.244 citra
    - Fold 3: 1.386 grup, 2.245 citra
    - Fold 4: 1.385 grup, 2.245 citra
  - Verifikasi kebocoran grup (Cross-Fold Leakage): **0 (100% PASS)**
  - Isolasi Outer Holdout vs Inner Train/Val: **100% PASS**
  - Path dataset terstandarisasi: **True (Zero research path leakage)**

---

### 2.2. Stage 8.4 — Ablation Study Preflight Dry-Run (EXP-0)

```powershell
python stage_8_4_run_ablation_experiments.py `
  --data-yaml cassava-v7-standardized/data.yaml `
  --exp 0 `
  --output-dir stage_9_7_readiness_remediation/dry_run_evidence/ablation_dry_run `
  --dry-run
```

- **Exit Code**: `0` (SUCCESS)
- **Ringkasan Output**:
  - Resolusi Data YAML: `cassava-v7-standardized/data.yaml`
  - Kontrol Baseline: EXP-0 (Uniform 1/N sampling, standard YOLO11n augmentation)
  - Hashing Dataset Inventory: Berhasil meng-hash seluruh 8.980 citra train dan 2.567 citra val
  - Pretrained Weights Checksum: Berhasil memverifikasi `yolo11n.pt` (SHA-256: `0ebbc80d...`)
  - Status Validasi: `[EXP-0] DRY-RUN PASS: Validasi preflight, dataset fingerprint, dan konfigurasi lolos 100%.`

---

## 3. Bukti Zero-Mutation Dataset

Pemeriksaan status git dan integritas direktori dataset segera setelah eksekusi dry-run:

```
git status --porcelain cassava-v7-standardized/ cassava-v7-research/
```

- **Perubahan File**: `0` (Tidak ada file diubah)
- **File Baru / Untracked**: `0` (Tidak ada file baru dibuat)
- **File Cache (.cache)**: `0` (Tidak ada `train/labels.cache` baru dibuat di folder dataset)
- **Status Integritas**: **100% TIDAK TERMUTASI (ZERO MUTATION CONFIRMED)**

---

## 4. Kesimpulan

Kedua modul orkestrasi inti (Stage 8.1 dan Stage 8.4) terbukti aman, membaca dataset penelitian secara *read-only*, tidak menimbulkan kebocoran jalur historis, dan siap dieksekusi dalam mode clean.

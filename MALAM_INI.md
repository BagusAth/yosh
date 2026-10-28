# Runbook GPU (Pakem v1.3)

Shell Claude di laptop ini berjalan di mesin virtual Linux tanpa akses GPU, jadi training dijalankan dari PowerShell Windows.

## Malam 1 — satu perintah

```powershell
cd D:\UNDIP\SKRIPSI
powershell -ExecutionPolicy Bypass -File .\jalankan_malam_1.ps1
```

Isinya, berurutan (berhenti otomatis bila ada yang gagal):

1. Uji regresi Fase 1 dan Stage 9.8 (CPU, sekitar 1–2 menit). Tiga tes Group B yang butuh torch baru berjalan di sini.
2. Manifest kesulitan v2 (CPU, sekitar 3–5 menit) ke `stage_v2_difficulty_20261028`.
3. EXP-0 seed 123 dan 2026 (GPU, sekitar 1,5 jam) ke `stage_v2_ablation/seed_123` dan `seed_2026`.
4. E-LEAK simulasi untuk RQ1 (GPU, sekitar 1 jam): latih baseline pada split acak per citra (`eleak_sim_rq1`), evaluasi subset bocor vs pembanding, lalu laporan `stage_v2_eleak_sim/eval_subsets/ELEAK_REPORT.md`.

Log tersimpan di `logs_v2\`.

## Malam 2–3 — ablasi multi-seed (12 run, sekitar 9–10 jam)

Dijalankan setelah manifest v2 dari Malam 1 ada.

```powershell
foreach ($s in 42, 123, 2026) {
  foreach ($e in 4, 1, 2, 3) {
    .\.venv-cuda\Scripts\python.exe stage_8_4_run_ablation_experiments.py --exp $e --seed $s `
      --data-yaml D:/UNDIP/SKRIPSI/cassava-v7-runtime-mirror/data.yaml `
      --manifest-path D:/UNDIP/SKRIPSI/stage_v2_difficulty_20261028/train_image_difficulty_manifest.csv `
      --pretrained-weights D:/UNDIP/SKRIPSI/yolo11n.pt `
      --output-dir D:/UNDIP/SKRIPSI/stage_v2_ablation/seed_$s
  }
}
```

Nomor eksperimen: 4 = EXP-0R, 1 = EXP-1, 2 = EXP-2, 3 = EXP-3. EXP-0 seed 42 memakai hasil yang sudah ada di `stage_8_ablation_clean_20261020/exp_0_baseline`.

## RQ1 (gabungan A + B)

- Bagian B (tanpa GPU, sudah selesai): `_analisis_rq1_audit/RQ1_AUDIT_FALSE_POSITIVE_REPORT.md` membuktikan audit berbasis nama file menghasilkan positif palsu; split bawaan Roboflow tidak bocor. Folder `eleak_rq1/` adalah pemeriksaan split asli yang sama (0 citra validasi bocor).
- Bagian A (GPU, langkah 4 Malam 1): simulasi split acak per citra di `eleak_sim_rq1/` (60,5% citra validasi bocor, test tidak disentuh).

## Catatan

- Colok laptop ke listrik dan matikan mode tidur selama training.
- Kalau sebuah run gagal di tengah, jalankan ulang perintah yang sama dengan `--skip-if-done`.
- Test set tidak disentuh di langkah mana pun.

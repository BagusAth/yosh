# Runbook GPU (Pakem v1.3.1)

Shell Claude di laptop ini berjalan di mesin virtual Linux tanpa akses GPU, jadi training dijalankan dari Windows dengan klik dua kali file `.bat` (atau perintah PowerShell yang sama).

## Status

- Malam 1 (uji regresi, manifest v2, EXP-0 seed 123/2026, E-LEAK): selesai 28 Oktober.
- RQ1 A + B: selesai 29 Oktober (`stage_v2_eleak_sim/per_image/analysis/`).
- Berikutnya: Fase 3, ablasi multi-seed (12 run).

## Fase 3 — tiga file, berurutan

| File | Isi | Perkiraan |
| --- | --- | --- |
| `cek_siap_fase3.bat` | Commit skrip baru, cek GPU, dry-run EXP-0R/1/2/3, prediksi per citra EXP-0 tiga seed (tanpa training) | 15–25 menit |
| `jalankan_malam_A.bat` | Seed 42 dan 123: EXP-0R, EXP-1, EXP-2, EXP-3 (8 run) + prediksi per citra | 6,5–7 jam |
| `jalankan_malam_B.bat` | Seed 2026: EXP-0R, EXP-1, EXP-2, EXP-3 (4 run) + prediksi per citra | 3,5 jam |

Kalau laptop bisa menyala sekitar 10 jam, malam B boleh langsung dijalankan setelah malam A selesai.

Ketiganya memanggil `jalankan_fase3.ps1`:

```powershell
cd D:\UNDIP\SKRIPSI
powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 42,123,2026 -CekSaja
powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 42,123 -Nama A
powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 2026 -Nama B
```

Perilaku skrip:

- Preflight gagal (GPU atau dry-run) = training tidak dimulai.
- Run yang gagal tidak menghentikan run berikutnya. Ringkasan di akhir (dan `logs_v2\fase3_*_ringkasan.csv`) menunjukkan mana yang perlu diulang.
- Aman diulang: run yang selesai dan cocok checksum-nya dilewati (`--skip-if-done`), prediksi yang sudah ada tidak diulang.
- Setelah tiap run, prediksi per citra pada 2.567 citra validasi bersih disimpan ke `stage_v2_ablation/per_image/preds_seed<seed>_<eksperimen>.csv` untuk bootstrap berpasangan di Fase 5. Prediksi EXP-0 seed 123/2026 dibuat juga; EXP-0 seed 42 disalin dari E-LEAK.
- Laptop dicegah tidur selama skrip berjalan (pengaturan Windows tidak diubah). Tetap colok ke listrik dan jangan tutup laptop.

Nomor eksperimen: 4 = EXP-0R, 1 = EXP-1, 2 = EXP-2, 3 = EXP-3. EXP-0 seed 42 memakai `stage_8_ablation_clean_20261020/exp_0_baseline`.

## Tanpa GPU (bisa dikerjakan kapan saja)

- Audit label 150 citra: buka `audit_label_150/LEMBAR_AUDIT_LABEL.xlsx`, baca sheet Panduan. Jangan buka `audit_label_150/_kunci/` sebelum selesai.
- Struktur kelas: `_analisis_struktur_kelas/STRUKTUR_KELAS_REPORT.md` (`cek_struktur_kelas.py`).

## Catatan

- Test set tidak disentuh di langkah mana pun.
- Bila ada yang gagal dua kali, kirim file log dari `logs_v2\`.

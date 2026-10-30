# Runbook GPU (Pakem v1.3.2)

Shell Claude di laptop ini berjalan di mesin virtual Linux tanpa akses GPU, jadi training dijalankan dari Windows dengan klik dua kali file `.bat` (atau perintah PowerShell yang sama).

## Status

- Malam 1 (uji regresi, manifest v2, EXP-0 seed 123/2026, E-LEAK): selesai 28 Oktober.
- RQ1 A + B: selesai 29–30 Oktober (`stage_v2_eleak_sim/per_image_v2/analysis/`).
- Malam A (seed 42 dan 123, 8 run): selesai 30 Oktober 01.49, tanpa kegagalan.
- Berikutnya: **malam B lengkap** (di bawah).

## Malam ini (30 Oktober): `jalankan_malam_B.bat`

Klik dua kali `jalankan_malam_B.bat`. Isinya sama dengan perintah ini:

```powershell
cd D:\UNDIP\SKRIPSI
powershell -ExecutionPolicy Bypass -File .\jalankan_fase3.ps1 -Seeds 2026 -Nama B -UjiBaji -Conv
```

| Urutan | Isi | Perkiraan |
| --- | --- | --- |
| 1 | Commit skrip baru, cek GPU, dry-run EXP-0R/1/2/3 dan dua konfigurasi CONV | 3–5 menit |
| 2 | Uji jalan pintas baji hitam (`uji_baji_hitam.py`), inferensi saja dengan EXP-0 tiga seed | 10–15 menit |
| 3 | Fase 3 seed 2026: EXP-0R, EXP-1, EXP-2, EXP-3 (4 run) + prediksi per citra | sekitar 3 jam |
| 4 | Fase 4 CONV: EXP-0 lalu EXP-1, seed 42, 60 epoch + prediksi per citra | sekitar 3 jam |

Total sekitar 6 jam. Bila dimulai pukul 20.00, selesai sekitar pukul 02.00.

**Yang perlu dilihat sebelum ditinggal (15–20 menit pertama):**

1. Baris preflight tidak ada yang GAGAL. Bila ada, training tidak dimulai dan ringkasan langsung muncul.
2. Uji baji menampilkan `Lintasan 1: masker baji ...`, lalu ringkasan berisi `n_W` dan `n_N`, lalu `Lintasan 2 ...` dengan hitungan citra yang terus naik.
3. Muncul `=== Training EXP-0R seed 2026 ...`. Sesudah ini laptop aman ditinggal (tetap colok listrik, jangan ditutup).

Bila uji baji gagal, training tetap jalan. Kirim log-nya dan ujinya bisa diulang terpisah: `.\.venv-cuda\Scripts\python.exe uji_baji_hitam.py`.

### Keluaran

- Fase 3: `stage_v2_ablation/seed_2026/` dan `stage_v2_ablation/per_image/preds_seed2026_*.csv`
- Uji baji: `stage_v2_baji/` berisi `preds_<model>_<varian>.csv` (18 file), `gt_cleanval.csv`, `meta_baji.csv`, `run_info.json`, `SELESAI.txt`
- CONV: `stage_v2_conv60/seed_42/exp_0_baseline/`, `stage_v2_conv60/seed_42/exp_1_adaptive_sampling/` dan `stage_v2_conv60/per_image/preds_seed42_*_e60.csv`
- Log: `logs_v2/fase3_malam_B_*.log` dan `*_ringkasan.csv`

### Uji baji hitam (pra-registrasi)

Latar: baji hitam isian rotasi ada pada sekitar 41% citra asal train Roboflow (CGM, CMD, HEALTHY) dan hanya 2,5% citra asal valid/test (CBB, CBSD). Model bisa memakainya sebagai jalan pintas kelas.

- Set W = citra validasi yang berbaji. Kriterianya kriteria sudut 24 px di `cek_struktur_kelas.py`, atau pola hitam yang cocok dengan rotasi (menangkap rotasi kecil sekitar 1°). Set N = sisanya.
- Varian set N: asli, diberi baji hitam, dan diberi baji berbentuk sama berwarna rata-rata citra (kontrol oklusi). Sudut baji dipinjam dari citra W.
- Varian set W: asli, baji diganti warna rata-rata, dan baji di-inpaint.
- Kontras primer: pada citra N berkelas CBB/CBSD, selisih AP50–95 per kelas dan selisih porsi keyakinan (conf ≥ 0,25) yang jatuh ke CGM/CMD/HEALTHY, baji hitam − baji rata.
- Kontras sekunder: pada citra W, asli − rata dan asli − inpaint.
- Keputusan memakai aturan pakem: "terbukti" bila tanda selisih sama di ketiga seed dan IK 95% (bootstrap klaster per grup, 1.000 iterasi) tidak memuat nol.
- Cek kewarasan: prediksi varian `N_asli`/`W_asli` harus sama dengan `stage_v2_ablation/per_image/preds_seed*_exp_0_baseline.csv` untuk citra yang sama.
- Catatan: sebagian citra N punya pinggiran bergaris (isian rotasi berupa piksel tepi yang diulang, bukan hitam). Ini tidak memengaruhi kontras hitam − rata, tetapi layak dihitung sebagai temuan struktur data.

### CONV (Fase 4)

Resepnya sama dengan Fase 3 (batch 16, imgsz 640, patience 10, close_mosaic 6); hanya jumlah epoch yang berubah menjadi 60. Karena jadwal learning rate mengikuti jumlah epoch, run 60 epoch bukan perpanjangan run 30 epoch, melainkan run baru.

Yang dibaca:
1. mAP50–95 validasi terbaik pada 30 vs 60 epoch untuk EXP-0 dan EXP-1.
2. Apakah selisih EXP-1 − EXP-0 menyempit, tetap, atau melebar.
3. Bentuk kurva di `results.csv`, yaitu apakah sudah mendatar.

CONV hanya satu seed, jadi dilaporkan secara deskriptif dan tidak diuji dengan aturan keputusan tiga seed.

## Perilaku skrip `jalankan_fase3.ps1`

- Preflight gagal (GPU atau dry-run) = training tidak dimulai.
- Run yang gagal tidak menghentikan run berikutnya. Ringkasan di akhir (dan `logs_v2\fase3_*_ringkasan.csv`) menunjukkan mana yang perlu diulang.
- Aman diulang: run yang selesai dan cocok checksum-nya dilewati (`--skip-if-done`), prediksi yang sudah ada tidak diulang, dan uji baji dilewati bila `stage_v2_baji/SELESAI.txt` sudah ada.
- Prediksi memakai NMS seperti val() (multi_label=True, tanpa batas waktu).
- Keluaran Python ikut tercatat di log `logs_v2\fase3_*.log`.
- Laptop dicegah tidur selama skrip berjalan (pengaturan Windows tidak diubah). Tetap colok ke listrik dan jangan tutup laptop.

Nomor eksperimen: 4 = EXP-0R, 1 = EXP-1, 2 = EXP-2, 3 = EXP-3. EXP-0 seed 42 memakai `stage_8_ablation_clean_20261020/exp_0_baseline`.

## Tanpa GPU (bisa dikerjakan kapan saja)

- Audit label 150 citra: buka `audit_label_150/LEMBAR_AUDIT_LABEL.xlsx`, baca sheet Panduan. Jangan buka `audit_label_150/_kunci/` sebelum selesai.
- Struktur kelas: `_analisis_struktur_kelas/STRUKTUR_KELAS_REPORT.md` (`cek_struktur_kelas.py`).
- Penulisan Bab 1–3 dan Bab 4.1 (RQ1).

## Catatan

- Test set tidak disentuh di langkah mana pun.
- Bila ada yang gagal dua kali, kirim file log dari `logs_v2\`.

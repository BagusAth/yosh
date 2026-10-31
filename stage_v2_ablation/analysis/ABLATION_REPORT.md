# Ablasi multi-seed — hasil (Fase 3)
Tanggal: 2026-10-31 09:40 · seed tersedia: 42, 123, 2026 · 2,567 citra validasi bersih · bootstrap klaster 1000 iterasi

## mAP@50–95 per model (hitung ulang dari prediksi per citra; angka val() Ultralytics dalam kurung)

| Model | Seed 42 | Seed 123 | Seed 2026 | Rata-rata ± SD |
| --- | --- | --- | --- | --- |
| EXP-0 | 25.67 (26.20) | 26.22 (26.55) | 26.06 (27.00) | 25.98 ± 0.29 |
| EXP-0R | 25.53 (26.09) | 25.52 (26.50) | 25.92 (26.69) | 25.66 ± 0.23 |
| EXP-1 | 24.35 (25.08) | 24.31 (25.05) | 24.37 (24.58) | 24.34 ± 0.03 |
| EXP-2 | 25.69 (26.53) | 26.19 (26.65) | 25.97 (26.79) | 25.95 ± 0.25 |
| EXP-3 | 24.76 (25.39) | 25.80 (26.12) | 25.18 (25.59) | 25.25 ± 0.53 |

## Selisih antar strategi (poin mAP@50–95; rata-rata seed yang tersedia untuk kedua sisi)

| Perbandingan | Arti | Per seed | Rata-rata | IK 95% | Selisih val() | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | Efek replacement saja (kontrol mekanisme) | -0.13 / -0.71 / -0.14 | -0.33 | -0.77 s.d. +0.14 | -0.11 / -0.05 / -0.31 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | RQ2 inti: efek pembobotan kesulitan | -1.18 / -1.21 / -1.55 | -1.31 | -1.92 s.d. -0.67 | -1.01 / -1.45 / -2.11 | menurunkan secara konsisten |
| EXP-1 − EXP-0 | RQ2: sampling adaptif vs baseline | -1.32 / -1.91 / -1.69 | -1.64 | -2.26 s.d. -0.93 | -1.12 / -1.50 / -2.42 | menurunkan secara konsisten |
| EXP-2 − EXP-0 | RQ3: context crop vs baseline (aturan keputusan bingkai) | +0.02 / -0.03 / -0.10 | -0.04 | -0.48 s.d. +0.45 | +0.33 / +0.10 / -0.21 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | Gabungan vs baseline | -0.91 / -0.42 / -0.88 | -0.74 | -1.26 s.d. -0.17 | -0.81 / -0.43 / -1.41 | menurunkan secara konsisten |
| EXP-3 − EXP-2 | Menambah sampling adaptif di atas crop | -0.93 / -0.39 / -0.78 | -0.70 | -1.26 s.d. -0.16 | -1.14 / -0.53 / -1.20 | menurunkan secara konsisten |
| EXP-3 − EXP-1 | Menambah crop di atas sampling adaptif | +0.41 / +1.49 / +0.81 | +0.90 | +0.32 s.d. +1.45 | +0.31 / +1.07 / +1.01 | meningkatkan secara konsisten |
| Interaksi | (EXP-3 − EXP-2) − (EXP-1 − EXP-0) | +0.39 / +1.52 / +0.91 | +0.94 | +0.18 s.d. +1.68 | -0.02 / +0.97 / +1.22 | meningkatkan secara konsisten |

## AP@50–95 per kelas (hitung ulang, rata-rata seed 42, 123, 2026 agar sebanding)

| Model | CBB | CBSD | CGM | CMD | HEALTHY |
| --- | --- | --- | --- | --- | --- |
| EXP-0 | 24.08 | 30.71 | 26.36 | 20.93 | 27.84 |
| EXP-0R | 23.18 | 30.45 | 26.53 | 21.03 | 27.11 |
| EXP-1 | 20.76 | 28.96 | 25.43 | 20.52 | 26.05 |
| EXP-2 | 24.10 | 30.79 | 26.27 | 20.97 | 27.61 |
| EXP-3 | 22.65 | 29.54 | 26.28 | 20.61 | 27.16 |

Catatan:
- IK 95% berasal dari bootstrap klaster per grup foto (resampel citra yang sama untuk semua model), lalu dirata-rata antar seed. IK ini menangkap ketidakpastian sampel citra; variasi antar seed terlihat dari kolom per seed.
- Angka val() Ultralytics menjalankan citra 640 px dalam bingkai 672 px (rect, pad 0,5) dan presisi setengah saat training, sehingga berbeda 0,3–0,9 poin dari hitung ulang. Metrik utama Fase 5 adalah hitung ulang (protokol sama untuk semua model).

# Ablasi multi-seed — hasil (Fase 6 test set)
Tanggal: 2026-11-02 20:01 · seed tersedia: 42, 123, 2026 · 1,282 citra test · bootstrap klaster 1000 iterasi

## mAP@50–95 per model (hitung ulang dari prediksi per citra; angka val() Ultralytics dalam kurung)

| Model | Seed 42 | Seed 123 | Seed 2026 | Rata-rata ± SD |
| --- | --- | --- | --- | --- |
| EXP-0 | 24.08 (24.83) | 24.21 (24.74) | 23.84 (24.66) | 24.04 ± 0.19 |
| EXP-0R | 24.06 (24.92) | 24.48 (24.62) | 23.42 (24.36) | 23.99 ± 0.53 |
| EXP-1 | 22.55 (22.87) | 22.98 (23.53) | 22.91 (23.42) | 22.81 ± 0.23 |
| EXP-2 | 24.14 (25.04) | 24.85 (25.63) | 24.50 (24.89) | 24.50 ± 0.36 |
| EXP-3 | 23.12 (23.36) | 23.09 (23.70) | 23.16 (23.72) | 23.12 ± 0.03 |

## Selisih antar strategi (poin mAP@50–95; rata-rata seed yang tersedia untuk kedua sisi)

| Perbandingan | Arti | Per seed | Rata-rata | IK 95% | Selisih val() | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | Efek replacement saja (kontrol mekanisme) | -0.02 / +0.27 / -0.42 | -0.06 | -0.63 s.d. +0.55 | +0.08 / -0.12 / -0.30 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | RQ2 inti: efek pembobotan kesulitan | -1.51 / -1.50 / -0.51 | -1.17 | -1.88 s.d. -0.47 | -2.04 / -1.09 / -0.94 | menurunkan secara konsisten |
| EXP-1 − EXP-0 | RQ2: sampling adaptif vs baseline | -1.53 / -1.23 / -0.93 | -1.23 | -1.95 s.d. -0.49 | -1.96 / -1.21 / -1.24 | menurunkan secara konsisten |
| EXP-2 − EXP-0 | RQ3: context crop vs baseline (aturan keputusan bingkai) | +0.05 / +0.64 / +0.67 | +0.45 | -0.12 s.d. +1.04 | +0.20 / +0.89 / +0.23 | tidak terbukti berbeda |
| EXP-3 − EXP-0 | Gabungan vs baseline | -0.96 / -1.12 / -0.68 | -0.92 | -1.60 s.d. -0.31 | -1.48 / -1.04 / -0.94 | menurunkan secara konsisten |
| EXP-3 − EXP-2 | Menambah sampling adaptif di atas crop | -1.01 / -1.75 / -1.35 | -1.37 | -2.07 s.d. -0.71 | -1.68 / -1.93 / -1.17 | menurunkan secara konsisten |
| EXP-3 − EXP-1 | Menambah crop di atas sampling adaptif | +0.57 / +0.11 / +0.25 | +0.31 | -0.30 s.d. +0.87 | +0.48 / +0.17 / +0.30 | tidak terbukti berbeda |
| Interaksi | (EXP-3 − EXP-2) − (EXP-1 − EXP-0) | +0.52 / -0.52 / -0.42 | -0.14 | -1.02 s.d. +0.65 | +0.28 / -0.72 / +0.07 | tidak terbukti berbeda |

## AP@50–95 per kelas (hitung ulang, rata-rata seed 42, 123, 2026 agar sebanding)

| Model | CBB | CBSD | CGM | CMD | HEALTHY |
| --- | --- | --- | --- | --- | --- |
| EXP-0 | 19.49 | 24.29 | 25.94 | 20.81 | 29.69 |
| EXP-0R | 19.14 | 24.50 | 26.31 | 20.24 | 29.75 |
| EXP-1 | 17.66 | 21.26 | 25.28 | 20.88 | 28.98 |
| EXP-2 | 19.45 | 25.69 | 26.53 | 21.18 | 29.63 |
| EXP-3 | 18.48 | 22.07 | 25.39 | 20.38 | 29.31 |

Catatan:
- IK 95% berasal dari bootstrap klaster per grup foto (resampel citra yang sama untuk semua model), lalu dirata-rata antar seed. IK ini menangkap ketidakpastian sampel citra; variasi antar seed terlihat dari kolom per seed.
- Angka val() Ultralytics menjalankan citra 640 px dalam bingkai 672 px (rect, pad 0,5) dan presisi setengah saat training, sehingga berbeda 0,3–0,9 poin dari hitung ulang. Metrik utama Fase 5 adalah hitung ulang (protokol sama untuk semua model).

# Fase 6 — test set dibanding validasi

Dibuat 2026-11-02T20:30:27. Test set: 1,273 citra, 5,089 objek GT (CBB 339, CBSD 330, CGM 968, CMD 1,447, HEALTHY 2,005).

Aturan (pra-registrasi): verdict test dilaporkan apa adanya; tidak ada pemilihan ulang atau penyetelan. Validasi tetap bukti utama; test sebagai konfirmasi.

## mAP@50–95 per strategi (rata-rata seed, hitung ulang; val() Ultralytics sebagai pendamping)

| Strategi | Test (rata-rata ± SD) | Seed | val() test |
| --- | --- | --- | --- |
| EXP-0 | 24.04 ± 0.19 | 3 | 24.74 |
| EXP-0R | 23.99 ± 0.53 | 3 | 24.63 |
| EXP-1 | 22.81 ± 0.23 | 3 | 23.27 |
| EXP-2 | 24.50 ± 0.36 | 3 | 25.18 |
| EXP-3 | 23.12 ± 0.03 | 3 | 23.59 |

## Kontras (poin mAP@50–95; IK 95% bootstrap klaster per grup foto)

| Kontras | Peran | Validasi | Verdict validasi | Test | Verdict test | Sama? |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0R − EXP-0 | pendukung | -0.33 (-0.77 s.d. +0.14) | tidak terbukti berbeda | -0.06 (-0.63 s.d. +0.55) | tidak terbukti berbeda | ya |
| EXP-1 − EXP-0R | RQ1 inti | -1.31 (-1.92 s.d. -0.67) | menurunkan secara konsisten | -1.17 (-1.88 s.d. -0.47) | menurunkan secara konsisten | ya |
| EXP-1 − EXP-0 | pendukung | -1.64 (-2.26 s.d. -0.93) | menurunkan secara konsisten | -1.23 (-1.95 s.d. -0.49) | menurunkan secara konsisten | ya |
| EXP-2 − EXP-0 | RQ2 inti | -0.04 (-0.48 s.d. +0.45) | tidak terbukti berbeda | +0.45 (-0.12 s.d. +1.04) | tidak terbukti berbeda | ya |
| EXP-3 − EXP-0 | pendukung | -0.74 (-1.26 s.d. -0.17) | menurunkan secara konsisten | -0.92 (-1.60 s.d. -0.31) | menurunkan secara konsisten | ya |
| EXP-3 − EXP-2 | pendukung | -0.70 (-1.26 s.d. -0.16) | menurunkan secara konsisten | -1.37 (-2.07 s.d. -0.71) | menurunkan secara konsisten | ya |
| EXP-3 − EXP-1 | pendukung | +0.90 (+0.32 s.d. +1.45) | meningkatkan secara konsisten | +0.31 (-0.30 s.d. +0.87) | tidak terbukti berbeda | TIDAK |
| Interaksi | pendukung | +0.94 (+0.18 s.d. +1.68) | meningkatkan secara konsisten | -0.14 (-1.02 s.d. +0.65) | tidak terbukti berbeda | TIDAK |

## CONV (seed 42; deskriptif, tanpa aturan keputusan tiga seed)

| Kontras | Validasi | Test |
| --- | --- | --- |
| EXP-0: 60ep − 30ep | +0.30 (-0.40 s.d. +1.07) | -0.12 (-1.08 s.d. +0.82) |
| EXP-1: 60ep − 30ep | -0.47 (-1.39 s.d. +0.38) | -0.11 (-1.11 s.d. +0.95) |
| EXP-1 − EXP-0 pada 30ep | -1.32 (-2.13 s.d. -0.43) | -1.53 (-2.63 s.d. -0.50) |
| EXP-1 − EXP-0 pada 60ep | -2.09 (-3.05 s.d. -1.21) | -1.52 (-2.76 s.d. -0.37) |
| Perubahan selisih (60ep − 30ep) | -0.77 (-1.89 s.d. +0.26) | +0.01 (-1.32 s.d. +1.39) |

Rincian: ABLATION_REPORT.md (test), conv/CONV_REPORT.md, ukuran/UKURAN_REPORT.md, error/ERROR_REPORT.md.

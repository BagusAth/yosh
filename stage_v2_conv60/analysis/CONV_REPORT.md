# CONV (Fase 4): EXP-0 dan EXP-1, seed 42, 30 vs 60 epoch
Tanggal: 2026-10-31 09:47 · 2,567 citra validasi bersih · bootstrap klaster 1000 iterasi

Satu seed: IK hanya menangkap ketidakpastian sampel citra. Hasil dilaporkan deskriptif, tidak memakai aturan keputusan tiga seed.

## Jalannya training (val() Ultralytics selama training)

| Run | Epoch selesai | Epoch terbaik | mAP@50–95 terbaik (val) | LR saat berhenti | Early stopping |
| --- | --- | --- | --- | --- | --- |
| EXP-0 30ep | 30 | 30 | 26.18 | 0.000048 | tidak |
| EXP-1 30ep | 30 | 29 | 25.09 | 0.000048 | tidak |
| EXP-0 60ep | 55 | 45 | 26.99 | 0.000121 | ya |
| EXP-1 60ep | 40 | 30 | 24.80 | 0.000396 | ya |

## mAP hitung ulang (bobot best.pt, prediksi per citra)

| Run | mAP@50 | mAP@50–95 | CBB | CBSD | CGM | CMD | HEALTHY |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0 30ep | 35.42 | 25.67 | 23.11 | 30.09 | 26.52 | 20.59 | 28.04 |
| EXP-1 30ep | 34.10 | 24.35 | 19.72 | 28.37 | 26.43 | 20.71 | 26.51 |
| EXP-0 60ep | 36.08 | 25.97 | 23.55 | 31.77 | 26.69 | 20.71 | 27.10 |
| EXP-1 60ep | 33.82 | 23.88 | 21.77 | 26.99 | 24.82 | 20.02 | 25.80 |

## Selisih (poin mAP@50–95, hitung ulang)

| Perbandingan | Selisih | IK 95% (sampel citra) |
| --- | --- | --- |
| EXP-0: 60ep − 30ep | +0.30 | -0.40 s.d. +1.07 |
| EXP-1: 60ep − 30ep | -0.47 | -1.39 s.d. +0.38 |
| EXP-1 − EXP-0 pada 30ep | -1.32 | -2.13 s.d. -0.43 |
| EXP-1 − EXP-0 pada 60ep | -2.09 | -3.05 s.d. -1.21 |
| Perubahan selisih (60ep − 30ep) | -0.77 | -1.89 s.d. +0.26 |

Catatan: run 60 epoch memakai jadwal LR dan close_mosaic (6 epoch terakhir) untuk 60 epoch. Early stopping (patience 10)
menghentikan run sebelum LR habis dan sebelum fase tanpa mosaic, sehingga run 60 epoch bukan versi panjang yang tuntas dari run 30 epoch.

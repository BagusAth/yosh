# CONV (Fase 4): EXP-0 dan EXP-1, seed 42, 30 vs 60 epoch
Tanggal: 2026-11-02 20:05 · 1,282 citra test · bootstrap klaster 1000 iterasi

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
| EXP-0 30ep | 33.40 | 24.08 | 20.07 | 24.45 | 25.09 | 21.11 | 29.71 |
| EXP-1 30ep | 31.90 | 22.55 | 17.12 | 20.84 | 25.34 | 20.58 | 28.87 |
| EXP-0 60ep | 33.86 | 23.96 | 19.41 | 24.21 | 26.39 | 19.91 | 29.90 |
| EXP-1 60ep | 32.23 | 22.44 | 15.65 | 21.86 | 24.48 | 20.28 | 29.96 |

## Selisih (poin mAP@50–95, hitung ulang)

| Perbandingan | Selisih | IK 95% (sampel citra) |
| --- | --- | --- |
| EXP-0: 60ep − 30ep | -0.12 | -1.08 s.d. +0.82 |
| EXP-1: 60ep − 30ep | -0.11 | -1.11 s.d. +0.95 |
| EXP-1 − EXP-0 pada 30ep | -1.53 | -2.63 s.d. -0.50 |
| EXP-1 − EXP-0 pada 60ep | -1.52 | -2.76 s.d. -0.37 |
| Perubahan selisih (60ep − 30ep) | +0.01 | -1.32 s.d. +1.39 |

Catatan: run 60 epoch memakai jadwal LR dan close_mosaic (6 epoch terakhir) untuk 60 epoch. Early stopping (patience 10)
menghentikan run sebelum LR habis dan sebelum fase tanpa mosaic, sehingga run 60 epoch bukan versi panjang yang tuntas dari run 30 epoch.

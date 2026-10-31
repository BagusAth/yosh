# Analisis log context crop (Fase 5 A4)
Tanggal: 2026-10-31 20:30 · 6 run (EXP-2, EXP-3 × 3 seed) · 30 epoch

Aturan crop: jendela = 2 × kotak lesi target, di-resize ke 640 × 640 dengan skala x dan y terpisah.
Citra latih yang punya target kecil di manifest: 755.

| Run | Crop total | Crop/epoch (median) | % citra latih per epoch | Citra unik | Jendela median (px) | Zoom median | Zoom p90 | Zoom ≥ 10× | Zoom ≥ 32× | Aspek terdistorsi* | Lesi target sesudah crop (median px) | Hanya 1 kotak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-2 seed 42 | 11,196 | 372 | 4.1% | 755 | 39.7 | 16.1× | 70.1× | 100.0% | 29.4% | 78.4% | 320 | 91.4% |
| EXP-2 seed 123 | 11,196 | 372 | 4.1% | 755 | 39.7 | 16.1× | 70.1× | 100.0% | 29.4% | 78.4% | 320 | 91.4% |
| EXP-2 seed 2026 | 11,196 | 372 | 4.1% | 755 | 39.7 | 16.1× | 70.1× | 100.0% | 29.4% | 78.4% | 320 | 91.4% |
| EXP-3 seed 42 | 15,516 | 513 | 5.7% | 755 | 39.7 | 16.1× | 68.5× | 100.0% | 28.5% | 78.6% | 320 | 91.4% |
| EXP-3 seed 123 | 15,714 | 522 | 5.8% | 755 | 39.7 | 16.1× | 68.5× | 100.0% | 28.7% | 78.3% | 320 | 91.3% |
| EXP-3 seed 2026 | 15,631 | 522 | 5.8% | 755 | 40.0 | 16.0× | 68.5× | 100.0% | 28.3% | 78.4% | 320 | 91.3% |

*Aspek terdistorsi = zoom_x / zoom_y di luar 0,8–1,25.

## Kesamaan peristiwa crop antar seed

Jaccard multiset peristiwa (epoch, citra, ukuran jendela) terhadap seed 42: EXP-2 42 vs 123 100.0%; EXP-2 42 vs 2026 100.0%; EXP-3 42 vs 123 24.8%; EXP-3 42 vs 2026 24.6%.

Penyebab: build_dataloader Ultralytics 8.3.235 menyemai generator DataLoader dengan konstanta 6148914691236517205 + RANK, bukan dengan seed pelatihan. Tanpa sampler khusus, urutan citra dan seluruh augmentasi (termasuk keputusan crop) identik di ketiga seed, sehingga variasi antar seed EXP-0 dan EXP-2 hanya berasal dari inisialisasi kepala deteksi dan nondeterminisme GPU. EXP-0R, EXP-1, dan EXP-3 memakai WeightedRandomSampler yang mengikuti seed global, sehingga urutan datanya berbeda antar seed (tampak pada jumlah crop EXP-3 yang berbeda per seed).

## Ukuran lesi: target crop vs objek validasi

- Lesi target sebelum crop: median sisi 19.8 px; 100.0% < 32 px (EXP-2 seed 42).
- Lesi target sesudah crop: median sisi 320 px; 100.0% > 96 px (ukuran "besar" COCO).
- Objek validasi: median sisi 145.2 px (p10 53.1, p90 376.4); 3.7% < 32 px.

## Kelas dominan citra yang di-crop (%)

| Run | CBB | CBSD | CGM | CMD | HEALTHY |
| --- | --- | --- | --- | --- | --- |
| EXP-2 seed 42 | 10.0 | 3.2 | 21.6 | 26.1 | 39.1 |
| EXP-2 seed 123 | 10.0 | 3.2 | 21.6 | 26.1 | 39.1 |
| EXP-2 seed 2026 | 10.0 | 3.2 | 21.6 | 26.1 | 39.1 |
| EXP-3 seed 42 | 10.9 | 3.6 | 21.3 | 26.3 | 37.9 |
| EXP-3 seed 123 | 10.7 | 3.8 | 21.6 | 26.0 | 37.9 |
| EXP-3 seed 2026 | 10.4 | 3.5 | 22.4 | 25.8 | 37.9 |

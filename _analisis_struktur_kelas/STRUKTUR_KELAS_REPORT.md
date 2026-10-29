# Struktur kelas pool latih + validasi bersih
Tanggal: 2026-10-29 09:22
Citra: 11,547 (test terkunci tidak disentuh). Kelas = kelas dominan per citra.

## Kelas x ukuran grup foto (1 = tanpa salinan, 2 = dua salinan augmentasi)
| Kelas | 1 | 2 | Total |
| --- | --- | --- | --- |
| (tanpa kotak) | 26 | 58 | 84 |
| CBB | 1,407 | 0 | 1,407 |
| CBSD | 1,274 | 288 | 1,562 |
| CGM | 0 | 2,838 | 2,838 |
| CMD | 0 | 2,822 | 2,822 |
| HEALTHY | 0 | 2,834 | 2,834 |
| Total | 2,707 | 8,840 | 11,547 |

## Kelas x split asal Roboflow v7
| Kelas | test | train | valid | Total |
| --- | --- | --- | --- | --- |
| (tanpa kotak) | 15 | 58 | 11 | 84 |
| CBB | 564 | 0 | 843 | 1,407 |
| CBSD | 765 | 288 | 509 | 1,562 |
| CGM | 0 | 2,838 | 0 | 2,838 |
| CMD | 0 | 2,822 | 0 | 2,822 |
| HEALTHY | 0 | 2,834 | 0 | 2,834 |
| Total | 1,344 | 8,840 | 1,363 | 11,547 |

## Objek per kelas x ukuran grup
| Kelas | 1 | 2 | Total |
| --- | --- | --- | --- |
| CBB | 3,195 | 0 | 3,195 |
| CBSD | 2,406 | 591 | 2,997 |
| CGM | 0 | 8,779 | 8,779 |
| CMD | 0 | 12,234 | 12,234 |
| HEALTHY | 0 | 16,663 | 16,663 |
| Total | 5,601 | 38,267 | 43,868 |

## Baji hitam di sudut (isian rotasi), sampel acak 400 citra per kelompok, seed 42

| Kelompok | n | Citra dengan baji hitam | Rata-rata piksel hitam murni |
| --- | --- | --- | --- |
| asal train | 400 | 41.2% | 1.5% |
| asal valid/test | 400 | 2.5% | 0.13% |

Baji hitam = lebih dari 30% piksel di salah satu sudut 24x24 px bernilai RGB <= 3.

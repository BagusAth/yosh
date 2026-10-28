# RQ1 bagian B — Audit kebocoran berbasis nama file menghasilkan positif palsu

Tanggal: 2026-10-28 10:25:34

**Kesimpulan:** split bawaan Roboflow Cassava v7 tidak bocor. Audit berbasis nama file menandai 661 grup lintas split
dan 64.34% citra validasi berbagi nama dengan citra latih, tetapi pemeriksaan berbasis isi citra menemukan
0 grup lintas split.

| Bukti | Berbasis nama file | Berbasis isi/kelas |
| --- | --- | --- |
| Grup yang melintasi split asli | 661 | 0 |
| Citra validasi yang "berpasangan" dengan citra latih | 969 (64.34%) | 0 |
| Duplikat eksak (SHA-256) lintas grup | — | 0 |
| Kandidat kemiripan visual (pHash) lintas grup | — | 4 (lintas split: 0) |

Rincian 661 grup lintas split pada arsip audit awal:

- 661 bernama file generik (`image_N`).
- 661 berkonflik label (kelas berbeda dengan nama sama).
- 34 memiliki bukti kemiripan visual tercatat; label "AUGMENTED_DERIVATIVE_PROVENANCE" diberikan berdasarkan nama, bukan isi.

Struktur grup bersih per split asli: test|1 citra: 1,505, train|2 citra: 4,909, valid|1 citra: 1,506.
Pola ini (grup latih berisi 2 citra hasil augmentasi, grup validasi dan uji berisi 1 citra) konsisten dengan pembagian data
SEBELUM augmentasi, yaitu praktik bawaan Roboflow.

File pendukung: `rq1_audit_summary.json`, `filename_cross_split_groups.csv`.

## Cek tambahan: 8 grup yang ditandai "mirip" pada audit awal

Audit awal memberi label STRONG/MODERATE_VISUAL_SIMILARITY pada 8 grup nama file lintas split. Jarak minimum yang dicatat
ternyata berasal dari pasangan kembaran augmentasi di split yang SAMA. Jarak dHash 256-bit antar anggota yang BERBEDA split:

| Grup | Min dHash lintas split | Min dHash dalam split |
| --- | --- | --- |
| image_130_jpg | 115 | 74 |
| image_147_jpg | 114 | 30 |
| image_170_jpg | 116 | 14 |
| image_213_jpg | 124 | 41 |
| image_221_jpg | 129 | 0 |
| image_315_jpg | 115 | 68 |
| image_458_jpg | 109 | 14 |
| image_58_jpg | 88 | 0 |

Dua citra acak yang tidak berhubungan berjarak sekitar 128 bit. Jarak 88–129 berarti anggota lintas split adalah citra yang berbeda,
sedangkan kembaran augmentasi berjarak 0–14 bit. Tidak ada duplikat lintas split. Skrip: `cek_dhash_grup_mirip.py`.

Catatan angka: arsip audit awal melaporkan 575 grup latih–validasi ("51,71% data validasi", dihitung per grup). Hitungan per citra di
laporan ini memberi 969 citra validasi (64,34%) yang berbagi nama file dengan citra latih. Keduanya mengukur hal yang sama, yaitu tabrakan
nama file, bukan kebocoran.

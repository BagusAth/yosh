# Pra-registrasi Fase 6: evaluasi test set

Status: DISETUJUI (ubah menjadi "Status: DISETUJUI" setelah disetujui; runner menolak berjalan sebelum itu)

Disusun 2 November 2026 (pakem v1.5.2), sebelum test set disentuh untuk penelitian v2. Dokumen ini
mengunci apa yang dihitung pada test set, bagaimana hasilnya ditafsirkan, dan apa yang tidak dilakukan.

## 1. Data

- Test set: `cassava-v7-standardized/test`, 1.282 citra dari 793 grup foto, label berformat kotak YOLO.
- Tidak ada grup foto yang sama dengan data latih (8.980 citra) maupun validasi (2.567 citra); dicek 2 November.
- Test set dijalankan satu kali lewat `jalankan_fase6.bat`. Pengaman: runner menolak berjalan bila
  `stage_v2_fase6/SELESAI.txt` sudah ada atau dokumen ini belum berstatus DISETUJUI. Bila terputus, menjalankan
  ulang hanya melengkapi model yang belum punya prediksi.

## 2. Model (17, semua sudah dilatih dan dibekukan)

| Strategi | Seed 42 | Seed 123 | Seed 2026 |
| --- | --- | --- | --- |
| EXP-0 | `stage_8_ablation_clean_20261020/exp_0_baseline` | `stage_v2_ablation/seed_123/exp_0_baseline` | `stage_v2_ablation/seed_2026/exp_0_baseline` |
| EXP-0R | `stage_v2_ablation/seed_42/exp_0r_uniform_replacement` | seed_123 | seed_2026 |
| EXP-1 | `stage_v2_ablation/seed_42/exp_1_adaptive_sampling` | seed_123 | seed_2026 |
| EXP-2 | `stage_v2_ablation/seed_42/exp_2_targeted_aug` | seed_123 | seed_2026 |
| EXP-3 | `stage_v2_ablation/seed_42/exp_3_full_proposed` | seed_123 | seed_2026 |
| CONV | `stage_v2_conv60/seed_42/exp_0_baseline` dan `exp_1_adaptive_sampling` (60 epoch) | – | – |

Bobot `weights/best.pt` (dipilih dengan validasi saat training). Hash SHA-256 setiap bobot dan skrip dicatat
di `stage_v2_fase6/MULAI.txt` saat runner mulai.

## 3. Prosedur

- Prediksi per citra dengan `predict_eleak_per_image.py`: 640 px, conf 0,001, NMS seperti val() (IoU 0,7,
  multi-label, tanpa batas waktu), max_det 300. Protokol sama dengan validasi.
- Angka `val()` Ultralytics pada split test (`val_test_fase6.py`) hanya pendamping.
- Analisis memakai skrip yang sama dengan validasi; perubahan hanya opsi folder/nama berkas (regresi pada data
  validasi menghasilkan angka identik, dicek 2 November).

## 4. Analisis utama (konfirmasi)

- Metrik: mAP@50–95 hitung ulang dari prediksi per citra (AP 101 titik, 10 ambang IoU).
- Kontras sama dengan validasi: EXP-0R − EXP-0, **EXP-1 − EXP-0R (RQ1 inti)**, EXP-1 − EXP-0,
  **EXP-2 − EXP-0 (RQ2 inti)**, EXP-3 − EXP-0, EXP-3 − EXP-2, EXP-3 − EXP-1, interaksi.
- IK 95%: bootstrap klaster per grup foto, 1.000 iterasi (4 × 250), resampel sama untuk semua model,
  dirata-rata antar seed.
- Aturan keputusan (TERKUNCI, sama dengan validasi): "meningkatkan/menurunkan secara konsisten" bila selisih
  bertanda sama di ketiga seed dan IK 95% tidak memuat nol; selain itu "tidak terbukti berbeda".

## 5. Analisis sekunder (deskriptif)

- mAP@50 dan AP@50–95 per kelas.
- AP dan recall per ukuran objek (kecil, sedang, besar) dengan kontras yang sama (`analyze_ukuran.py`).
- Pergeseran jenis error pada τ 0,25 (`analyze_error.py`).
- CONV: EXP-1 − EXP-0 pada 30 dan 60 epoch, seed 42, IK bootstrap (tanpa aturan tiga seed).
- Tabel validasi berdampingan dengan test (`analyze_fase6.py`).

## 6. Penafsiran

- Hasil test dilaporkan apa adanya. Tidak ada pemilihan model, penyetelan, atau pelatihan ulang setelah test.
- Validasi (3 seed, 2.567 citra) tetap bukti utama; test (1.282 citra) adalah konfirmasi pada data yang belum
  dipakai untuk keputusan apa pun di penelitian v2.
- Bila verdict test sama dengan validasi: kesimpulan RQ dinyatakan terkonfirmasi di test.
- Bila berbeda: kedua hasil dilaporkan; kesimpulan dinyatakan "tidak terkonfirmasi di test" dan dibahas dengan
  lebar IK (test lebih kecil, IK lebih lebar). Verdict validasi tidak diubah.

## 7. Yang tidak dilakukan pada test set

- Uji jalan pintas (baji hitam, pinggiran bergaris): hanya di validasi (keputusan 2 November).
- Audit label dan analisis komposisi sampel.
- Analisis lain di luar dokumen ini. Bila kelak diperlukan, ditandai sebagai analisis pasca-hoc.

## 8. Keterbatasan yang sudah diketahui

- Test set pernah dipakai di pipeline lama sebelum bingkai penelitian diubah (27 Oktober). Keputusan penelitian v2
  (skor kesulitan, strategi, aturan keputusan, CONV) diambil di validasi. Diungkapkan di Bab 3.
- Seed DataLoader Ultralytics konstan: EXP-0 dan EXP-2 berbagi urutan data dan augmentasi di ketiga seed.
- Satu seed untuk CONV.

## 9. Keluaran

`stage_v2_fase6/`: `per_image/` (15 model + `gt_test.csv`), `per_image_conv/` (2 model), `val_test.json`,
`analysis/` (`ABLATION_REPORT.md`, `conv/`, `ukuran/`, `error/`, `FASE6_RINGKASAN.md`), `MULAI.txt`, `SELESAI.txt`,
log di `logs_v2/fase6_jalan_*.log`.

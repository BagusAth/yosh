# Spesifikasi analisis pasca-hoc butir 6–7 (TIDE dan AP per ambang IoU)

Ditulis 5 November 2026 pukul 17.00 WIB, sebelum analisis dijalankan. Disetujui pengguna 5 November 2026 pukul 16.37
("kerjakan butir 1-5 dahulu, setelah selesai baru kerjakan 6-7"). Skrip: `analyze_tide_iou.py` (docstring = isi dokumen ini).

## Status

- **Pasca-hoc dan eksploratif.** Tidak mengubah rumusan masalah, desain, aturan keputusan, maupun verdict RQ1–RQ3.
- **Hanya validasi** (2.567 citra). Test set tidak disentuh (pra-registrasi Fase 6, bagian 7).
- Tidak ada pelatihan model baru dan tidak ada GPU; hanya prediksi per citra yang sudah ada.

## Data

- Prediksi per citra 15 model: `stage_v2_ablation/per_image/preds_seed<s>_<exp>.csv` (640 px, conf 0,001, NMS seperti
  val(), max_det 300). GT kotak: `gt_cleanval.csv`. Grup foto: `cassava-v7-stage-6/resplit_group_manifest.csv`.
- Urutan citra sama dengan analisis utama (`analyze_eleak_per_image.prepare_run`), sehingga resampel bootstrap dapat
  dibuat identik.

## Butir 6 — kontribusi jenis error dalam satuan AP (TIDE)

- Mengikuti kode rujukan tidecv 1.0.1 (github.com/dbolya/tide, commit 49a5d2a) tanpa mengubah aturan:
  pos_threshold 0,5; background_threshold 0,1; max_dets 100 per citra (bawaan tidecv); mode kotak; semua citra
  validasi didaftarkan sebagai citra GT (prediksi pada citra tanpa kotak dihitung sebagai Bkg, seperti pemuat COCO tidecv).
- Pencocokan: per citra, prediksi diurutkan menurut skor (stabil), 100 teratas; tiap prediksi mengambil GT sekelas yang
  belum terpakai dengan IoU tertinggi; TP bila IoU ≥ 0,5.
- Penggolongan prediksi salah (urutan uji seperti tidecv): Loc (0,1 ≤ IoU sekelas maks ≤ 0,5), Cls (IoU maks dengan GT
  kelas lain ≥ 0,5), Dupe (IoU maks dengan GT sekelas yang sudah terpakai ≥ 0,5), Bkg (IoU maks dengan GT mana pun ≤ 0,1),
  selain itu Both. Miss = GT tak terpakai yang tidak dapat diperbaiki lewat Loc/Cls.
- Perbaikan oracle dan AP seperti tidecv: Loc/Cls menjadi TP bila GT sasarannya belum terpakai dan prediksi itu yang
  berskor tertinggi untuk GT tersebut, selain itu ditekan; Both, Dupe, Bkg ditekan; Miss mengurangi jumlah GT.
  dAP = AP50 setelah satu jenis diperbaiki − AP50 asli, dipotong pada 0 per model (seperti tidecv). Error khusus:
  FalsePos (semua FP diberi skor 0) dan FalseNeg (jumlah GT dikurangi semua GT tak terdeteksi), tidak dipotong.
  AP: 101 titik recall, rata-rata kelas, skala 0–100.
- Implementasi numpy sendiri (agar dapat di-bootstrap). **Syarat pakai:** hasil titik pada ke-15 model sama dengan kode
  tidecv asli (selisih setiap besaran < 1e-6 poin). Bila tidak sama, hasil butir 6 tidak dilaporkan.
- AP50 TIDE tidak sama persis dengan mAP@50 hitung ulang (aturan pencocokan dan max_dets berbeda); selisihnya dilaporkan.

## Butir 7 — AP per ambang IoU (diilhami PISA)

- Memakai pencocokan dan AP hitung ulang yang sama dengan analisis utama (`prepare_run`, AP 101 titik gaya Ultralytics).
- **Syarat pakai:** rata-rata 10 ambang = mAP@50–95 di `point.json` untuk ke-15 model (toleransi 1e-12), dan rata-rata
  10 ambang per resampel = kolom `|m` di `boot_seed0..3.csv` (toleransi 1e-12).

## Kontras, bootstrap, dan luaran (ditetapkan sebelum dijalankan)

- Delapan kontras analisis utama. Per seed, rata-rata 3 seed, IK 95% bootstrap klaster per grup foto, 1.000 iterasi
  (4 × 250, generator seed 9000 + k seperti `analyze_ablation.py`, resampel sama untuk semua model), dirata-rata antar seed.
- **Luaran utama butir 6:** selisih dAP_Bkg untuk EXP-1 − EXP-0R. Dugaan dari analisis jenis error (A2: FP latar naik
  +0,26 per citra): dAP_Bkg EXP-1 lebih besar.
- Luaran pendamping butir 6: selisih dAP jenis lain, FalsePos, FalseNeg, AP50 TIDE; jumlah error per jenis per citra
  (deskriptif); semua kontras.
- **Luaran utama butir 7:** selisih AP pada tiap ambang (10 angka) untuk EXP-1 − EXP-0R, dan ringkasan
  G = rata-rata selisih pada IoU 0,75–0,95 − rata-rata selisih pada IoU 0,50–0,70 (G < 0 berarti kerugian membesar pada
  IoU tinggi, yaitu pada ketepatan lokalisasi; G > 0 berarti kerugian lebih besar pada ambang longgar).
- Luaran pendamping butir 7: selisih relatif R_t = AP_EXP-1,t / AP_EXP-0R,t − 1 (AP dijumlah antar seed) dan
  G_rel dengan definisi yang sama; kontras lain.
- Penafsiran memakai bahasa aturan keputusan penelitian ("konsisten" bila tanda sama di 3 seed dan IK 95% tidak memuat
  nol), tetapi ditulis sebagai analisis pasca-hoc eksploratif tanpa koreksi multiplisitas, dengan kata "sejalan dengan",
  bukan "disebabkan oleh".

## Keluaran

`stage_v2_ablation/analysis/tide_iou/`: `TIDE_IOU_REPORT.md`, `tide_iou_results.json`, `point_tide_iou.json`,
`boot_tide_iou_seed0..3.csv`, `cek_tidecv.json`, `cek_utama.json`.

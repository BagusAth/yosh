# Ringkasan konteks chat skripsi (lengkap, per 5 November 2026 pagi)

Tempel dokumen ini di awal chat baru, lalu tulis "lanjutkan dari sini". Dokumen ini menggabungkan semua ringkasan sebelumnya (27 Oktober – 5 November 2026).

**Dokumen acuan (Claude Doc):**
- Pakem penelitian (acuan tunggal, v1.5.7, 5 Nov): https://claude.ai/artifact/E5wWxH8XooniNv3FWmYJqN
- Draf Bab 4 Hasil dan Pembahasan — draf 4.1–4.5 (4.3 dengan audit dan Jawaban RQ3; 4.4 konfirmasi test set; 4.5 keterbatasan): https://claude.ai/artifact/JNPhRkGsXe3E32gF79Sg3k
- Bahan Bab 3 Metodologi Penelitian (2 Nov, diperbarui 4 Nov): https://claude.ai/artifact/2vQLD4RDS1sh5JbXduAywz
- Bahan Bab 5 Kesimpulan dan Saran (5 Nov): https://claude.ai/artifact/SSdJNaDyvS1b9vHMMmUd1M
- Matriks Literatur Bab 2 — Sampel Sulit dan Deteksi Penyakit Singkong (5 Nov): https://claude.ai/artifact/UxtAAjuB9iVTtd1a3qBcvG
- Laporan hasil malam B (31 Okt, penomoran RQ lama): https://claude.ai/artifact/PPcTCg8tLDjVz3esst311X
- Laporan Fase 5, analisis lanjutan (31 Okt malam + uji pinggiran bergaris 1 Nov, penomoran RQ lama): https://claude.ai/artifact/HoPBBnKqG4aLQHVySNct5G
- Dokumen proyek Claude: `claude/ringkasan-chat-skripsi.md` (dokumen ini) dan `claude/pakem-penelitian-skripsi.md` (penunjuk pakem).

---

## 1. Identitas dan aturan kerja

- **Mahasiswa:** Bagus Athallah, S1 UNDIP. Pembimbing sulit ditemui; penelitian jalan dengan judul kerja. Gate G0 (persetujuan pembimbing) masih terbuka.
- **Perangkat:** `D:\UNDIP\SKRIPSI` di laptop Windows 11, GPU RTX 4060 Laptop. `.venv-cuda`: Ultralytics 8.3.235, PyTorch 2.9.1+cu130, pycocotools 2.0.11. Branch git `fase1-skor-kesulitan-v2`; HEAD saat Fase 6: `0e35022a`.
- **GPU hanya dijalankan pengguna, dan hanya bila ia meminta**, dengan klik dua kali `.bat`. Computer use (kontrol layar) ditolak pengguna; jangan ditawarkan lagi.
- **Naskah ditulis sendiri oleh pengguna** (aturan prodi soal AI dipatuhi). Draf Claude hanya bahan. **E2 (draf untuk pembimbing) tidak disetujui**; jangan dikerjakan.
- **Jangan menghapus file.** Pindahkan (mv) ke folder arsip bila perlu.
- **Jangan menjalankan perintah git yang menulis dari VM** (meninggalkan `.lock`); perintah baca pakai `git --no-optional-locks`. **Commit dilakukan pengguna secara manual** (keputusan 2 Nov).
- **Bila pengguna meminta rancangan dulu, jangan menjalankan apa pun sebelum ia menyetujui** (4 Nov: analisis audit sempat dijalankan sebelum persetujuan; pertanyaan "sekarang apa yang harus dilakukan" bukan persetujuan). Langkah yang tidak bisa dibatalkan (membuka kunci, test set) selalu menunggu persetujuan eksplisit.
- Kunci audit `audit_label_150/_kunci/` **sudah dibuka 4 Nov** lewat `analyze_audit_label.py`; lembar audit sekarang final (perubahan apa pun = pasca-kunci dan harus dilaporkan). Claude tidak bisa lagi menjadi penilai audit yang independen (sudah membaca isian penilai pertama).
- **Test set (1.282 citra) sudah dijalankan sekali (Fase 6, 2 Nov) dan tidak boleh dijalankan ulang.** `stage_v2_fase6/SELESAI.txt` ada; runner akan menolak.
- **Akses Claude ke laptop:** shell Linux VM (2 core, 3 GB), tanpa GPU/torch (ada numpy, pandas, PIL, OpenCV, openpyxl; tanpa scipy). Workspace cloud punya scipy dan scikit-learn (statsmodels tidak bisa dipasang). Batas 120 detik per perintah (bisa `timeout_ms` hingga 180 detik); proses latar mati saat perintah selesai, jadi pekerjaan panjang dipecah.
- **Citra di `cassava-v7-runtime-mirror` berupa hardlink**; tidak bisa di-stage langsung. Olah di VM, tulis hasil (montase) ke folder lalu stage file itu.
- **Staging:** paling andal 3–4 file per panggilan, berurutan; panggilan paralel/besar sering gagal, cukup diulang.
- **Menulis balik ke laptop:** tulis ke `/mnt/user-data/outputs/` dengan **nama file baru** (commit file dengan nama lama bisa mengirim isi basi; terjadi lagi 2 Nov), commit ke path tujuan, lalu cek md5.
- **Workspace cloud tidak bisa memasang torch/ultralytics.** Skrip GPU diuji dengan data/model tiruan.
- **Analisis berat:** file kecil di-stage ke workspace cloud dan dihitung di sana, atau dijalankan di laptop lewat runner `.bat` (bootstrap di VM lambat).
- **Aturan PowerShell 5.1:** `.ps1` harus ASCII; argumen array lewat `-File` diterima sebagai teks; pakai `return` (bukan `exit`) di skrip yang dijalankan `.bat` dengan `-NoExit`; keluaran Python lewat `Invoke-Py` agar masuk log; `.bat` CRLF.
- **Format label:** `cassava-v7-clean-split` dan `cassava-v7-stage-*` menyimpan sebagian label sebagai poligon. `cassava-v7-runtime-mirror` (latih/validasi) dan `cassava-v7-standardized` (latih/validasi/test) berformat kotak YOLO. Pelatihan, gambar audit, dan semua analisis memakai label kotak. Skrip baru wajib memakai runtime-mirror/standardized (2 Nov sempat terbaca poligon di skrip kerja panduan; tertangkap sebelum ada hasil yang dipakai).
- **Claude Docs:** muat skill `anthropic-skills:docs` sebelum memakai konektor; gambar diunggah lewat Artifact `asset: true` ke URL dokumen, lalu `create blob`, lalu `![alt](blob/<id>)`. Menulis dokumen proyek (`Projects project_write`) harus didahului `project_read`.

## 2. Bingkai penelitian

- **Judul kerja (dikunci gate G1, 31 Okt):** "Analisis Pengaruh Strategi Pelatihan Berbasis Error dan Tingkat Kesulitan Objek terhadap Kinerja YOLO11n untuk Deteksi Gejala Penyakit Daun Singkong". Studi analitis, bukan "metode usulan yang unggul". Judul cadangan gugur karena EXP-2 tidak meningkat konsisten.
- **Dataset:** Roboflow Cassava v7 (cassava-disease/cassava-zpvrj, CC BY 4.0), kelas CBB, CBSD, CGM, CMD, HEALTHY. Metadata: 7.920 citra sumber; split bawaan 4.909/1.506/1.505; augmentasi rotasi ±10° dengan 2 keluaran per citra latih → versi 7 = 12.829 citra (train 9.818 = 4.909 × 2). Roboflow hanya mengaugmentasi citra latih setelah split.
- **Split bersih per grup foto:** 8.980 latih / 2.567 validasi / 1.282 test (7.920 grup; test 793 grup; tidak ada grup test di latih/validasi, dicek 2 Nov). Stratifikasi 70/20/10, seed 42. Alasan re-split: split bawaan menaruh semua CBB dan sebagian besar CBSD di valid/test.
- **Label latih:** dari 8.980 berkas label, hanya 10 memuat >1 kelas, 66 tanpa kotak; 34.219 kotak.

### Rumusan masalah (pakem v1.5, TERKUNCI; direvisi 1 Nov)

| Kode | Rumusan masalah | Tujuan (yang diukur) | Hipotesis kerja |
|---|---|---|---|
| RQ1 | Apakah mixture adaptive sampling berbasis skor kesulitan meningkatkan kinerja YOLO11n dibanding sampling uniform? | Δ mAP@50–95 EXP-1 terhadap EXP-0R dan EXP-0, rata-rata 3 seed | Δ > 0 (temuan awal mengarah ke Δ < 0) |
| RQ2 | Apakah targeted context augmentation meningkatkan kinerja, khususnya pada lesi kecil? | Δ mAP@50–95, AP dan recall objek kecil EXP-2 terhadap EXP-0 | Δ > 0, terbesar pada objek kecil |
| RQ3 | Faktor apa yang menjelaskan berhasil atau gagalnya kedua strategi? | Komposisi sampel yang ditekankan, hasil audit label, pergeseran jenis error, uji jalan pintas artefak rotasi (baji hitam dan pinggiran bergaris) | Strategi merugikan bila sampel yang ditekankan didominasi label noise |

- **Pemetaan penomoran lama** (laporan malam B, laporan Fase 5, file MD lama): RQ2 lama = RQ1, RQ3 lama = RQ2, RQ4 lama = RQ3.
- **RQ1 lama dihapus (1 Nov, keputusan pengguna):** bagian B (audit kebocoran berbasis nama file) berasal dari kesalahan audit internal sendiri, jadi tidak boleh dicantumkan dan tidak bisa menjadi rumusan masalah. Simulasi E-LEAK dipindah ke Bab 3 sebagai validasi protokol split.
- **Asal klaim "51,71% validasi bocor":** audit internal 5 Okt (`cassava_duplicate_leakage_audit/AUDIT_VERDICT.txt`) yang hanya membandingkan nama file generik. Tidak ada literatur yang menyatakan dataset ini bocor; metadata Roboflow (`roboflow_v7_api_resp.json`) dan dokumentasinya membantah. Split bawaan tidak bocor.

### Desain eksperimen (TERKUNCI)

| Eksperimen | Konfigurasi | Status |
|---|---|---|
| EXP-0 | baseline (sampling bawaan Ultralytics) | 3 seed |
| EXP-0R | bobot seragam dengan penggantian (kontrol mekanisme) | 3 seed |
| EXP-1 | mixture adaptive sampling P(k) = 0,3/N + 0,7·D_k/ΣD, dengan penggantian (WeightedRandomSampler) | 3 seed |
| EXP-2 | targeted context crop | 3 seed |
| EXP-3 | EXP-1 + crop | 3 seed |
| E-LEAK | simulasi split acak per citra (Bab 3) | selesai |
| CONV | EXP-0 dan EXP-1, seed 42, 60 epoch | selesai (early stopping) |
| EXP-1c | varian sampling sadar noise (opsional) | tidak dijalankan kecuali pembimbing meminta / audit mengarah ke sana |

- Seed 42, 123, 2026. Nomor `--exp`: 0, 4 (EXP-0R), 1, 2, 3. EXP-0 seed 42 = `stage_8_ablation_clean_20261020/exp_0_baseline`; lainnya `stage_v2_ablation/seed_<s>/<exp>`; CONV `stage_v2_conv60/seed_42/`.
- **Resep pelatihan:** YOLO11n pralatih COCO; 30 epoch, batch 16, imgsz 640; optimizer auto, lr0 0,01, lrf 0,01, momentum 0,937, weight decay 0,0005, warmup 3; mosaic 1,0 (close_mosaic 6), fliplr 0,5, HSV 0,015/0,7/0,4, translate 0,1, scale 0,5, degrees 0; AMP; patience 10; deterministic; workers 2. best.pt dipilih dengan validasi.
- **EXP-1:** bobot relatif N·P(k) 0,32–1,64; ESS 0,91; 10% teratas menerima 15,0% tarikan.
- **Context crop (EXP-2/EXP-3):** hanya citra dengan lesi kecil sulit (objek kecil, D_i ≥ 0,70), peluang 0,5 per kemunculan; target = lesi tersulit; jendela 2× kotak; kotak lain dipertahankan bila visibilitas ≥ 0,30; diubah ke 640 px; tanpa mosaic.
- **Aturan keputusan (TERKUNCI):** "meningkatkan/menurunkan secara konsisten" bila selisih bertanda sama di 3 seed dan IK 95% tidak memuat nol; selain itu "tidak terbukti berbeda". IK = bootstrap klaster per grup foto, 1.000 iterasi (4 × 250), resampel sama untuk semua model, dirata-rata antar seed.
- **Skor kesulitan v2 (TERKUNCI):** OOF 4 fold per grup (YOLO11n 12 epoch per fold, nested). Pencocokan sadar kelas τ = 0,25, kandidat IoU ≥ 0,10. D_i = 0,4(1−C*) + 0,4(1−IoU*) + 0,2·S_i (+0,35 bila kelas salah); terlewat D_i = 0,80 + 0,20·S_i; S_i = 1 kecil, 0,5 sedang, 0 besar; dipotong 0–1. D_image = 0,5·max + 0,5·mean. Sanity gate lolos (citra sulit 14,35%, DR 0,6173, ESS 0,9099). Taksonomi OOF: TP 35,7%, terlewat 48,2%, lokalisasi 10,2%, kelas 3,1%.
- **Skor kualitas label (terinspirasi ObjectLab, disederhanakan):** missing_score, swap_score, badloc_score dari prediksi OOF; label_quality = 1 − max(ketiganya). File `stage_v2_difficulty_20261028/train_image_label_quality.csv`; manifest `train_image_difficulty_manifest.csv` (kolom d_image, p_sampling).
- **Keterbatasan seed DataLoader (temuan 31 Okt):** Ultralytics menyemai generator DataLoader dengan konstanta 6148914691236517205 + RANK, jadi EXP-0 dan EXP-2 punya urutan data dan augmentasi identik di 3 seed. EXP-0R/1/3 (sampler berbobot) mengikuti seed global. Masuk keterbatasan Bab 3, Bab 5, dan pra-registrasi Fase 6.

## 3. Kronologi

- **27 Okt:** analisis kegagalan adaptive sampling; bingkai diubah menjadi studi analitis; pakem v1.0–1.1 (bagian orisinalitas/keterbaruan, lima penelitian terdekat); deep research referensi 2023–2026.
- **28 Okt:** lanjut tanpa menunggu pembimbing (v1.2: log zoom crop, skor kualitas label OOF, bootstrap berpasangan, pool E-LEAK tanpa test); Fase 1 diimplementasikan (pencocokan v2, EXP-0R); klaim 51,71% bocor terbukti positif palsu (v1.2.1–v1.3, RQ1 lama A + B); malam 1 (manifest v2, EXP-0 seed 123/2026, E-LEAK).
- **29 Okt:** analisis per citra dan DiD E-LEAK (v1.3.1); temuan struktur kelas dan baji hitam; skrip Fase 3 dan lembar audit label v1 (`build_label_audit.py`) disiapkan; bug NMS `predict()` diperbaiki (NMS seperti val); `rapikan_git.bat` dibuat (belum diketahui apakah dijalankan).
- **30 Okt:** malam A (8 run); ablasi sementara; v1.3.2. Malam: pengguna menyetujui CONV, uji baji hitam, dan erratum (`Literatur terkait penelitian YOLO singkong.md`); `uji_baji_hitam.py` dibuat. Malam B 19.50–~01.06 tanpa kegagalan (uji baji, seed 2026, CONV).
- **31 Okt siang:** ablasi 3 seed final, uji baji (`analyze_baji.py`), CONV (`analyze_conv.py`); laporan malam B; pakem v1.4; gate G1 → judul kerja.
- **31 Okt malam (rencana disetujui 19.46):** pengguna menjalankan `jalankan_fase5a.bat` (B1 uji baji EXP-1/EXP-2, 9,7 menit GPU; B2 pasang pycocotools + `cek_coco_ukuran.py`). Claude: A1 ukuran objek, A2 jenis error, A4 log crop (+ temuan seed), A5 pinggiran bergaris, analisis B1; laporan Fase 5; v1.4.1. Rencana: 1–2 Nov pengguna audit (C1); Claude A3, A6, E1. Keputusan 21.00: uji pinggiran bergaris disetujui; E1 sebagai Claude Doc.
- **1 Nov pagi:** uji pinggiran bergaris dijalankan pengguna (09.41, 6,4 menit GPU); dianalisis; v1.4.2.
- **1 Nov siang:** A3 komposisi, A6 gambar, E1 draf Bab 4 sebagai Claude Doc. Pengguna memutuskan 2 model CONV ikut test set.
- **1 Nov sore:** penelusuran dasar klaim kebocoran Roboflow → tidak ada literatur; klaim dari audit internal. Pengguna: kesalahan audit sendiri tidak dicantumkan → RQ direvisi jadi 3 (v1.5); E-LEAK ke Bab 3.
- **1 Nov malam:** penjelasan mengapa audit 150 citra perlu dan mengapa `_kunci/` jangan dibuka. Pengguna tidak bisa membedakan daun sehat/sakit dan tidak punya akses ahli → disetujui **audit tanpa keahlian (lembar v2)**; amandemen v1.5.1.
- **2 Nov pagi:** diskusi "kalau tidak diaudit" (audit tetap dikerjakan). Pengguna mulai audit. Rencana kerja paralel disetujui (commit manual): pra-registrasi analisis audit, draf 4.3, pra-registrasi + runner Fase 6, Bahan Bab 3 Doc; keputusan uji jalan pintas hanya di validasi; v1.5.2.
- **2 Nov 19.36–20.30:** pengguna menyetujui pra-registrasi (Status: DISETUJUI) dan menjalankan **Fase 6** (tanpa kegagalan).
- **2 Nov malam:** cek jawaban RQ; diskusi hasil negatif; keterbaruan (penelusuran literatur); mengapa penelitian lain melaporkan peningkatan; contoh jawaban untuk penguji; ringkasan versi 2 Nov.
- **2 Nov 21.02–21.03:** pengguna menyelesaikan lembar penilai pertama (150/150; file "llll" diganti nama menjadi `LEMBAR_AUDIT_LABEL.xlsx`) dan meng-commit-nya (`3d4ae7ca`) sebelum kunci dibuka. Claude mengecek lembar tanpa membuka kunci: lengkap, 0 isian tidak dikenal, 0 kontradiksi.
- **4 Nov 16.34:** penilai kedua selesai (30/30). 16.36 Claude menulis `BEKU_LEMBAR_AUDIT_20261104.txt` (SHA-256 kedua lembar + skrip) dan **16.37 menjalankan `analyze_audit_label.py` sebelum pengguna menyetujui** (pesan "jangan jalankan apa pun dahulu, buat rancangan" masuk 16.36 dan baru terbaca setelah perintah selesai). Claude berhenti, mengakui kesalahan, dan menyusun rancangan Tahap A–D.
- **4 Nov 19.32:** pengguna menyetujui Tahap A–C dan meminta audit dijalankan ulang. Hasil 16.37 dipindah ke `analysis_run1_20261104_1637_sebelum_persetujuan/`; run ulang 19.33 identik (kecuali stempel waktu). Tahap A (verifikasi) lolos; Tahap B–C: draf Bab 4 (4.3.3, 4.3.4, Jawaban RQ3, koreksi 4.1.5), Bahan Bab 3 (3.9, 3.11), pakem v1.5.3 (hasil test + audit), ringkasan ini.
- **4 Nov 20.09:** pengguna: penilai kedua = dirinya sendiri di hari lain; menyetujui analisis pasca-hoc; commit dilakukan sendiri. Claude: istilah intra-rater di Bab 4, Bab 3, pakem; menulis `analyze_audit_pascahoc.py` (spesifikasi dulu), menjalankan (17 detik di VM), memverifikasi ulang di cloud; Bab 4 (analisis pasca-hoc + Tabel 4.11 komposisi; sintesis jadi Tabel 4.12; Pembahasan dan Jawaban RQ3 diperbarui), Bab 3 (3.9, 3.11), pakem v1.5.4.
- **5 Nov pagi:** pengguna: "lanjutkan ke langkah berikutnya". Claude: draf **4.4 Konfirmasi pada test set** (Tabel 4.13 mAP per seed test + validasi; Tabel 4.14 delapan kontras validasi vs test; Tabel 4.15 AP per kelas; paragraf ukuran, jenis error, CONV, Pembahasan, Ringkasan konfirmasi); **Gambar 4.6** forest plot validasi vs test (`buat_gambar_test.py` → `_gambar_bab4/gbr4_8_validasi_vs_test.png`, 300 dpi, biru/lingkaran = validasi, oranye/persegi = test); catatan test pada klaim 4.1–4.2 yang tidak tereplikasi; 365 angka di 4.4 dicek otomatis ke file sumber (semua cocok, termasuk tanda); judul dokumen jadi "draf 4.1–4.4"; pakem v1.5.5.
- **5 Nov ~09.25:** pengguna: "lanjutkan ke langkah berikutnya". Claude: **4.5 Keterbatasan** di draf Bab 4 (Tabel 4.16, 12 baris: keterbatasan / dampak / yang sudah dilakukan; judul dokumen "draf 4.1–4.5"); **Bahan Bab 5** sebagai Claude Doc baru (5.1 kesimpulan per RQ hanya dari hasil terkonfirmasi di validasi+test; 5.2 kontribusi + batas klaim keterbaruan; 5.3 tujuh saran lanjutan; 5.4 lima saran praktis; catatan penyusun); tautan Bab 5 di draf Bab 4; pakem v1.5.6.
- **5 Nov ~09.35–10.00:** pengguna bertanya apakah "adaptive sampling belum pernah dipakai pada dataset singkong" termasuk keterbaruan → jawab: ya (keterbaruan konteks/penerapan), asal rumusannya sempit dan terdokumentasi. Pengguna: "kerjakan keduanya". Claude: **penelusuran terdokumentasi** (11 pencarian web Inggris/Indonesia + Garuda, 21 halaman dibuka; 6 bukan singkong, 2 tak terbaca) → Claude Doc **Matriks Literatur Bab 2** (hasil, protokol, matriks A/B/C 19 baris, implikasi + susunan Bab 2, catatan penyusun); kalimat keterbaruan sempit dimasukkan ke 5.2 Bahan Bab 5 (+ butir catatan); pakem v1.5.7 (catatan di bawah pernyataan keterbaruan yang masih memakai "yang pertama"; kalimat pakem tidak diubah).

## 4. Hasil kunci — validasi (2.567 citra)

**Protokol ukur:** prediksi per citra 640 px, conf 0,001, NMS seperti `val()` (multi-label, tanpa batas waktu), max_det 300; mAP@50–95 hitung ulang (101 titik, 10 IoU). `val()` Ultralytics 0,2–1,0 poin lebih tinggi (pendamping).

**Struktur data:** semua 1.407 citra CBB dan 1.274 dari 1.562 CBSD (pool latih+validasi 11.547) berasal dari valid/test Roboflow tanpa salinan; CGM/CMD/HEALTHY dari train dengan 2 salinan rotasi. Di validasi: 935 dari 2.567 citra (36,4%) berbaji hitam; kelas dominan CGM 301, CMD 291, HEALTHY 290, CBSD 45, CBB 2.

**E-LEAK (Bab 3):** split acak per citra (seed 42) → 1.554 citra validasi (60,5%) punya kembaran di latih. Angka dilaporkan +6,39 (IK +4,14..+8,56); kemampuan pada 227 citra sama +0,14 (−2,40..+2,59); inflasi di luar kemampuan +6,25 (+2,86..+9,59); efek kembaran +11,05 (+5,41..+18,15; kontrol 108) dan +9,06 (+5,36..+12,53; kontrol 227). Pada 333 citra yang kembarannya di latih: +9,23; pada 108 kontrol: −1,82. mAP@50 +10,01; recall +7,65.

**Ablasi 3 seed (mAP@50–95):** EXP-0 25,98 ± 0,29; EXP-0R 25,66 ± 0,23; EXP-1 24,34 ± 0,03; EXP-2 25,95 ± 0,25; EXP-3 25,25 ± 0,53.

| Kontras | Rata-rata | IK 95% | Verdict |
|---|---|---|---|
| EXP-0R − EXP-0 | −0,33 | −0,77..+0,14 | tidak berbeda |
| EXP-1 − EXP-0R (RQ1 inti) | −1,31 | −1,92..−0,67 | menurunkan konsisten |
| EXP-1 − EXP-0 | −1,64 | −2,26..−0,93 | menurunkan konsisten |
| EXP-2 − EXP-0 (RQ2 inti) | −0,04 | −0,48..+0,45 | tidak berbeda |
| EXP-3 − EXP-0 | −0,74 | −1,26..−0,17 | menurunkan konsisten |
| EXP-3 − EXP-2 | −0,70 | −1,26..−0,16 | menurunkan konsisten |
| EXP-3 − EXP-1 | +0,90 | +0,32..+1,45 | meningkatkan konsisten |
| Interaksi | +0,94 | +0,18..+1,68 | meningkatkan konsisten |

AP per kelas EXP-1 − EXP-0: CBB −3,32, HEALTHY −1,79, CBSD −1,75, CGM −0,93, CMD −0,41. Porsi sampling per kelas EXP-1 hampir tidak berubah (0,93–1,06).

**CONV (seed 42):** EXP-0 25,67 → 25,97 (60 ep, berhenti ep 55); EXP-1 24,35 → 23,88 (berhenti ep 40). EXP-1 − EXP-0: −1,32 (30 ep), −2,09 (60 ep). 30 epoch memadai.

**A1 ukuran objek** (replikasi numpy COCOeval = pycocotools, 24 angka, selisih < 1e-5): kecil 361 (3,7%), sedang 2.452, besar 6.836 (71%). AP kecil 0,1–0,4 di semua model; recall kecil conf ≥ 0,25 praktis nol. EXP-2 − EXP-0: AP kecil +0,06 (−0,22..+0,86), AR kecil −1,30 (−3,85..+0,80), recall kecil −0,28 (−0,85..0,00), semua tidak terbukti. EXP-3 − EXP-0 AP kecil −0,23, AR kecil −1,73 (konsisten). EXP-1 − EXP-0R AP besar −1,06 (−1,77..−0,30, konsisten). EXP-3 − EXP-1 AP besar +0,86, interaksi AP besar +1,06 (konsisten). Koreksi: objek kecil yang lolos conf ≥ 0,25 ada tiga (bukan satu), semuanya EXP-0 seed 123.

**A2 jenis error (τ 0,25):** EXP-0 TP 39,9%, terlewat 44,1%, kotak meleset 11,3%, salah kelas 2,5%, FP 2,15/citra. EXP-1 − EXP-0: TP +1,65, terlewat −2,09, FP +0,38/citra (konsisten). EXP-1 − EXP-0R: TP +1,72 (tidak terbukti), terlewat −1,83 (konsisten), FP total +0,41/citra (latar +0,26, sekelas +0,13; konsisten). EXP-2 − EXP-0: TP −1,53, FP −0,21/citra (konsisten). Salah kelas tidak berubah.

**A3 komposisi EXP-1 (rasio paparan vs uniform):** citra 1 objek 0,82, citra ≥ 7 objek 1,20, citra bertarget kecil 1,37; objek kecil 1,24, sedang 1,12, besar 0,94; objek OOF-TP 0,83, OOF tidak terdeteksi 1,10, OOF salah kelas 1,30, wrong_class 1,29; kelas dan asal tidak bergeser (0,94–1,02); artefak rotasi tidak bergeser (0,98–0,99); EXP-1 menarik 11,5% lebih banyak objek per epoch.

**A4 log crop:** lesi target median 19,8 px → ~320 px (zoom median 16×, p90 70×, maks 99×); 372 crop/epoch (4,1% citra latih) di EXP-2, 513–522 di EXP-3; 755 citra unik; aspek terdistorsi 78%; 91% hanya 1 kotak tersisa. Log EXP-2 identik 100% antar seed (EXP-3 ~25%) → temuan seed DataLoader.

**A5 pinggiran bergaris:** detektor petak `cek_pinggiran_bergaris.py` (PETAK 40, AMBANG_PETAK 0,30, TEKSTUR_MIN 3, MIN_PETAK 2; fungsi `nilai_citra`/`nilai_array`), kalibrasi 40 citra berlabel visual (TP 11, FN 2, FP 0; sensitivitas 85%, spesifisitas 100% pada 38 citra jelas). Validasi: bergaris 27,5% citra tanpa baji (~33% terkoreksi); CBB 42,3%, CBSD 31,9%, CGM 20,4%, CMD 22,2%, HEALTHY 22,2%; asal valid 48%, train 21%, test 31%; artefak rotasi apa pun 53,9%. Sampel 1.000 citra latih: pola sama.

**Fakta data lain:** kotak bersarang (kotak kecil ≥ 90% di dalam kotak lain) ada di 537 dari 2.549 citra validasi (indikasi anotasi tumpang tindih; dihitung untuk panduan audit, bukan dari sampel audit). Aturan bingkai lama: judul cadangan hanya bila EXP-2 meningkat konsisten (tidak terpenuhi).

**Uji baji hitam** (set N 1.632 citra, N_cb 620 citra CBB/CBSD; set W 935; hitam − rata):

| | EXP-0 | EXP-1 | EXP-2 |
|---|---|---|---|
| Porsi keyakinan ke CGM/CMD/HEALTHY | +11,77 (+10,12..+13,54) | +12,18 (+10,58..+13,85) | +12,37 (+10,65..+14,16) |
| AP CBB | −2,00 (−2,84..−1,24) | −1,80 (−2,60..−1,05) | −1,76 (−2,55..−0,97) |
| AP CBSD | −1,34 (−1,95..−0,74) | −1,34 (−2,00..−0,82) | −1,27 (−1,82..−0,74) |

Prediksi teratas ke CGM/CMD/HEALTHY: 16,9% (rata) → 27,8% (hitam). Set W: menghapus baji menaikkan porsi ke CBB/CBSD 2,30 (rata, IK +1,70..+2,99) dan 3,27 (inpaint); AP HEALTHY asli − rata +2,67 (+1,92..+3,32). Selisih efek vs EXP-0: EXP-1 +0,41 (−1,15..+1,93), EXP-2 +0,60 (−0,74..+2,13); AP CBB +0,20/+0,24; tidak terbukti → baji tidak menjelaskan perbedaan strategi.

**Uji pinggiran bergaris** (set G 1.183 citra: G_tr 784 CGM/CMD/HEALTHY, G_cb 390 CBB/CBSD; garis − rata):

| | Subset | EXP-0 | EXP-1 | EXP-2 |
|---|---|---|---|---|
| Porsi ke CBB | G_tr | +1,35 (+0,87..+1,86)* | +1,74 (+1,19..+2,38)* | +1,94 (+1,37..+2,71)* |
| Rata-rata AP CGM/CMD/HEALTHY | G_tr | +0,14 (−0,18..+0,42) | +0,20 (−0,01..+0,45) | +0,31 (+0,04..+0,58)* |
| Porsi ke CBB/CBSD | G_cb | +7,35 (+5,77..+9,19)* | +6,75 (+5,24..+8,48)* | +7,94 (+6,23..+9,86)* |
| AP CBB | G_cb | +0,97 (−0,06..+2,16) | +1,13 (+0,23..+2,27)* | +1,25 (+0,19..+2,38)* |
| AP CBSD | G_cb | −0,87 (−1,65..+0,07) | +0,15 (−0,49..+0,82) | −0,20 (−0,91..+0,63) |

(* terbukti.) Garis − asli porsi CBB +1,03; rata − asli −0,32. Isian menurunkan AP CGM/CMD/HEALTHY ~5,5 poin vs asli (oklusi). Selisih efek vs EXP-0: EXP-1 +0,40 (−0,02..+0,84) tidak terbukti; EXP-2 +0,60 (+0,11..+1,12) terbukti tapi ditafsirkan hati-hati (seed DataLoader). Cek manipulasi: isian median 15,5%; detektor menandai 76% G_garis, 9% G_rata. Geometri: rotasi U(−20°,20°) + geser U(−0,15,0,15), seed `random.Random("garis-<image_id>")`; dikalibrasi pada 70 citra (detektor menangkap 86%, kuartil petak 4/5/6 vs asli 3/5/7, luas isian median 16%); sudut donor uji baji ditolak karena tidak terdeteksi (0/8). Varian G_garis = isian BORDER_REPLICATE ditempel ke citra asli; G_rata = area sama, warna rata-rata garis per komponen. Pra-registrasi di docstring `uji_pinggiran_bergaris.py` (9 model, keluaran `stage_v2_garis/`); analisis `analyze_garis.py prep` → `boot --seed 0..3 --iters 250` → `report`; laporan `stage_v2_garis/analysis/GARIS_REPORT.md`. Dijalankan 1 Nov 09.41 (6,4 menit GPU).

## 5. Hasil test set — Fase 6 (FINAL, 2 Nov 20.30)

Dijalankan sekali oleh pengguna (19.36–20.30, tanpa kegagalan) sesuai `PRA_REGISTRASI_FASE6.md`. Test = `cassava-v7-standardized/test`: 1.282 citra (9 tanpa kotak → 1.273 citra ber-GT), 5.089 objek (CBB 339, CBSD 330, CGM 968, CMD 1.447, HEALTHY 2.005); kecil 190, sedang 1.404, besar 3.495. Hasil: `stage_v2_fase6/analysis/` (`FASE6_RINGKASAN.md`, `ABLATION_REPORT.md`, `conv/`, `ukuran/`, `error/`).

**mAP@50–95 (hitung ulang; val() dalam kurung):** EXP-0 24,04 ± 0,19 (24,74); EXP-0R 23,99 ± 0,53 (24,63); EXP-1 22,81 ± 0,23 (23,27); EXP-2 24,50 ± 0,36 (25,18); EXP-3 23,12 ± 0,03 (23,59).

| Kontras | Validasi | Test | Sama? |
|---|---|---|---|
| EXP-0R − EXP-0 | −0,33, tidak berbeda | −0,06 (−0,63..+0,55), tidak berbeda | ya |
| EXP-1 − EXP-0R (RQ1) | −1,31, turun konsisten | −1,17 (−1,88..−0,47), turun konsisten | ya |
| EXP-1 − EXP-0 | −1,64, turun konsisten | −1,23 (−1,95..−0,49), turun konsisten | ya |
| EXP-2 − EXP-0 (RQ2) | −0,04, tidak berbeda | +0,45 (−0,12..+1,04), tidak berbeda | ya |
| EXP-3 − EXP-0 | −0,74, turun konsisten | −0,92 (−1,60..−0,31), turun konsisten | ya |
| EXP-3 − EXP-2 | −0,70, turun konsisten | −1,37 (−2,07..−0,71), turun konsisten | ya |
| EXP-3 − EXP-1 | +0,90, naik konsisten | +0,31 (−0,30..+0,87), tidak berbeda | **TIDAK** |
| Interaksi | +0,94, naik konsisten | −0,14 (−1,02..+0,65), tidak berbeda | **TIDAK** |

- EXP-2 − EXP-0 di test: per seed +0,05 / +0,64 / +0,67, tetap "tidak terbukti" (IK memuat nol; EXP-0/EXP-2 berbagi urutan data).
- **AP per kelas test:** EXP-0 CBB 19,49, CBSD 24,29, CGM 25,94, CMD 20,81, HEALTHY 29,69; EXP-1 17,66 / 21,26 / 25,28 / 20,88 / 28,98. EXP-1 − EXP-0: CBB −1,83, **CBSD −3,03**, CGM −0,66, CMD +0,07, HEALTHY −0,71 (di validasi kerugian terbesar CBB).
- **Per ukuran (test):** EXP-2 − EXP-0 AP kecil +0,06 (−0,31..+0,41), AR kecil −1,41 (−6,10..+1,00), recall kecil 0 (semua nol) → tidak terbukti. EXP-1 − EXP-0R AP besar −0,98 (−1,86..−0,05) **tidak terbukti** (seed 2026 +0,02; di validasi terbukti); AP kecil +0,37 (+0,01..+0,98) "meningkatkan konsisten" (objek kecil hanya 190). EXP-3 − EXP-0 AP besar −1,10 (konsisten).
- **Jenis error (test):** EXP-1 − EXP-0R FP total +0,36/citra (konsisten; latar +0,25, sekelas +0,11); terlewat −1,21 dan TP +1,05 tidak terbukti. EXP-2 − EXP-0: TP −1,15 (turun konsisten), terlewat +1,98 (naik konsisten), FP −0,21/citra (turun konsisten).
- **CONV (test):** EXP-1 − EXP-0: −1,53 (30 ep), −1,52 (60 ep; IK −2,76..−0,37); perubahan selisih +0,01.
- **Implikasi yang harus direvisi di naskah/pakem:** (1) klaim "crop memulihkan kerugian EXP-1" (interaksi) tidak terkonfirmasi di test; (2) "kerugian EXP-1 terbesar di CBB" bukan temuan umum (test: CBSD); (3) AP objek besar EXP-1 − EXP-0R tidak terbukti di test (searah).

## 6. Audit label manual (C1)

- **Tujuan:** menguji hipotesis RQ3 (label noise). Skor kesulitan tidak bisa membedakan objek sulit dari label keliru; perlu mata manusia. Literatur: Park dkk. NeurIPS 2023 (bagian 4.2 dan 4.5: metode yang mengutamakan sampel sulit memilih banyak sampel berlabel salah — kutipan diverifikasi 2 Nov), Mindermann dkk. ICML 2022 (titik loss tinggi sering noisy/tidak bisa dipelajari).
- **Desain:** 100 citra latih D_image tertinggi (strata tinggi) + 50 acak, maksimal 1 per grup foto; urutan acak, kode A001–A150; strata di `audit_label_150/_kunci/audit_key.csv` (jangan dibuka). 112 citra punya D_image = 1,0 (seri diacak). Rater 2 opsional: 30 citra (20 tinggi + 10 acak) di `LEMBAR_AUDIT_RATER2.xlsx`. Gambar `gambar/Annn.jpg` (kiri asli, kanan kotak bernomor; label runtime-mirror).
- **Lembar v2 (amandemen 1 Nov, sebelum ada penilaian dan sebelum kunci dibuka):** pengguna bukan ahli penyakit tanaman, tak ada akses ahli → ketepatan kelas TIDAK dinilai (keterbatasan). Kategori: Daun tanpa kotak, Kotak bersarang/ganda, Objek bukan daun, Kotak tidak presisi (4 masalah label); Artefak citra (hanya bila area rusak ≥ ~¼ citra), Citra padat/sulit; Tidak ada masalah terlihat; Lainnya. Kolom: Kategori utama, tambahan, Daun tanpa kotak (0/1-2/3-5/>5/Ragu), Keyakinan penilaian (Yakin/Kurang yakin), Kotak presisi?, Catatan. "Daun tanpa kotak" = daun yang tidak berada di kotak mana pun; citra HEALTHY semua daun utuh; citra berpenyakit hanya daun yang polanya mirip daun berkotak. Prioritas: masalah label sebelum kondisi citra. Lembar v1 kosong di `_arsip_v1/`. `PANDUAN_BERGAMBAR.pdf` (9 halaman, 8 contoh dari citra validasi).
- **Rencana analisis (dikunci 2 Nov, `analyze_audit_label.py`, SHA-256 04f2c61646dd98b4…):** utama = selisih proporsi Y_label (≥ 1 dari 4 masalah label, utama atau tambahan) tinggi − acak, IK 95% Newcombe, Fisher pendamping; terdukung bila batas bawah IK > 0. Sensitivitas: tanpa "Kurang yakin"; hanya kategori utama. Eksploratif: per kategori, Y_hilang, Y_presisi, sebaran, kappa rater 2 (bila 30 citra lengkap), validasi skor OOF vs manusia (AUC ≥ 0,70 dan batas bawah > 0,50), paparan EXP-1 via logistik P(Y_label | d_image). Diuji `--uji-tiruan`; statistik dicek ke nilai rujukan. Menolak berjalan bila belum lengkap; menangani "1-2"/"3-5" yang diubah Excel menjadi tanggal. Keluaran `audit_label_150/analysis/`.
- **Daya uji (koreksi 2 Nov):** dengan 100 vs 50 citra, selisih yang terdeteksi andal ~20–25 poin persen (bukan 15–20).
- **Status: SELESAI dan dianalisis (4 Nov).** Penilai pertama 150/150 (`LEMBAR_AUDIT_LABEL.xlsx`, commit `3d4ae7ca`, SHA-256 49760a3b…); penilai kedua 30/30 (`LEMBAR_AUDIT_RATER2.xlsx`, SHA-256 7fa3c53d…, belum di-commit). **Penilai kedua = pengguna sendiri pada hari lain** (dikonfirmasi 4 Nov 20.09) → dilaporkan sebagai **konsistensi intra-rater**, bukan kesepakatan antarpenilai; tidak ada penilai independen (keterbatasan). Lembar v2 kosong dan versi "llll" 30/150 tersimpan di riwayat git (llll tercatat "deleted" di working tree).
- **Hasil utama (pra-registrasi):** Y_label tinggi 22/100 (22,0%) vs acak 36/50 (72,0%); selisih −50,0 poin (IK Newcombe −62,6..−33,6; Fisher p = 4,6e-9) → **berlawanan** dengan hipotesis. S1 (tanpa 39 "Kurang yakin"; 32 di antaranya "Daun tanpa kotak"): 11/88 vs 10/23, −31,0 (−51,4..−11,2). S2 identik dengan utama.
- **Sekunder/eksploratif:** Y_hilang 11/91 vs 29/45 (−52,4; −65,7..−35,6); Y_presisi 14/100 vs 22/49 (−30,9; −45,7..−15,5). Per kategori: daun tanpa kotak 11% vs 60% (−49,0), citra padat 1% vs 18% (−17,0), tidak ada masalah 72% vs 26%; bersarang 4% vs 10%, bukan daun 1% vs 0%, tidak presisi 10% vs 16%, artefak 6% vs 4% (tidak berbeda). 72% citra latih acak bermasalah label (IK 58,3–82,5%).
- **Konsistensi intra-rater (30 citra, 20 tinggi + 10 acak):** setuju 90%; kappa Y_label 0,80 (bootstrap 0,55–1,00), kategori utama 0,84 (0,63–1,00), daun tanpa kotak berbobot 0,88 (0,61–1,00; 28 citra). Tiga ketidaksepakatan semuanya di strata tinggi (A100, A108, A112: penilai kedua "Daun tanpa kotak", penilai pertama "Tidak ada masalah"); bila dihitung bermasalah, tinggi menjadi 25%.
- **Validasi skor OOF:** 1 − label_quality vs Y_label AUC 0,50 (0,40–0,60) pada 150 citra → tidak valid (kriteria AUC ≥ 0,70 dan batas bawah > 0,50); pada acak 0,78 (0,65–0,89). missing_score vs Y_hilang 0,66 (semua) / 0,72 (acak); badloc_score vs Y_presisi 0,64 / 0,68. Porsi lq < 0,5 (rasio 1,07) tidak ditafsirkan.
- **Paparan EXP-1 (pra-registrasi):** logistik b0 1,88, b1 −2,86; rasio 0,92 (0,90–0,94). **Tidak dipakai**: pemeriksaan pasca-hoc saat verifikasi (bukan pra-registrasi) menunjukkan hubungan tidak monoton — di 50 citra acak (D 0,15–0,89) P(Y_label) naik bersama D_image (AUC 0,71; IK 0,54–0,86; 59% pada D ≤ 0,40, 80%, 75%, 90% pada 0,70–0,89), lalu 22% pada D ≥ 0,99. 122 citra puncak (D ≥ nilai terendah strata tinggi, 1,4% citra latih) menerima 2,2% tarikan EXP-1. Pengganti pasca-hoc: lihat butir pasca-hoc di bawah.
- **Tafsiran (draf 4.3.3):** citra tersulit tidak didominasi masalah label yang terlihat; sejalan dengan rumus D (hanya dinilai pada kotak GT, jadi daun tanpa kotak tidak menaikkan D; D ≥ 0,99 berarti hampir semua objek terlewat dan kecil, atau kelas lain). Label kelas keliru (A3: objek OOF salah kelas ×1,30) tidak teruji. Anotasi tidak lengkap umum di data, sejalan dengan kenaikan FP EXP-1 (belum diuji).
- **Verifikasi Tahap A (4 Nov, semua lolos):** strata cocok dengan manifest (tinggi D 0,99–1,00, rank ≤ 122; acak 0,15–0,89); gambar `gambar/Annn.jpg` cocok piksel dengan citra di kunci (MAE ≤ 2,7 vs ≥ 28,6 citra lain); jumlah/kelas kotak di lembar cocok dengan label runtime-mirror; kode rater 2 = flag di kunci; semua statistik dihitung ulang dengan scipy/sklearn (sama); rumus Newcombe dicek ke contoh Newcombe 1998 (56/70 vs 48/80 → 0,0524..0,3339). Skrip: `_panduan_kerja/verif_audit/`.
- **Keluaran:** `audit_label_150/analysis/` (`AUDIT_REPORT.md`, `audit_results.json`, `audit_per_citra.csv`); arsip run sebelum persetujuan `audit_label_150/analysis_run1_20261104_1637_sebelum_persetujuan/`; `audit_label_150/BEKU_LEMBAR_AUDIT_20261104.txt`.
- **Analisis pasca-hoc (disetujui 4 Nov 20.09; `analyze_audit_pascahoc.py`, SHA-256 c2db4d7c…, spesifikasi di docstring ditulis sebelum dijalankan; keluaran `audit_label_150/analysis_pascahoc/`; angka utama diverifikasi ulang dengan sklearn, sama):**
    - P1 komposisi (tinggi / acak / populasi): satu kotak 87% / 28% / 33%; objek besar 92% / 65% / 69%; objek OOF Classification_Confusion 93% / 0,5% / 3,1%; OOF TP 0% / 41% / 36%; OOF terlewat 7% / 45% / 48%; CBB 28% / 8% / 12%; HEALTHY 12% / 40% / 25%; asal valid+test Roboflow 46% / 24% / 24%; baji hitam 26% / 38% / 37%*; bergaris (tanpa baji) 37% / 19% / 27%* (*sampel 1.000); swap>0 94% / 2% / 10%; median label_quality 0,58 / 0,66 / 0,70. → citra tersulit = satu daun besar yang kelasnya tidak disepakati OOF (gejala ambigu atau label kelas keliru; audit tak bisa membedakan).
    - P2 Y_label per kelas dominan (tinggi vs acak): CBB 6/28 vs 1/4, CBSD 1/18 vs 2/8, CGM 2/16 vs 5/7, CMD 7/26 vs 9/11, HEALTHY 6/12 vs 19/20; selisih Mantel–Haenszel −40,1 poin (IK bootstrap −55,3..−24,0).
    - P3 AUC D_image vs Y_label di acak 0,712 (0,549–0,862).
    - P4 paparan tanpa asumsi monoton: pita kuartil Sisa (batas 0,329 / 0,533 / 0,702 / 0,990) Y_label 7/14, 12/16, 8/10, 9/10; puncak 22/100; porsi tarikan EXP-1 14,3 / 21,7 / 27,8 / 33,9 / 2,2% vs citra 24,7 / 24,6 / 24,7 / 24,7 / 1,4%. E1 rasio 1,051 (0,999–1,106; 73,0% → 76,7%) belum dapat ditentukan; E2 logistik acak 1,060 (1,007–1,115) lebih sering; E3 tanpa gradien 0,994; E1-S1 tanpa "Kurang yakin" 1,103 (0,911–1,215). Tafsiran: perubahan paparan kecil ke arah mana pun (≈ ¾ citra latih sudah bermasalah).
    - Implikasi: label kelas keliru pada citra tersulit = pertanyaan terbuka utama; saran lanjutan: penilaian kelas oleh ahli pada 100 citra itu.

## 7. Status jawaban rumusan masalah (cek 2 Nov, diperbarui dengan test final)

- **RQ1 — terjawab dan terkonfirmasi:** tidak; mixture adaptive sampling menurunkan mAP secara konsisten (validasi −1,31; test −1,17). Hipotesis Δ > 0 ditolak. CONV: kerugian bertahan pada 60 epoch.
- **RQ2 — terjawab (hasil nol):** tidak terbukti meningkatkan (validasi −0,04; test +0,45, keduanya tidak terbukti). Untuk mAP keseluruhan, IK validasi membatasi efek < ~±0,5 poin. Untuk lesi kecil: belum dapat disimpulkan (objek kecil 3,7%, AP kecil di lantai) → tulis "tidak ada bukti peningkatan", bukan "terbukti tidak berpengaruh".
- **RQ3 — terjawab (4 Nov):** komposisi sampel, pergeseran jenis error, ukuran objek, log crop, uji jalan pintas, dan audit label sudah ada. Hipotesis label noise **tidak terdukung untuk masalah label yang terlihat (arah berlawanan, 22% vs 72%)**; label kelas keliru tidak teruji (keterbatasan), padahal pasca-hoc menunjukkan 93% objek citra tersulit diprediksi OOF ke kelas lain; EXP-1 hanya sedikit menggeser paparan ke citra bermasalah (rasio 1,05, belum dapat ditentukan). Hubungan faktor → hasil masih asosiasi (tulis "sejalan dengan", bukan "disebabkan oleh"); penjelasan hasil nol EXP-2 (objek kecil sedikit, dosis crop 4,1%, distorsi 78%) belum diuji kausal. Uji jalan pintas = intervensi (kausal) dan menyingkirkan artefak sebagai penjelas perbedaan.

## 8. Keterbaruan dan posisi terhadap literatur (diskusi 2 Nov)

- **Pandangan pengguna (disetujui):** metode yang tidak mengungguli baseline tetap bermanfaat bila alasannya dijelaskan dengan bukti kuat. Selaras dengan bingkai studi analitis. Catatan: kekuatan hasil negatif ada pada penjelasan (audit menentukan); bedakan "tidak terbukti" vs "terbukti tidak"; pastikan ke pembimbing (risiko "pembimbing mensyaratkan metode unggul"; cadangan EXP-1c).
- **Adaptive sampling berbasis error deteksi BUKAN hal baru.** Membuktikan "belum pernah ada" tidak mungkin; hanya bisa penelusuran terdokumentasi dan klaim "sepanjang penelusuran penulis hingga [tanggal]". Hindari kata "pertama".
- **Literatur terdekat (diverifikasi 2 Nov):**
  - Loshchilov & Hutter (2015), Online Batch Selection — https://arxiv.org/pdf/1511.06343
  - Shrivastava dkk. (CVPR 2016), OHEM — https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html
  - Yu dkk. (2018), Loss Rank Mining, YOLOv2, +5% mAP KITTI, +2% VOC — https://arxiv.org/abs/1804.04606
  - Chen dkk. (2020), Stitcher → Dynamic Scale Training (umpan balik optimisasi memandu penyiapan data) — https://arxiv.org/abs/2004.12432v1
  - Cap dkk. (2023), Hard-sample Re-mining, penyakit mentimun/tomat, Computers and Electronics in Agriculture; pemilihan sampel sulit sembarangan menurunkan kinerja — https://arxiv.org/abs/2309.01903v2 · https://www.sciencedirect.com/science/article/abs/pii/S0168169923007639
  - Park dkk. (NeurIPS 2023), Robust Data Pruning under Label Noise — https://papers.neurips.cc/paper_files/paper/2023/file/ebb6bee50913ba7e1efeb91a1d47a002-Paper-Conference.pdf
  - Mindermann dkk. (ICML 2022), RHO-LOSS — https://proceedings.mlr.press/v162/mindermann22a.html
  - Xie dkk. (arXiv Maret 2026), AFSS: sampling citra per epoch untuk YOLO berdasarkan min(presisi, recall) per citra; tier mudah/sedang/sulit; citra sulit tiap epoch; terutama percepatan (COCO YOLO11s AP 47,0 → 47,2) — https://www.alphaxiv.org/abs/2603.17684
  - Sudah di pakem: Kalezhi dkk. 2025, Ramos dkk. 2025, Ahmed & Abbas 2026 (CropCop), Yu dkk. 2026 (MDDC), Cap dkk. 2023.
- **Yang membedakan (sepanjang penelusuran ~12 pencarian, bukan bukti):** skor kesulitan OOF (bukan sinyal saat pelatihan); mixture dengan komponen seragam + kontrol EXP-0R; split per grup + E-LEAK; 3 seed + aturan keputusan terkunci + test sekali; analisis penyebab dengan uji intervensi jalan pintas dan audit label; konteks deteksi multi-kelas penyakit singkong.
- **Penelusuran 5 Nov (matriks literatur):** **TIDAK benar** bahwa sampel sulit belum pernah dipakai pada singkong. Ramcharan dkk. (pracetak arXiv 1805.08692; Frontiers Plant Sci 2019) — deteksi objek SSD MobileNet daun singkong Tanzania, **hard negative mining bawaan SSD (maks 3.000, rasio 3:1)**, tingkat anchor. Klasifikasi singkong: Sambasivam & Opiyo 2021 (class weights, SMOTE, focal loss), Liu dkk. 2022 Frontiers (focal loss), Zhong dkk. 2022 Agriculture (FAMP-Softmax), IET Image Processing 2022 (smooth-Taylor CE untuk label noise), Che dkk. 2025 PeerJ CS (progressive learning = tingkat kesulitan triplet loss), Kaggle (focal, bi-tempered, Taylor). Deteksi singkong tanpa mekanisme: Kalezhi dkk. 2025 (YOLOv9/GELAN). Garuda: judul "singkong yolo" 0; "penyakit singkong" 24 (10 teratas: sistem pakar/klasifikasi). Rumusan yang didukung: "belum ditemukan sampling **tingkat citra** berbasis skor kesulitan pada **deteksi objek** penyakit singkong"; AFSS (2026) = sampling citra untuk YOLO tetapi di luar singkong. Sisa: ulangi di Google Scholar/Scopus (akses kampus), baca teks penuh Ramcharan 2019/Kalezhi/Che/Zhong, rujukan focal loss asli (Lin dkk. 2017) belum diverifikasi.
- **Keterbaruan skripsi (empiris-analitis, inkremental), urut dari terkuat:** (1) bukti terkontrol bahwa sampling berbasis error merugikan di konteks ini (dengan EXP-0R, terkonfirmasi test); (2) penjelasan mengapa (bergantung audit); (3) artefak augmentasi Roboflow (baji hitam, pinggiran bergaris) sebagai jalan pintas model — perlu cek literatur dulu sebelum diklaim (shortcut learning umum: Geirhos dkk. 2020); (4) hasil nol context crop + batasnya; (5) protokol split per grup + besaran inflasi (pendukung; kebocoran umum: Kapoor & Narayanan 2023).
- **Kerangka pernyataan keterbaruan (bahan; kalimat final ditulis pengguna):** "Penelitian ini tidak mengusulkan metode baru. Kebaruannya adalah evaluasi terkontrol terhadap dua strategi pelatihan berbasis error pada YOLO11n untuk deteksi penyakit singkong, yang menunjukkan kapan dan mengapa strategi tersebut tidak membantu, termasuk bukti bahwa artefak augmentasi pada dataset publik menjadi jalan pintas model."
- **Mengapa penelitian lain melaporkan peningkatan sedangkan di sini merugikan:** literatur tidak seragam (AFSS gain kecil; Cap dkk. sampel sulit sembarangan merugikan). Perbedaan kondisi: (a) yang ditekankan vs yang dinilai mAP — objek besar 71%, paparan ×0,94, AP besar −1,06 [terukur, asosiasi]; (b) presisi–recall: FP naik +0,41/citra [terbukti]; (c) kualitas label: paparan +30% ke objek OOF salah kelas [audit]; (d) tingkat sampling: region (OHEM/LRM) vs seluruh citra [belum diuji]; (e) sinyal statis OOF 12 epoch vs dinamis [belum diuji]; (f) kapasitas YOLO11n 2,6 juta parameter [dugaan]. Tersingkir: variasi acak (3 seed + test), kurang epoch (CONV), jalan pintas artefak (sama kuat).
- **Jawaban penguji "apa bedanya dengan penelitian sebelumnya"** (urutan posisi → rancangan → temuan → manfaat): akui metode bukan baru (OHEM 2016, AFSS 2026); bedanya pertanyaan (menguji apakah membantu dan mengapa); rancangan (OOF, EXP-0R, split per grup + inflasi ~6 poin, 3 seed + aturan terkunci + test sekali); temuan (turun ~1,3 poin, terkonfirmasi test; pergeseran fokus ke objek ambigu/padat; artefak rotasi sebagai jalan pintas tetapi bukan pembeda); manfaat (periksa kualitas label dan artefak augmentasi sebelum sampling berbasis kesulitan). Pertanyaan lanjutan: guna hasil negatif; bukti label bermasalah (audit); "apakah yang pertama" → tidak mengklaim.
- **Tawaran:** penelusuran literatur — **dikerjakan 5 Nov** (matriks literatur). Masih terbuka: menambahkan Ramcharan dkk. 2019, AFSS, Loss Rank Mining, OHEM ke tabel "penelitian terdekat" di pakem (belum diminta); mengganti kata "yang pertama" di pernyataan keterbaruan pakem (pengguna yang menulis).

## 9. File penting di `D:\UNDIP\SKRIPSI`

- **Runner:** `jalankan_fase3.ps1` (`-UjiBaji`, `-Conv`), `jalankan_fase5a.*`, `jalankan_uji_garis.*`, `jalankan_fase6.ps1/.bat` (sudah dijalankan; menolak ulang), `cek_siap_fase3.bat`, `jalankan_malam_A/B.bat`, `rapikan_git.bat`. Runbook lama `MALAM_INI.md`.
- **Training/skor:** `stage_8_1_group_oof_profiling.py`, `stage_8_2_build_difficulty_manifest.py`, `stage_8_3_adaptive_pipeline.py`, `stage_8_4_run_ablation_experiments.py`, `research_utils/detection_metrics.py`.
- **Analisis:** `predict_eleak_per_image.py`, `analyze_eleak_per_image.py`, `analyze_eleak_did.py`, `analyze_ablation.py` (+ `--gt-name --nama-data --judul`), `analyze_conv.py` (+ `--pred-30-dir --pred-60-dir --nama-data`), `analyze_ukuran.py` / `analyze_error.py` (+ `--per-image-dir --gt-name --nama-data`; default tidak berubah, regresi identik), `cek_coco_ukuran.py`, `analyze_crop.py`, `analyze_komposisi.py`, `cek_struktur_kelas.py`, `uji_baji_hitam.py`, `analyze_baji.py`, `cek_pinggiran_bergaris.py`, `analyze_pinggiran.py`, `uji_pinggiran_bergaris.py`, `analyze_garis.py`, `val_test_fase6.py`, `analyze_fase6.py`.
- **Audit:** `build_label_audit.py` (v1), `revisi_lembar_audit.py` (v2; `--bangun-ulang` hanya bila kosong), `buat_panduan_bergambar.py`, `analyze_audit_label.py`, `analyze_audit_pascahoc.py` (pasca-hoc: `fitur` lalu `analisis`); folder `audit_label_150/` (lembar, PDF, README, `_arsip_v1/`, `_kunci/`, `gambar/`).
- **Gambar:** `buat_gambar_bab4.py` → `_gambar_bab4/gbr4_1`–`gbr4_6`; `buat_gambar_jalan_pintas.py` → `gbr4_7_jalan_pintas.jpg`; `buat_gambar_test.py` → `gbr4_8_validasi_vs_test.png` (Gambar 4.6 draf Bab 4).
- **Pra-registrasi:** `PRA_REGISTRASI_FASE6.md` (Status: DISETUJUI).
- **Hasil:** `stage_v2_ablation/analysis/` (+ `ukuran/`, `error/`, `crop/`, `pinggiran/`, `komposisi/`, `coco_cek/`), `_analisis_rq1_audit/`, `_analisis_struktur_kelas/`, `stage_v2_baji/` (+ `analysis/`, `_cek_visual/`), `stage_v2_baji_exp12/`, `stage_v2_garis/`, `stage_v2_conv60/`, `stage_v2_eleak_sim/per_image_v2/analysis/`, `stage_v2_fase6/` (+ `MULAI.txt`, `SELESAI.txt`), `logs_v2/` (Fase 6: `fase6_jalan_20261102_1936.log`).
- **Kerja sementara (gitignore):** `_panduan_kerja/` (montase panduan, uji tiruan audit, uji kering Fase 6, regresi, cadangan skrip lama di `cadangan_skrip/`).
- **Literatur:** `Literatur terkait penelitian YOLO singkong.md` (erratum 30 Okt).
- `.gitignore` sudah ditambah pola `stage_v2_garis/`, `_panduan_kerja/`, `stage_v2_fase6/...`.

## 10. Isi dokumen Claude

- **Pakem v1.5.6 (5 Nov):** RQ terkunci (tabel di atas), protokol data (+ kesesuaian metadata Roboflow), E-LEAK sebagai validasi protokol (Bab 3), rancangan eksperimen, audit v2 + rencana analisis, checklist Fase 5 dan Fase 6 (semua dicentang), risiko (termasuk "penilai audit bukan ahli", diperbarui), struktur naskah Bab 4 (4.1 RQ1, 4.2 RQ2, 4.3 RQ3, 4.4 test, 4.5 keterbatasan), log versi sampai 1.5.6 (1.5.4: intra-rater + pasca-hoc; 1.5.5: draf 4.4; 1.5.6: draf 4.5 + Bahan Bab 5). Struktur naskah: 4.4 "Konfirmasi pada test set"; checklist Fase 7 ditambah butir draf 4.1–4.4. **Memuat bagian baru "Hasil test set (Fase 6)" dan "Hasil audit label"**; ringkasan penelitian ditambah satu kalimat hasil utama; paragraf AP per kelas diberi catatan test (interaksi tidak terkonfirmasi, kerugian terbesar CBSD).
- **Draf Bab 4:** 4.1 (RQ1; Tabel 4.1–4.4, Gambar 4.1–4.2; Pembahasan 4.1.5 dikoreksi 4 Nov: dugaan penguatan label kelas keliru tidak bisa diuji audit; 5 Nov: catatan test pada urutan kerugian per kelas (4.1.2) dan AP objek besar (4.1.5), Jawaban RQ1 + konfirmasi test), 4.2 (RQ2; Tabel 4.5–4.7, Gambar 4.3–4.4; paragraf interaksi "crop memulihkan" dan klaim AP/AR kecil EXP-3 kini diberi catatan "tidak terkonfirmasi di test"), 4.3 (4.3.1 baji Tabel 4.8; 4.3.2 garis Gambar 4.5, Tabel 4.9; **4.3.3 audit lengkap dengan Tabel 4.10** dan paragraf hasil, konsistensi intra-rater, validasi skor, paparan pra-registrasi, **analisis pasca-hoc + Tabel 4.11 komposisi**, Pembahasan; 4.3.4 sintesis kini **Tabel 4.12** + baris "anotasi tidak lengkap" + catatan test; **Jawaban RQ3 ditulis** (termasuk temuan pasca-hoc)), bagian "Bahan Bab 3" tautan ke dokumen Bab 3, **4.4 Konfirmasi pada test set** (4.4.1 kinerja, Tabel 4.13; 4.4.2 uji hipotesis, Gambar 4.6 + Tabel 4.14; 4.4.3 per kelas/ukuran/error/CONV, Tabel 4.15; Pembahasan; Ringkasan konfirmasi per RQ); Jawaban RQ2 + konfirmasi test dan pencabutan klaim "crop memulihkan"; **4.5 Keterbatasan** (Tabel 4.16: satu dataset/arsitektur/resolusi; satu bentuk tiap strategi; seed DataLoader; objek kecil sedikit; audit satu penilai non-ahli tanpa kelas; paparan; anotasi tidak lengkap 72%; artefak rotasi; asosiasi; multiplisitas; riwayat test + CONV satu seed; metrik hitung ulang vs val()); tautan ke Bahan Bab 5; catatan penyusun (semua butir lama dicentang; butir baru terbuka: Gambar 4.6 dan Tabel 4.14 berisi angka sama, pilih salah satu bila perlu ringkas).
- **Bahan Bab 5 (5 Nov):** 5.1 Kesimpulan (RQ1 −1,31 val / −1,17 test vs EXP-0R, bertahan 60 ep; RQ2 −0,04 / +0,45 tidak berbeda, lesi kecil "tidak ada bukti", EXP-3 −0,74/−0,92; RQ3 paparan objek besar + FP +0,41/+0,36, label noise terlihat tidak terdukung 22% vs 72%, citra tersulit 93% OOF salah kelas, jalan pintas sama kuat, hasil nol EXP-2 sejalan dengan objek kecil/dosis/distorsi); 5.2 Kontribusi (bukan metode baru; OHEM, LRM, AFSS; pembeda rancangan; jangan pakai "pertama"); 5.3 Saran lanjutan (1 ahli menilai kelas 100 citra tersulit, 2 EXP-1c sadar noise [Park 2023, RHO-LOSS], 3 sinyal dinamis tingkat objek, 4 lengkapi anotasi subset validasi + ukur ulang FP, 5 lesi kecil: data/resolusi/potongan/crop aspek, 6 latih tanpa artefak rotasi, 7 lebih banyak seed + arsitektur lebih besar); 5.4 Saran praktis (5 butir); catatan penyusun (verifikasi rujukan Geirhos 2020, SAHI, Kapoor & Narayanan 2023; kalimat keterbaruan ditulis pengguna; gate G0; keputusan EXP-1c).
- **Bahan Bab 3 (4 Nov):** 3.9 diperbarui (penilaian ulang intra-rater, pembekuan, pelaksanaan run 16.37/19.33, verifikasi, analisis pasca-hoc); 3.11 keterbatasan audit ditambah (satu penilai; konsistensi hanya dari penilaian ulang 30 citra oleh penilai yang sama; paparan pra-registrasi tidak dapat dipakai, pengganti pasca-hoc bertumpu pada 50 citra acak).
- **Bahan Bab 3:** 3.1 alur; 3.2 dataset dan standarisasi label; 3.3 split per grup + E-LEAK (Gambar 3.1, Tabel 3.1); 3.4 skor kesulitan OOF + kualitas label; 3.5 strategi dan desain (Tabel 3.2 konfigurasi, Tabel 3.3 hiperparameter, CONV); 3.6 protokol evaluasi; 3.7 statistik; 3.8 uji jalan pintas; 3.9 audit; 3.10 test set; 3.11 keterbatasan; lampiran skrip.
- **Laporan Fase 5 dan laporan malam B:** penomoran RQ lama (catatan di atas dokumen); kalimat yang menyiratkan kebocoran Roboflow sudah dihapus.

## 11. Posisi sekarang dan langkah berikutnya

| Fase | Status |
|---|---|
| 0 (persetujuan pembimbing) | terbuka |
| 1–4 | selesai |
| 5 (analisis RQ3) | **selesai** (audit dianalisis 4 Nov) |
| 6 (test set) | **selesai 2 Nov 20.30**; hasil sudah di pakem v1.5.3; draf 4.4 belum |
| 7 (penulisan) | draf 4.1–4.5 lengkap (5 Nov), Bahan Bab 3 dan Bahan Bab 5 lengkap; Bab 1 (penyesuaian 3 RQ + keterbaruan) belum; naskah final ditulis pengguna |

**Tugas yang tertunda (urut):**
1. ~~Audit, analisis, 4.3.3, Jawaban RQ3, pakem~~ — **selesai 4 Nov.** Penilai kedua = pengguna sendiri (intra-rater) dan analisis pasca-hoc **selesai 4 Nov malam**; tidak ada sisa untuk audit. Saran Bab 5: penilaian ulang kelas oleh ahli pada 100 citra tersulit.
2. ~~Draf 4.4, revisi klaim 4.1–4.2, draf 4.5, Bahan Bab 5~~ — **selesai 5 Nov pagi.**
3. ~~Draf 4.5 dan bahan Bab 5~~ — selesai 5 Nov. Langkah berikutnya yang tersisa untuk Claude: bahan penyesuaian Bab 1 (tiga RQ, tujuan, manfaat, batasan, kerangka keterbaruan) bila pengguna memintanya; verifikasi rujukan yang ditandai di catatan penyusun Bab 4 dan Bab 5.
4. Bab 1 disesuaikan dengan 3 RQ dan pernyataan keterbaruan (pengguna menulis).
5. Opsional (menunggu jawaban): penelusuran literatur sistematis; tambah AFSS/LRM/OHEM/Stitcher ke tabel penelitian terdekat; cek literatur artefak augmentasi Roboflow sebagai jalan pintas.
6. Commit manual oleh pengguna. **Pengguna commit `eec3b35d` (4 Nov 20.09) dan `5c480ff1` (4 Nov 20.25: pasca-hoc + ringkasan 4nov). Yang belum di-commit per 5 Nov pagi:** `buat_gambar_test.py`, folder `_gambar_bab4/` (seluruhnya belum pernah di-commit, termasuk gbr4_1–gbr4_8), dan `ringkasan-chat-skripsi-5nov.md`. Pengguna commit sendiri. HEAD `5c480ff1`. (Daftar lama sebelum commit pengguna: berkas Fase 5 (`uji_pinggiran_bergaris.py`, `analyze_garis.py`, `jalankan_uji_garis.*`, `analyze_komposisi.py`, `buat_gambar_bab4.py`, `stage_v2_ablation/analysis/komposisi`, `_gambar_bab4`), audit (`revisi_lembar_audit.py`, `buat_panduan_bergambar.py`, `analyze_audit_label.py`, `audit_label_150/` setelah audit selesai), Fase 6 (`PRA_REGISTRASI_FASE6.md`, `jalankan_fase6.*`, `val_test_fase6.py`, `analyze_fase6.py`, `stage_v2_fase6/analysis`, `MULAI.txt`, `SELESAI.txt`), `buat_gambar_jalan_pintas.py`, `gbr4_7_jalan_pintas.jpg`, skrip analisis yang diberi opsi, `.gitignore`. Pengguna sendiri sempat commit `9a3a444a` (analyze_baji.py, analyze_conv.py); HEAD saat Fase 6 `0e35022a`.
7. Gate G0: pastikan pembimbing menerima bingkai studi analitis dengan hasil negatif.

## 12. Catatan kesalahan dan koreksi (agar tidak terulang)

- Klaim kebocoran Roboflow (51,71%) = kesalahan audit internal → dihapus dari RQ dan naskah.
- Jumlah objek kecil yang lolos conf ≥ 0,25: tiga, bukan satu.
- Perkiraan daya uji audit: dikoreksi menjadi ~20–25 poin persen.
- Skrip kerja panduan sempat membaca label poligon (clean-split) sebagai kotak; tertangkap, diganti runtime-mirror; tidak ada hasil yang terdampak.
- Commit file dengan nama lama bisa mengirim isi basi → selalu nama baru + cek md5.
- `exit` di `.ps1` dengan `-NoExit` diganti `return`.
- `global MODELS` harus di awal `main()` (analyze_baji).
- Kutipan Park dkk. 2023 diverifikasi ke isi makalah (bukan abstrak).
- 4 Nov: analisis audit (membuka kunci) dijalankan sebelum persetujuan pengguna; balasan berupa pertanyaan dianggap persetujuan. Dampak ilmiah tidak ada (lembar dan skrip sudah dibekukan, run ulang identik), tetapi dicatat di pakem dan Bab 3. Aturan: langkah tak terbalikkan menunggu "setuju" eksplisit.
- Model logistik paparan pada rencana audit mengasumsikan hubungan monoton; tidak terpenuhi (ketahuan saat verifikasi). Untuk rencana berikutnya, cek bentuk hubungan sebelum memakai model parametrik.

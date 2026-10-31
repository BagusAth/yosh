# Ringkasan konteks chat skripsi (per 30 Oktober 2026)

Tempel dokumen ini di awal chat baru, lalu tulis "lanjutkan dari sini". Acuan utama tetap **pakem** (Claude Doc, v1.3.2): https://claude.ai/artifact/E5wWxH8XooniNv3FWmYJqN

## 1. Identitas dan aturan kerja

- **Mahasiswa:** Bagus Athallah, S1 UNDIP. Pembimbing sulit ditemui, jadi penelitian dilanjutkan dengan judul kerja sebagai default. Gate G0 (persetujuan pembimbing) masih terbuka.
- **Folder dan perangkat:** `D:\UNDIP\SKRIPSI` di laptop Windows 11 dengan GPU RTX 4060 Laptop. Lingkungan `.venv-cuda`: Ultralytics 8.3.235, PyTorch 2.9.1+cu130. Branch git `fase1-skor-kesulitan-v2`.
- **GPU hanya dijalankan pengguna dan hanya saat ia memintanya.** Caranya dengan klik dua kali file `.bat`. Kontrol layar (computer use) ditolak pengguna, jadi jangan ditawarkan lagi.
- **Akses Claude ke laptop:** shell Linux VM, tanpa GPU dan tanpa torch. Batas sekitar 45 detik per perintah. Tidak boleh menghapus file. Jangan menjalankan perintah git yang menulis, karena meninggalkan file `.lock`; commit dilakukan lewat `.bat`.
- **Analisis berat:** file di-stage ke workspace cloud, dihitung di sana, lalu hasilnya ditulis balik ke folder.
- **Test set (1.282 citra) terkunci** sampai evaluasi final (Fase 6), dan dijalankan sekali saja.
- **Aturan PowerShell 5.1:**
  - File `.ps1` harus ASCII saja.
  - Argumen array lewat `-File` diterima sebagai teks.
  - Keluaran Python dialirkan lewat `Invoke-Py` agar tercatat di log.
- **Commit hanya bila diminta pengguna.** Pengguna juga sering commit sendiri.

## 2. Bingkai penelitian

- **Judul kerja:** studi analitis strategi pelatihan berbasis error pada YOLO11n untuk deteksi gejala penyakit daun singkong. Dua strategi yang diuji: mixture adaptive sampling berbasis skor kesulitan out-of-fold, dan targeted context augmentation (context crop).
- **Dataset:** Roboflow Cassava v7, 5 kelas: CBB, CBSD, CGM, CMD, HEALTHY.
- **Split bersih per grup:** 8.980 latih / 2.567 validasi / 1.282 test, dari 7.920 grup foto.
- **Alasan dibingkai ulang (27 Oktober):** sampling adaptif awalnya justru menurunkan performa. Penyebabnya: label noise dan anotasi tidak lengkap, efek sampling dengan replacement yang tercampur, dan bug pencocokan v1. Penelitian diubah dari "metode usulan" menjadi studi analitis.
- **Rumusan masalah:**
  - **RQ1:** seberapa andal audit kebocoran berbasis nama file (B), dan seberapa besar metrik menggelembung bila data dibagi per citra setelah augmentasi (A).
  - **RQ2:** apakah mixture adaptive sampling (EXP-1) lebih baik daripada sampling uniform (EXP-0R, EXP-0).
  - **RQ3:** apakah context crop (EXP-2) membantu, khususnya pada lesi kecil.
  - **RQ4:** faktor yang menjelaskan berhasil atau gagalnya kedua strategi (komposisi sampel, audit label, pergeseran jenis error).
- **Eksperimen (30 epoch, batch 16, imgsz 640):**

| Eksperimen | Konfigurasi |
|---|---|
| EXP-0 | baseline uniform |
| EXP-0R | uniform dengan replacement (kontrol) |
| EXP-1 | mixture P(k), λ = 0,7 |
| EXP-2 | context crop |
| EXP-3 | gabungan EXP-1 dan EXP-2 |
| E-LEAK | simulasi split acak per citra |
| CONV | 60 epoch |
| EXP-1c | opsional |

  Seed: 42, 123, 2026. Nomor `--exp`: 0, 4 (EXP-0R), 1, 2, 3.
- **Aturan keputusan (TERKUNCI):**
  - "Meningkatkan/menurunkan secara konsisten" berlaku bila selisihnya bertanda sama di ketiga seed dan IK 95% tidak memuat nol. Selain itu "tidak terbukti berbeda".
  - Aturan bingkai: judul cadangan (context crop sebagai metode usulan) hanya dipakai bila EXP-2 meningkat secara konsisten terhadap EXP-0.
- **Skor kesulitan v2 (TERKUNCI):** pencocokan sadar kelas dengan τ = 0,25.
  - D_i = 0,4(1−C*) + 0,4(1−IoU*) + 0,2·S_i, ditambah 0,35 bila kelasnya salah. Bila tidak ada kandidat sekelas, D_i = 0,8 + 0,2·S.
  - D_image = 0,5·max + 0,5·mean.
  - Sanity gate lolos: citra sulit 14,35%, DR 0,6173, ESS 0,9099, 755 target kecil.
  - Taksonomi v2: TP 35,7%, missed 48,2%, lokalisasi 10,2%, kelas 3,1%. 112 citra punya D = 1,0.

## 3. Kronologi

- **27 Okt:**
  - Mempelajari ide dan menganalisis kegagalan adaptive sampling.
  - Bingkai diubah menjadi studi analitis, lalu pakem v1.0 dibuat.
  - Penilaian orisinalitas, kontribusi, dan keterbaruan (v1.1).
  - Deep research referensi 2023–2026: 15 Scopus dan 4 arXiv pembanding.
- **28 Okt:**
  - Lanjut tanpa menunggu pembimbing.
  - Fase 1 diimplementasikan: pencocokan v2, skor kualitas label, mode EXP-0R, log crop, dan uji regresi.
  - Ditemukan bahwa klaim "51,71% validasi bocor" adalah **positif palsu** dari audit berbasis nama file. Audit berbasis isi menemukan 0 grup lintas split. RQ1 diubah menjadi gabungan A + B (v1.3).
  - Malam 1: manifest v2, EXP-0 seed 123 dan 2026, dan E-LEAK.
- **29 Okt:**
  - Analisis per citra dan bootstrap klaster E-LEAK, ditambah uji efek kembaran (selisih-dalam-selisih).
  - Temuan struktur kelas dan baji hitam rotasi.
  - Skrip Fase 3 dan lembar audit label disiapkan.
  - Ditemukan bug prediksi: NMS bawaan `predict()` memotong 49–179 citra per model. Sudah diperbaiki.
  - `.gitignore` diperbarui dan `rapikan_git.bat` dibuat. Belum diketahui apakah sudah dijalankan.
- **30 Okt:**
  - Malam A selesai (8 run, tanpa gagal).
  - Analisis ablasi sementara dibuat dan RQ1 dihitung ulang.
  - Pakem diperbarui ke v1.3.2.

## 4. Hasil kunci

**Protokol ukur.** Metrik utama Fase 5 dihitung ulang dari prediksi per citra:
- inferensi 640 px,
- NMS dibuat seperti `val()`: multi_label=True, tanpa batas waktu, torchvision,
- AP 101 titik pada 10 ambang IoU.

Angka `val()` Ultralytics 0,3–1,3 poin lebih tinggi, karena `val()` memproses citra 640 px dalam bingkai 672 px (rect, pad 0,5) dan memakai presisi setengah saat training. Angka `val()` dilaporkan sebagai pendamping.

**Struktur data (seluruh pool 11.547 citra):**
- Semua 1.407 citra CBB dan 1.274 dari 1.562 citra CBSD berasal dari valid/test Roboflow, tanpa salinan augmentasi.
- Semua citra CGM, CMD, dan HEALTHY berasal dari train Roboflow, dengan 2 salinan (rotasi ±10°).
- Split bawaan Roboflow hampir tidak punya CBB di data latih.
- Baji hitam isian rotasi ada pada 41,2% citra asal train vs 2,5% citra asal valid/test. Ini risiko jalan pintas kelas yang sudah dicatat di bagian Risiko pakem.

**EXP-0 (mAP@50–95):**

| Seed | `val()` | Hitung ulang |
|---|---|---|
| 42 | 26,20 | 25,67 |
| 123 | 26,55 | 26,22 |
| 2026 | 27,00 | 26,06 |

**RQ1 (final, prediksi yang sudah diperbaiki, bootstrap 1.000 iterasi):**

| Uji (mAP@50–95) | Selisih | IK 95% |
|---|---|---|
| Kemampuan pada 227 citra yang sama | +0,14 | −2,40 s.d. +2,59 |
| Angka yang dilaporkan | +6,39 | +4,14 s.d. +8,56 |
| Inflasi di luar kemampuan | +6,25 | +2,86 s.d. +9,59 |
| Efek kembaran, kontrol 108 citra | +11,05 | +5,41 s.d. +18,15 |
| Efek kembaran, kontrol 227 citra | +9,06 | +5,36 s.d. +12,53 |

Bagian B: split bawaan Roboflow tidak bocor. Audit nama file menemukan 661 grup lintas split, tetapi semuanya positif palsu.

**Ablasi sementara (seed 42 dan 123, hitung ulang, mAP@50–95):**

| Model | Rata-rata |
|---|---|
| EXP-0 | 25,98 ± 0,29 (3 seed) |
| EXP-0R | 25,53 |
| EXP-1 | 24,33 |
| EXP-2 | 25,94 |
| EXP-3 | 25,28 |

| Perbandingan | Selisih | IK 95% | Verdict sementara |
|---|---|---|---|
| EXP-1 − EXP-0R (RQ2) | −1,19 | −1,88 s.d. −0,50 | menurunkan konsisten |
| EXP-0R − EXP-0 | −0,42 | −0,99 s.d. +0,15 | tidak berbeda |
| EXP-2 − EXP-0 (RQ3) | −0,01 | −0,54 s.d. +0,55 | tidak berbeda |
| EXP-3 − EXP-0 | −0,66 | −1,31 s.d. +0,04 | tidak berbeda |
| EXP-3 − EXP-1 | +0,95 | +0,30 s.d. +1,65 | meningkat; antar seed bervariasi |

- **Kerugian EXP-1 terpusat pada CBB (−2,5 poin AP) dan CBSD (−2,0).** Porsi sampling per kelas hampir tidak berubah (rasio 0,93–1,06), jadi yang ditekankan adalah citra tersulit di dalam kelas.
- **Kurva validasi masih naik di epoch 30**, sehingga uji CONV perlu.

## 5. File penting di `D:\UNDIP\SKRIPSI`

- **Runbook:** `MALAM_INI.md`.
- **Runner:**
  - `jalankan_fase3.ps1`, dipanggil oleh `cek_siap_fase3.bat`, `jalankan_malam_A.bat` (`-UlangELeak`), dan `jalankan_malam_B.bat`.
  - `rapikan_git.bat`.
- **Training:**
  - `stage_8_2_build_difficulty_manifest.py`
  - `stage_8_3_adaptive_pipeline.py`
  - `stage_8_4_run_ablation_experiments.py` (`--exp`, `--seed`, `--skip-if-done`, `--dry-run`)
  - `research_utils/detection_metrics.py`
- **Analisis:**
  - `predict_eleak_per_image.py` (NMS seperti `val()`)
  - `analyze_eleak_per_image.py`
  - `analyze_eleak_did.py` (`--control A|TW`, variabel lingkungan `ELEAK_PER_IMAGE_DIR`)
  - `analyze_ablation.py` (`prep`, `boot`, `report`)
  - `cek_struktur_kelas.py`
  - `build_label_audit.py`
- **Hasil:**
  - Manifest kesulitan: `stage_v2_difficulty_20261028/`
  - Run ablasi: `stage_v2_ablation/seed_*/`. EXP-0 seed 42 ada di `stage_8_ablation_clean_20261020/exp_0_baseline`.
  - Prediksi per citra: `stage_v2_ablation/per_image/`
  - Laporan ablasi: `stage_v2_ablation/analysis/ABLATION_REPORT.md`
  - Laporan RQ1: `stage_v2_eleak_sim/per_image_v2/analysis/`
  - Data dan laporan pendukung: `eleak_sim_rq1/`, `_analisis_rq1_audit/`, `_analisis_struktur_kelas/`
  - Log: `logs_v2/`
- **Audit label:** `audit_label_150/`. Isinya:
  - `LEMBAR_AUDIT_LABEL.xlsx` (150 citra: 100 dengan D tertinggi + 50 acak, satu per grup, buta strata, 9 kategori),
  - `LEMBAR_AUDIT_RATER2.xlsx` (30 citra),
  - folder `_kunci/`, yang jangan dibuka sebelum audit selesai.

## 6. Posisi sekarang dan langkah berikutnya

- **Status fase:**

| Fase | Status |
|---|---|
| 0 (persetujuan pembimbing) | terbuka |
| 1 dan 2 | selesai |
| 3 | 11 dari 15 run selesai |

- **Berikutnya:** malam B (`jalankan_malam_B.bat`), 4 run seed 2026, sekitar 3 jam.
- **Keputusan yang belum dijawab pengguna:** CONV untuk Fase 4.
  - Usulan Claude: EXP-0 dan EXP-1 dengan 60 epoch, seed 42, sekitar 3 jam.
  - Alasannya, tidak ada strategi yang mengalahkan baseline.
  - Bisa ditambahkan ke malam B, sehingga totalnya sekitar 6 jam.
- **Setelah malam B:**
  1. Stage 15 CSV prediksi, `gt_cleanval.csv`, `cassava-v7-stage-6/resplit_group_manifest.csv`, dan skrip analisis ke workspace cloud.
  2. Jalankan `analyze_ablation.py prep`, lalu `boot` 4 × 250 iterasi, lalu `report`.
  3. Tulis hasilnya ke `stage_v2_ablation/analysis/`.
  4. Terapkan aturan keputusan dan gate G1 (judul kerja kemungkinan dipakai), lalu perbarui pakem.
- **Tawaran yang belum dijawab:**
  - uji jalan pintas baji hitam (beberapa menit GPU),
  - erratum laporan literatur soal klaim kebocoran 51,71%.
- **Sesudahnya:**
  - Fase 4: CONV, dan EXP-1c bila perlu.
  - Fase 5 dan RQ4: audit label 150 citra oleh pengguna, AP dan recall per ukuran objek, pergeseran jenis error, log crop.
  - Fase 6: test set sekali.
  - Fase 7: penulisan. Bab 1–3 dan Bab 4.1 (RQ1) sudah bisa ditulis.

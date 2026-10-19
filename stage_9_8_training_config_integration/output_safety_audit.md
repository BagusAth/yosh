# Output Safety and Zero Dataset Mutation Audit — Stage 9.8

## 1. Executive Summary

Audit ini memverifikasi perlindungan terhadap artefak historis penelitian dan membuktikan prinsip *zero dataset mutation* (nol mutasi dataset) pada repositori penelitian.

Sesuai batasan Stage 9.8, seluruh operasi kode, validasi preflight, eksekusi dry-run, dan pengujian unit dilarang keras memodifikasi, menimpa, atau mencemari:
1. Direktori artefak eksperimen historis (`stage_8_1_oof`, `stage_8_2_difficulty`, `stage_8_ablation`, dsb.).
2. Dataset kanonis penelitian (`cassava-v7-standardized` dan `cassava-v7-research`).

---

## 2. Perlindungan Direktori Historis (Historical Protection)

### 2.1 Daftar Direktori Terlarang (`FORBIDDEN_HISTORICAL_DIRS`)

Fungsi [`validate_output_directory_safety()`](file:///d:/UNDIP/SKRIPSI/research_utils/detection_metrics.py#L670-L720) secara ketat memblokir setiap upaya penulisan ke folder historis berikut (termasuk seluruh subdirektori anak):

| Direktori Terlarang | Tahapan Terkait | Alasan Perlindungan |
|---|---|---|
| `stage_7_baseline/` | Baseline YOLO11n historis | Mempertahankan bukti baseline orisinal |
| `cassava-v7-stage-7/` | Artefak Stage 7 | Mempertahankan model & metrik historis |
| `stage_8_1_oof/` | OOF profiling historis | Menjaga integritas prediksi OOF lama |
| `stage_8_2_difficulty/` | Manifest kesulitan historis | Menjaga integritas manifest lama |
| `stage_8_ablation/` | Ablasi EXP-0..EXP-3 historis | Mempertahankan riwayat ablasi awal |
| `stage_8_5_model_selection/` | Validasi model historis | Mempertahankan seleksi checkpoint |
| `stage_8_6_final_test/` | Benchmark test locked | Mempertahankan hasil locked test |
| `runs/` | Folder default Ultralytics | Mencegah penulisan tidak terstruktur |

### 2.2 Mekanisme Penolakan Fail-Closed

1. **PermissionError pada Direktori Historis**:
   Setiap path yang mengarah ke salah satu direktori terlarang di atas langsung memicu `PermissionError` seketika sebelum berkas apa pun ditulis.
2. **FileExistsError pada Clean Run**:
   Jika direktori output sudah ada dan berisi artefak eksperimen (misalnya `metrics_summary.json` atau folder `weights/`), fungsi menolak eksekusi dengan `FileExistsError` untuk mencegah penimpaan tidak disengaja, kecuali flag reuse eksplisit (`--skip-if-done`) diaktifkan dan metadata terverifikasi cocok 100%.

---

## 3. Bukti Nol Mutasi Dataset (Zero Mutation Verification)

### 3.1 Eliminasi Celah FINDING-9.8-05

Pada audit Stage 9.7, ditemukan bahwa pemeriksaan Git standar (`git status`) dapat melewatkan berkas `.cache` karena pola `*.cache` diabaikan dalam `.gitignore`. Akibatnya, preflight yang ceroboh dapat menghasilkan file cache pada dataset resmi tanpa terdeteksi oleh Git.

Pada Stage 9.8, dilakukan remediasi:
1. **Preflight Validator Murni Read-Only**:
   Fungsi `validate_yolo_dataset_integrity()` membaca berkas citra dan label secara murni menggunakan operasi *read-only* (membaca baris teks label tanpa memanggil DataLoader YOLO yang memicu pembuatan cache).
2. **Isolasi DataLoader pada Unit Test**:
   Setiap unit test yang menguji DataLoader (`AdaptiveYOLODataset`) menggunakan direktori sementara (`tempfile.mkdtemp`), sehingga pembuatan cache label hanya terjadi pada folder temporer yang dihapus setelah pengujian selesai.
3. **Pemeriksaan Direktori Citra Standar**:
   Pemeriksaan langsung pada direktori citra `cassava-v7-standardized/train/images` memastikan bahwa **nol file `.cache`** tercipta di direktori citra.

### 3.2 Bukti Integritas Git Status

Pemeriksaan status Git pada kedua direktori dataset riset membuktikan nol perubahan:

```bash
$ git status cassava-v7-standardized cassava-v7-research
On branch stage-9-8-training-config-integration
nothing to commit, working tree clean
```

### 3.3 Bukti Fingerprint SHA-256 Sebelum dan Sesudah

Fingerprint inventaris dataset dihitung sebelum dan sesudah seluruh siklus preflight dry-run:

| Dataset Split | File Count (Images + Labels) | Inventory Fingerprint SHA-256 | Status |
|---|---|---|---|
| `standardized/train` | 8,980 + 8,980 = 17,960 | `085a6a61f5164ca6e00b8529f79caefbc809b4db070ad7d727b3d33cb42c7333` | **IDENTIK / UNMUTATED** |
| `standardized/val` | 2,567 + 2,567 = 5,134 | `d59cb65b0953a73c155490a1f26f29a0f443b74704a2cbaf863260c878b6ee38` | **IDENTIK / UNMUTATED** |

---

## 4. Pembuktian Uji Regresi (Group F)

| Test Method | Target Pengujian | Hasil |
|---|---|---|
| `test_all_forbidden_historical_directories_rejected` | Menguji penolakan seluruh `FORBIDDEN_HISTORICAL_DIRS` dan subfolder anaknya | **PASS** (100% diblokir dengan `PermissionError`) |
| `test_clean_output_directory_permitted` | Menguji perizinan penulisan pada folder baru yang bersih | **PASS** (Lolos validasi keamanan) |
| `test_zero_mutation_dataset_verification` | Memverifikasi tidak ada file `.cache` pada direktori citra riset | **PASS** (0 cache files) |
| `test_verify_dataset_files_and_checksums_unmutated_before_after` | Memverifikasi fingerprint SHA-256 tidak berubah setelah validasi integritas | **PASS** (Identik sebelum & sesudah) |

---

## 5. Kesimpulan

Pipeline penelitian telah terbukti memiliki lapisan pengaman yang kokoh:
1. Data historis terlindungi secara mutlak dari risiko penimpaan.
2. Dataset kanonis terbukti steril dan bebas dari mutasi maupun pencemaran file cache.

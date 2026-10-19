# AUDIT INTEGRITAS DATASET, CITRA LATAR BELAKANG & RESOLUSI SPLIT KANONIS

**Tahap Riset**: STAGE 9.7 — Pipeline Safety, Provenance & End-to-End Readiness Remediation  
**Status Audit**: PASS (100% Content-Aware & Standardized)  
**Komponen Utama**:
- `research_utils/detection_metrics.py::compute_dataset_inventory_fingerprint()`
- `research_utils/detection_metrics.py::validate_yolo_dataset_integrity()`
- `research_utils/detection_metrics.py::resolve_data_yaml_split()`
- `research_utils/detection_metrics.py::derive_yolo_label_path()`

---

## 1. Hashing Inventaris Konten Citra (Content-Aware SHA-256)

### 1.1. Kelemahan Audit Terdahulu
Sebelumnya, pemeriksaan inventaris dataset berisiko hanya membandingkan ukuran byte file (*file size*) atau daftar nama file. Hal ini berpotensi gagal mendeteksi korupsi data atau manipulasi di mana byte citra berubah tetapi ukuran file tetap identik (misal pada citra yang ditimpa byte acak dengan panjang yang sama).

### 1.2. Implementasi Stage 9.7
Fungsi `compute_dataset_inventory_fingerprint()` kini mengiterasi seluruh citra dan label secara deterministik terurut:
1. Setiap file citra dibaca secara biner (`f.read_bytes()`) dan di-hash menggunakan SHA-256.
2. Setiap file label dibaca dan di-hash menggunakan SHA-256, serta dihitung distribusi jumlah objek per kelas.
3. Jika `data_yaml_path` diberikan, konfigurasi YAML dan seluruh isi citra/label pada split validasi (`val` / `valid`) juga di-hash ke dalam fingerprint.
4. Terbukti secara empiris: memodifikasi byte citra dengan ukuran file persis sama (misal 100 byte 'A' diganti 100 byte 'Z') menghasilkan fingerprint yang sepenuhnya berbeda (`test_image_byte_modification_same_size_detected` -> PASS).

---

## 2. Kontrak Citra Latar Belakang (Background Images)

Dalam tugas deteksi objek YOLO:
- Citra latar belakang (*background image*) adalah citra daun yang tidak memuat lesi penyakit atau objek yang dianotasi sama sekali.
- Sesuai standar resmi Ultralytics YOLO, file label untuk background image adalah file teks kosong (ukuran 0 byte).
- Pada audit terdahulu, terdapat risiko file 0 byte dianggap sebagai file korup atau label yang hilang.
- Pada Stage 9.7, fungsi `validate_yolo_dataset_integrity()` secara eksplisit memverifikasi dan mengakui file label 0 byte sebagai citra background yang sah:

```python
if len(non_empty_lines) == 0:
    # Valid background image in YOLO convention (no objects)
    background_images += 1
    continue
```

- Pada dataset kanonis `cassava-v7-standardized/train`, terdapat tepat **66 citra background** (0 box). Seluruh 66 citra ini diakui secara sah oleh audit label Stage 8.1 dan validasi integritas YOLO.

---

## 3. Resolusi Single Source of Truth (SSOT) data.yaml

Fungsi `resolve_data_yaml_split()` dirancang untuk menjadi pintu gerbang tunggal dalam membaca data.yaml dan mendukung seluruh variasi layout dataset Ultralytics:

1. **Path Relatif & Absolut**:
   Mendukung entri `path:` berupa string absolut maupun relatif terhadap direktori `data.yaml`.
2. **Layout Direktori Standar**:
   Mendukung penunjukan langsung ke folder `train/images` atau folder `train` yang memuat subfolder `images/` dan `labels/`.
3. **Layout Berbasis File List (`.txt`)**:
   Mendukung split yang didefinisikan sebagai file `.txt` berisi daftar path citra per baris.
4. **Layout Berbasis List Direktori (`[dir1, dir2]`)**:
   Mendukung entri YAML berupa list direktori (diperbaiki pada Stage 9.7 untuk mencegah `TypeError`).
5. **Derivasi Path Label Kanonis (`derive_yolo_label_path`)**:
   Menjamin substitusi path direktori `images` -> `labels` dan ekstensi citra (`.jpg`, `.png`, `.jpeg`) -> `.txt` secara konsisten di seluruh kode riset.

---

## 4. Matriks Pembuktian Pengujian (Group D & Group E)

| ID Tes | Skenario Pengujian Integritas | Respon Sistem | Status |
| :--- | :--- | :--- | :---: |
| `test_image_byte_modification_same_size_detected` | Byte citra diubah dengan ukuran file sama | Fingerprint berubah seketika | PASS |
| `test_label_byte_modification_detected` | Byte label diubah | Fingerprint berubah seketika | PASS |
| `test_validation_split_content_change_detected` | Byte citra split validasi diubah | Fingerprint berubah seketika | PASS |
| `test_validate_yolo_dataset_integrity_missing_label_rejected` | Citra tanpa file label pasangan | Ditolak: `Missing label file for image` | PASS |
| `test_validate_yolo_dataset_integrity_orphan_label_rejected` | File label tanpa pasangan citra | Ditolak: `Orphan label file found` | PASS |
| `test_validate_yolo_dataset_integrity_empty_label_treated_as_background` | Label 0 byte | Lolos validasi, dihitung sebagai background | PASS |
| `test_validate_yolo_dataset_integrity_invalid_coordinates_rejected` | Koordinat bbox x > 1.0 | Ditolak: `outside [0.0, 1.0] range` | PASS |
| `test_resolver_relative_and_absolute_paths` | Resolusi path relatif vs absolut | Menghasilkan path absolut identik | PASS |
| `test_resolver_txt_file_subset` | Split berupa daftar file list .txt | Mengembalikan subset citra terdaftar | PASS |
| `test_resolver_multi_directory_list` | Split berupa list direktori | Mengagregasi seluruh citra dari semua folder | PASS |
| `test_resolver_empty_split_raises` | Folder split tanpa citra | Ditolak: `No valid images found` | PASS |
| `test_resolver_missing_image_file_in_txt_raises` | Path citra fiktif di dalam .txt | Ditolak: `FileNotFoundError` | PASS |

---

## 5. Kesimpulan

Integritas dataset `cassava-v7-standardized` terverifikasi 100% bebas dari missing label, orphan label, dan anomali koordinat. Resolusi data.yaml kini bersifat universal dan tahan terhadap variasi struktur input.

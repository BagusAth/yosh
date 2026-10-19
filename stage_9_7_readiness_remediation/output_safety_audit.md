# AUDIT KEAMANAN DIREKTORI OUTPUT & PENCEGAHAN PENIMPAAN ARTEFAK HISTORIS

**Tahap Riset**: STAGE 9.7 — Pipeline Safety, Provenance & End-to-End Readiness Remediation  
**Status Audit**: PASS (100% Mengeras / Hardened)  
**Komponen Utama**: `research_utils/detection_metrics.py::validate_output_directory_safety()`  
**Pemanggil**: `stage_7`, `stage_8_1`, `stage_8_2`, `stage_8_4`, `stage_8_5`, `stage_8_6`, `run_stage_8_profiling`

---

## 1. Latar Belakang & Celah Keamanan yang Ditemukan (P0)

Sebelum Stage 9.7, fungsi `validate_output_directory_safety()` memiliki beberapa celah kritis yang dapat membahayakan integritas artefak penelitian:

1. **Pemeriksaan Hanya Berdasarkan Nama Leaf Folder**:
   Fungsi sebelumnya mengecek apakah `Path(output_dir).name` ada dalam daftar terlarang. Akibatnya, path subdirektori bersarang seperti `stage_8_1_oof/fold_1/weights` atau `stage_8_ablation/exp0_baseline` lolos dari validasi dan dapat menimpa sub-artefak historis.
2. **Path Traversal Loopholes (`..`)**:
   Penggunaan traversal path relatif seperti `stage_8_ablation_clean/../stage_8_ablation` tidak dinormalisasi secara tuntas sebelum pencocokan nama.
3. **Konflik Tingkat Root vs Anak pada Orkestrator Stage 8.4**:
   Orkestrator Stage 8.4 mengeksekusi empat eksperimen berurutan (`EXP-0` s.d. `EXP-3`) ke dalam subdirektori di bawah root `stage_8_ablation_clean`. Pada kode sebelum Stage 9.7, setiap iterasi eksperimen memeriksa root `output_dir`. Setelah `EXP-0` selesai menulis artefak, ketika `EXP-1` hendak berjalan, ia memeriksa `output_dir` yang sudah tidak kosong, lalu memunculkan false error `Output directory is not empty`.

---

## 2. Remediasi yang Diterapkan

### 2.1. Penutupan Celah Nested & Ancestor Path

Dalam `research_utils/detection_metrics.py`, `validate_output_directory_safety()` diperbaiki secara komprehensif:

```python
FORBIDDEN_HISTORICAL_DIRS = {
    "stage_7_baseline",
    "stage_8_1_oof",
    "stage_8_2_difficulty",
    "stage_8_ablation",
    "cassava-v7-stage-7",
    "runs/stage_8_1_oof",
    "runs/stage_8_4_ablation",
}
```

Pemeriksaan dilakukan melalui 3 lapis perlindungan:
1. **Normalisasi Path Absolut**: Menggunakan `Path(output_dir).resolve()` untuk mengeliminasi simlink, separator OS (Windows `\` vs Unix `/`), dan traversal `..`.
2. **Inspeksi Seluruh Komponen Path (`p.parts`)**: Memeriksa setiap elemen direktori (ancestor) dalam path yang telah dinormalisasi maupun path mentah (`p_raw.parts`). Jika salah satu komponen cocok dengan `FORBIDDEN_HISTORICAL_DIRS`, eksekusi dihentikan seketika dengan `PermissionError`.
3. **Pemeriksaan Relatif terhadap CWD (`p.is_relative_to(cwd_forb)`)**: Menjamin tidak ada path absolut atau relatif yang merupakan anak dari direktori historis.

> [!NOTE]
> Nama direktori "clean sibling" seperti `stage_8_ablation_clean` atau `stage_8_1_oof_clean` tetap diizinkan secara sah karena komponen namanya (`stage_8_ablation_clean`) tidak sama dengan entri terlarang (`stage_8_ablation`).

### 2.2. Pemisahan Validasi Root vs Sub-Eksperimen (Parameter `is_root`)

Pada Stage 8.4, orkestrator dan sub-eksperimen kini memiliki kontrak yang jelas:

- **Orkestrator Root (`is_root=True`)**:
  Dijalankan tepat satu kali di awal `run_ablation_pipeline()`.
  Memvalidasi bahwa root folder (misal `stage_8_ablation_clean`) aman dan tidak mengarah ke historis.
  Jika `--skip-if-done` aktif, root check mengizinkan keberadaan sub-folder eksperimen lama.
- **Sub-Eksperimen Child (`is_root=False`)**:
  Dijalankan pada setiap subdirektori eksperimen (`exp_dir = output_dir / spec["name"]`).
  Memvalidasi direktori eksperimen tersebut secara terisolasi.
  Dengan pemisahan ini, `EXP-1` dapat berjalan mulus setelah `EXP-0` selesai tanpa konflik false-positive.

### 2.3. Kontrak Ketat `--skip-if-done`

Pemeriksaan reuse direktori kini menuntut `expected_identity` lengkap:
- Jika direktori output sudah ada isinya dan `--skip-if-done` / `allow_reuse=False`, direktori langsung ditolak dengan `FileExistsError`.
- Jika `allow_reuse=True`, fungsi wajib menerima `expected_identity` dan membaca metadata (`metrics_summary.json` atau `fold_metadata.json`).
- Jika terdapat perbedaan pada parameter eksperimen, skrip, data YAML, bobot model, atau manifest, reuse ditolak dengan `PermissionError` dan meminta direktori baru yang bersih.

---

## 3. Matriks Pembuktian Pengujian (Group A)

Seluruh skenario pengujian diimplementasikan pada `test_stage_9_7_regression.py::TestGroupAOutputDirectorySafety`:

| ID Tes | Skenario Uji | Ekspektasi | Hasil Aktual | Status |
| :--- | :--- | :--- | :--- | :---: |
| `test_clean_directory_passes` | Direktori kosong baru | Lolos (`is_safe=True`, `"clean target"`) | PASS | OK |
| `test_historical_target_rejected` | Target tepat pada `stage_8_1_oof` | `PermissionError` seketika | PASS | OK |
| `test_nested_subdirectories_rejected` | Target pada `stage_8_ablation/exp0/sub` | `PermissionError` seketika | PASS | OK |
| `test_path_traversal_dotdot_rejected` | Target traversal `clean/../stage_8_2_difficulty` | `PermissionError` seketika | PASS | OK |
| `test_case_insensitivity_rejected` | Target `STAGE_8_1_OOF` | `PermissionError` seketika | PASS | OK |
| `test_clean_sibling_permitted` | Target `stage_8_ablation_clean` | Lolos validasi | PASS | OK |
| `test_partial_output_rejected` | Non-empty dir tanpa `--skip-if-done` | `FileExistsError` | PASS | OK |
| `test_valid_reuse_passes` | Non-empty dir dengan metadata cocok | Lolos reuse (`action="REUSE"`) | PASS | OK |
| `test_identity_mismatch_rejected` | Non-empty dir dengan metadata tidak cocok | `PermissionError` | PASS | OK |

---

## 4. Kesimpulan

Mekanisme proteksi direktori output telah mengeras 100%. Tidak ada jalur eksekusi yang dapat menembus atau menimpa artefak historis, baik melalui subdirektori bersarang, traversal path relatif, maupun variasi kapitalisasi nama direktori.

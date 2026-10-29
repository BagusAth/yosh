# Audit label manual (150 citra)

1. Buka `LEMBAR_AUDIT_LABEL.xlsx`, baca sheet **Panduan**, lalu isi sheet **Audit** (sel kuning).
2. Klik "Buka" di kolom Gambar untuk melihat `gambar/Annn.jpg` (kiri citra asli, kanan kotak label bernomor).
3. Rater kedua (opsional, 30 citra) mengisi `LEMBAR_AUDIT_RATER2.xlsx` tanpa melihat lembar rater 1.
4. Jangan buka folder `_kunci/` sebelum semua citra dinilai.

Sampel: 100 citra latih dengan D_image tertinggi + 50 citra acak, maksimal satu citra per grup foto, urutan diacak (seed tetap). Dibuat ulang dengan `python build_label_audit.py`; lembar yang sudah ada tidak ditimpa.

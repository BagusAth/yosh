# Audit label manual (150 citra) - lembar v2

Versi lembar v2 (1 November 2026): audit tanpa keahlian penyakit tanaman. Benar-salahnya kelas
penyakit TIDAK dinilai; yang dinilai hanya masalah label yang terlihat tanpa diagnosis.

1. Baca `PANDUAN_BERGAMBAR.pdf` (9 halaman, contoh dari citra validasi, bukan sampel audit).
2. Buka `LEMBAR_AUDIT_LABEL.xlsx`, baca sheet **Panduan**, lalu isi sheet **Audit** (sel kuning).
3. Klik "Buka" di kolom Gambar untuk melihat `gambar/Annn.jpg` (kiri citra asli, kanan kotak label bernomor).
4. Rater kedua (opsional, 30 citra) mengisi `LEMBAR_AUDIT_RATER2.xlsx` tanpa melihat lembar rater 1.
5. Jangan buka folder `_kunci/` dan jangan membandingkan dengan prediksi model sebelum semua citra dinilai.

Sampel: 100 citra latih dengan D_image tertinggi + 50 citra acak, maksimal satu citra per grup foto,
urutan diacak (seed tetap). Sampel, kode, urutan, dan gambar tidak berubah dari v1.

Riwayat:
- v1 (29 Okt 2026): `build_label_audit.py`. Lembar v1 (kosong) disimpan di `_arsip_v1/`.
- v2 (1 Nov 2026): `revisi_lembar_audit.py` (kategori tanpa diagnosis, kolom kelas diganti
  "Keyakinan penilaian") dan `buat_panduan_bergambar.py`. Dibuat sebelum ada penilaian dan
  sebelum kunci dibuka.

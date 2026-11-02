"""Panduan bergambar untuk audit label v2 (tanpa keahlian penyakit tanaman).

Semua contoh diambil dari split VALIDASI (cassava-v7-runtime-mirror/valid), sehingga
tidak ada yang tumpang tindih dengan 150 citra audit (semuanya citra latih).
Label dibaca dari runtime-mirror (format kotak YOLO yang sama dengan yang dipakai
pelatihan dan gambar audit). Skrip ini tidak membaca folder _kunci/.

Keluaran: audit_label_150/PANDUAN_BERGAMBAR.pdf (+ halaman JPG di _panduan_kerja/halaman/).
Pakai: python buat_panduan_bergambar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_label_audit import read_labels, render  # noqa: E402

V = Path("cassava-v7-runtime-mirror/valid")
KERJA = Path("_panduan_kerja/halaman")
OUT = Path("audit_label_150/PANDUAN_BERGAMBAR.pdf")
W, H = 1754, 1240  # A4 lanskap 150 dpi
MERAH = (220, 30, 30)


def font(size: int, bold: bool = False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tulis(d, xy, teks, f, lebar, fill=(20, 20, 20), spasi=8) -> int:
    """Tulis teks dengan bungkus kata; kembalikan y setelah teks."""
    x, y = xy
    for par in teks.split("\n"):
        baris, kata = "", par.split(" ")
        for k in kata:
            uji = (baris + " " + k).strip()
            if d.textlength(uji, font=f) <= lebar:
                baris = uji
            else:
                d.text((x, y), baris, font=f, fill=fill)
                y += f.size + spasi
                baris = k
        d.text((x, y), baris, font=f, fill=fill)
        y += f.size + spasi
    return y


CONTOH = [
    dict(f="629341978_jpg.rf.fa1ad49537842f50504e22ed16157a8a.jpg",
         judul="Daun tanpa kotak (citra berlabel HEALTHY)",
         jelas="Pada citra berlabel HEALTHY, setiap daun utuh yang jelas terlihat seharusnya berada di dalam kotak. "
               "Di sini hanya sekelompok daun di kiri bawah yang berkotak (kotak 1). Daun-daun lain di bagian atas "
               "dan kanan (dilingkari merah) tidak berada di dalam kotak mana pun.",
         isi=["Daun tanpa kotak", "", ">5", "Yakin", "Ya"],
         lingkar=[(170, 255, 110), (380, 185, 95), (548, 290, 90)]),
    dict(f="3179751044_jpg.rf.ed5f7a4672e478f5b1fde7ced6d6a8ef.jpg",
         judul="Daun tanpa kotak (citra berlabel penyakit)",
         jelas="Kamu tidak perlu tahu nama penyakitnya. Lihat daun di dalam kotak 1: daunnya berbintik pucat. "
               "Lalu cari daun lain dengan pola serupa yang tidak berkotak (dilingkari merah). Daun yang polos hijau "
               "tidak dihitung.",
         isi=["Daun tanpa kotak", "", "3-5", "Yakin", "Ya"],
         lingkar=[(330, 105, 80), (330, 335, 75), (95, 345, 65)]),
    dict(f="1717047149_jpg.rf.4a40ddd37f9c713ae298e515fb1b6378.jpg",
         judul="Kotak bersarang/ganda",
         jelas="Kotak 1 berada di dalam kotak 3, dan kotak 2 menumpuk dengan kotak 3. Daun yang sama tercakup "
               "lebih dari satu kotak. Tidak perlu menilai penyakitnya; cukup lihat posisi kotak.",
         isi=["Kotak bersarang/ganda", "", "1-2", "Yakin", "Sebagian"],
         lingkar=[]),
    dict(f="3953857113_jpg.rf.dd9fdebccff78ad07b9a2017cd3bf799.jpg",
         judul="Objek bukan daun",
         jelas="Kotak 1 menandai batang (tegak di tengah atas), bukan daun. Kategori ini juga dipakai bila kotak "
               "menandai tanah, umbi, tangan, langit, atau benda lain.",
         isi=["Objek bukan daun", "", "Ragu", "Yakin", "Ya"],
         catatan="Kotak 1 menandai batang.",
         lingkar=[]),
    dict(f="822532439_jpg.rf.5c553ee9045012672ef031cc5f53ab8c.jpg",
         judul="Kotak tidak presisi",
         jelas="Daun kuning berbercak di tengah adalah daun yang sejenis dengan isi kotak, tetapi kotak 1 hanya "
               "menutupi sudut kiri bawahnya. Kotak terlalu sempit dan bergeser. Kategori yang sama dipakai bila "
               "kotak terlalu longgar (banyak latar ikut masuk).",
         isi=["Kotak tidak presisi", "", "0", "Yakin", "Tidak"],
         lingkar=[]),
    dict(f="image_287_jpg.rf.d4df578b0fc6950e237d3c36ce59287e.jpg",
         judul="Artefak citra",
         jelas="Bagian atas dan kanan citra berisi garis-garis berulang hasil rotasi (bukan tanaman). Pilih kategori "
               "ini bila area rusak kira-kira seperempat citra atau lebih, atau citra sangat buram/gelap. Sudut hitam "
               "kecil akibat rotasi ada di banyak citra; abaikan bila kecil.",
         isi=["Artefak citra", "", "0", "Yakin", "Ya"],
         lingkar=[]),
    dict(f="385391871_jpg.rf.d701811eabcde61a143beed74ad54b47.jpg",
         judul="Beberapa masalah sekaligus",
         jelas="Ada tiga hal: banyak kotak berada di dalam kotak lain, beberapa daun belang di kanan atas tidak "
               "berkotak (dilingkari merah), dan citranya padat. Dahulukan masalah label daripada kondisi citra. Di "
               "antara dua masalah label, pilih yang menyangkut paling banyak daun sebagai Kategori utama dan yang lain "
               "sebagai Kategori tambahan. Kondisi padat cukup ditulis di Catatan. Kasus seperti ini wajar dinilai "
               "\"Kurang yakin\".",
         isi=["Kotak bersarang/ganda", "Daun tanpa kotak", "3-5", "Kurang yakin", "Sebagian"],
         catatan="Citra juga padat.",
         lingkar=[(530, 105, 95)]),
    dict(f="image_413_jpg.rf.cbfc512156a55661f43a86a7f0944163.jpg",
         judul="Tidak ada masalah terlihat",
         jelas="Daun kuning belang di tengah sudah berkotak, kotaknya cukup pas, dan tidak ada daun lain dengan pola "
               "serupa yang tidak berkotak. Garis-garis di tepi kanan dan bawah hanya sedikit, jadi diabaikan (bukan "
               "artefak citra). Penilaian tetap boleh \"Kurang yakin\" bila ada bagian yang meragukan.",
         isi=["Tidak ada masalah terlihat", "", "0", "Yakin", "Ya"],
         lingkar=[]),
]
KOLOM = ["Kategori utama", "Kategori tambahan", "Daun tanpa kotak", "Keyakinan penilaian", "Kotak presisi?"]


def halaman_sampul() -> Image.Image:
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    y = tulis(d, (90, 80), "Panduan bergambar audit label (v2)", font(46, True), W - 180)
    y = tulis(d, (90, y + 10), "Untuk penilai tanpa latar belakang penyakit tanaman", font(28), W - 180,
              fill=(90, 90, 90))
    y += 30
    teks = [
        ("Cara membaca gambar audit",
         "Setiap gambar punya dua panel: kiri citra asli, kanan citra yang sama dengan kotak label bernomor. "
         "Warna kotak menunjukkan kelas, tetapi kamu tidak perlu menilai benar-salahnya kelas."),
        ("Yang dinilai",
         "Hanya hal yang terlihat tanpa diagnosis. Empat masalah label: daun tanpa kotak, kotak bersarang/ganda, "
         "objek bukan daun, kotak tidak presisi. Dua kondisi citra: artefak citra, citra padat/sulit. "
         "Atau: tidak ada masalah terlihat."),
        ("Patokan daun tanpa kotak",
         "Satu kotak boleh menutupi satu daun atau sekelompok daun. Yang dihitung adalah daun yang tidak berada di "
         "dalam kotak mana pun. Citra HEALTHY: semua daun utuh. Citra berpenyakit: hanya daun yang polanya mirip "
         "daun yang sudah berkotak di citra yang sama."),
        ("Bila ada beberapa masalah",
         "Dahulukan masalah label daripada kondisi citra. Di antara masalah label, pilih yang menyangkut paling "
         "banyak daun; sisanya ke Kategori tambahan."),
        ("Bila ragu",
         "Pilih \"Kurang yakin\" di kolom Keyakinan penilaian atau \"Ragu\" di kolom lain. Itu jawaban yang sah, "
         "bukan kesalahan."),
        ("Asal contoh",
         "Semua contoh di panduan ini diambil dari citra validasi, bukan dari 150 citra audit. Jawaban pada contoh "
         "adalah ilustrasi cara mengisi, bukan kunci."),
    ]
    for kepala, isi in teks:
        y = tulis(d, (90, y), kepala, font(27, True), W - 180)
        y = tulis(d, (90, y), isi, font(25), W - 180) + 18
    return im


def halaman(no: int, c: dict) -> Image.Image:
    tmp = KERJA / f"_render_{no}.jpg"
    lab = read_labels(V / "labels" / (Path(c["f"]).stem + ".txt"))
    render(V / "images" / c["f"], lab, f"Contoh {no}", tmp)
    pan = Image.open(tmp).convert("RGB")
    if c["lingkar"]:  # (cx, cy, r) dalam koordinat citra asli 640 px, digambar di kedua panel
        dd = ImageDraw.Draw(pan)
        w0 = (pan.width - 10) // 2
        for cx, cy, r in c["lingkar"]:
            for ox in (0, w0 + 10):
                for t in range(4):
                    dd.ellipse([ox + cx - r - t, 34 + cy - r - t, ox + cx + r + t, 34 + cy + r + t], outline=MERAH)
    skala = (W - 120) / pan.width
    pan = pan.resize((W - 120, int(pan.height * skala)))
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    y = tulis(d, (60, 40), f"Contoh {no}. {c['judul']}", font(36, True), W - 120)
    y = tulis(d, (60, y + 4), c["jelas"], font(23), W - 120) + 10
    im.paste(pan, (60, y))
    y += pan.height + 18
    x = 60
    isian = list(zip(KOLOM, c["isi"])) + ([("Catatan", c["catatan"])] if c.get("catatan") else [])
    for k, v in isian:
        if not v:
            continue
        t1 = f"{k}: "
        d.text((x, y), t1, font=font(21, True), fill=(60, 60, 60))
        x2 = x + d.textlength(t1, font=font(21, True))
        d.rectangle([x2 - 4, y - 3, x2 + d.textlength(v, font=font(21)) + 6, y + 27], fill=(255, 242, 204))
        d.text((x2, y), v, font=font(21), fill=(20, 20, 20))
        x = x2 + d.textlength(v, font=font(21)) + 40
        if x > W - 400:
            x, y = 60, y + 36
    return im


def main() -> None:
    KERJA.mkdir(parents=True, exist_ok=True)
    pages = [halaman_sampul()] + [halaman(i, c) for i, c in enumerate(CONTOH, start=1)]
    for i, p in enumerate(pages):
        p.save(KERJA / f"hal_{i:02d}.jpg", quality=88)
    pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=150.0)
    print(f"{len(pages)} halaman -> {OUT}")


if __name__ == "__main__":
    main()

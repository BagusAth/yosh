"""Revisi lembar audit label ke versi v2 (tanpa keahlian penyakit tanaman).

Alasan (amandemen protokol, 1 November 2026): penilai bukan ahli penyakit tanaman dan
tidak ada akses ke ahli, sehingga pertanyaan "apakah kelasnya benar" dikeluarkan dari
audit manual. Yang dinilai hanya masalah label yang terlihat tanpa diagnosis:
daun tanpa kotak (dibandingkan dengan daun berkotak di citra yang sama), kotak
bersarang/ganda, objek bukan daun, kotak tidak presisi, artefak citra, citra padat.
Ketepatan kelas menjadi keterbatasan penelitian.

Amandemen dibuat sebelum ada satu citra pun dinilai dan sebelum kunci strata dibuka.
Skrip ini TIDAK membaca folder _kunci/. Kode, urutan, dan gambar citra tidak berubah;
baris diambil dari lembar v1 yang sudah ada (Kode, Jumlah kotak, Kelas pada label).

Lembar v1 dipindahkan ke audit_label_150/_arsip_v1/ (tidak dihapus).
Skrip menolak berjalan bila lembar v1 sudah berisi penilaian.

Pakai: python revisi_lembar_audit.py [--out audit_label_150] [--bangun-ulang]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

VERSI = "v2 (tanpa keahlian penyakit tanaman, 1 Nov 2026)"

KATEGORI = [
    ("Daun tanpa kotak",
     "Ada daun yang jelas terlihat tetapi tidak berada di dalam kotak mana pun (satu kotak boleh menutupi satu daun "
     "atau sekelompok daun). Citra berlabel HEALTHY: daun utuh mana pun. Citra berlabel penyakit: daun yang polanya "
     "(bercak, bintik, belang, menguning, keriting, mengering) mirip dengan daun yang SUDAH berkotak di citra yang "
     "sama. Tidak perlu tahu nama penyakitnya."),
    ("Kotak bersarang/ganda",
     "Ada kotak yang berada di dalam kotak lain, atau satu daun diberi lebih dari satu kotak."),
    ("Objek bukan daun",
     "Kotak menandai batang, tangkai, umbi, tanah, tangan, langit, atau benda lain, bukan daun."),
    ("Kotak tidak presisi",
     "Kotak memang menandai daun, tetapi terlalu longgar (banyak latar atau daun lain ikut masuk), terlalu sempit "
     "(memotong daun), atau bergeser."),
    ("Artefak citra",
     "Area rusak (hitam atau garis-garis berulang di tepi akibat rotasi) kira-kira seperempat citra atau lebih, atau "
     "citra sangat buram/gelap/terang. Sudut hitam atau garis kecil di tepi ada di banyak citra; abaikan bila kecil."),
    ("Citra padat/sulit",
     "Label tampak lengkap dan rapi, tetapi daun sangat banyak, kecil, atau saling tumpang tindih sehingga sulit "
     "dipisahkan satu sama lain."),
    ("Tidak ada masalah terlihat",
     "Semua daun yang tampak sejenis dengan daun berkotak sudah diberi kotak, kotak pas, dan citra jelas."),
    ("Lainnya", "Tidak cocok dengan kategori di atas; jelaskan di kolom Catatan."),
]
TANPA_KOTAK = ["0", "1-2", "3-5", ">5", "Ragu"]
YAKIN = ["Yakin", "Kurang yakin"]
PRESISI = ["Ya", "Sebagian", "Tidak", "Ragu"]

COLS = [  # (judul, lebar)
    ("No", 5), ("Kode", 8), ("Gambar", 11), ("Jumlah kotak", 9), ("Kelas pada label", 18),
    ("Kategori utama", 26), ("Kategori tambahan (opsional)", 26), ("Daun tanpa kotak (perkiraan)", 14),
    ("Keyakinan penilaian", 13), ("Kotak presisi?", 12), ("Catatan (sebut nomor kotak bila perlu)", 42),
]

ARIAL = "Arial"
F_BODY, F_HEAD = Font(name=ARIAL, size=10), Font(name=ARIAL, size=10, bold=True, color="FFFFFF")
F_BOLD = Font(name=ARIAL, size=10, bold=True)
FILL_HEAD, FILL_IN = PatternFill("solid", fgColor="2F5597"), PatternFill("solid", fgColor="FFF2CC")
FILL_DONE = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")


def style_header(ws, row: int, ncol: int) -> None:
    for j in range(1, ncol + 1):
        c = ws.cell(row=row, column=j)
        c.font, c.fill, c.border = F_HEAD, FILL_HEAD, BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")


def baca_v1(path: Path, sheet: str) -> list[dict]:
    wb = load_workbook(path, read_only=True)
    ws = wb[sheet]
    rows, terisi = [], 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r or not r[1]:
            continue
        if any(v not in (None, "") for v in r[5:11]):
            terisi += 1
        rows.append({"kode": str(r[1]), "n_gt": r[3], "kelas": r[4]})
    wb.close()
    if terisi:
        raise SystemExit(f"{path.name}: {terisi} baris sudah berisi penilaian. Revisi dibatalkan agar isian tidak hilang.")
    return rows


def sheet_daftar(wb) -> dict[str, str]:
    ws = wb.create_sheet("Daftar")
    ranges = {}
    for j, (name, vals) in enumerate((("Kategori", [k for k, _ in KATEGORI]), ("TanpaKotak", TANPA_KOTAK),
                                      ("Yakin", YAKIN), ("Presisi", PRESISI)), start=1):
        ws.cell(row=1, column=j, value=name).font = F_BOLD
        for i, v in enumerate(vals, start=2):
            ws.cell(row=i, column=j, value=v).font = F_BODY
        col = get_column_letter(j)
        ranges[name] = f"=Daftar!${col}$2:${col}${len(vals) + 1}"
    ws.sheet_state = "hidden"
    return ranges


def sheet_audit(wb, title: str, rows: list[dict], ranges: dict[str, str]) -> None:
    ws = wb.create_sheet(title)
    for j, (name, width) in enumerate(COLS, start=1):
        ws.cell(row=1, column=j, value=name)
        ws.column_dimensions[get_column_letter(j)].width = width
    style_header(ws, 1, len(COLS))
    ws.row_dimensions[1].height = 42
    for i, r in enumerate(rows, start=2):
        for j, v in enumerate([i - 1, r["kode"], "Buka", r["n_gt"], r["kelas"]], start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font, c.border, c.alignment = F_BODY, BORDER, WRAP
        link = ws.cell(row=i, column=3)
        link.hyperlink = f"gambar/{r['kode']}.jpg"
        link.font = Font(name=ARIAL, size=10, color="0563C1", underline="single")
        for j in range(6, len(COLS) + 1):
            c = ws.cell(row=i, column=j)
            c.fill, c.border, c.font, c.alignment = FILL_IN, BORDER, F_BODY, WRAP
    last = len(rows) + 1
    for col, key in (("F", "Kategori"), ("G", "Kategori"), ("H", "TanpaKotak"), ("I", "Yakin"), ("J", "Presisi")):
        dv = DataValidation(type="list", formula1=ranges[key], allow_blank=True, showErrorMessage=True,
                            errorTitle="Pilihan tidak dikenal", error="Pilih dari daftar (klik panah di sel).")
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")
    ws.conditional_formatting.add(f"A2:K{last}", FormulaRule(formula=['$F2<>""'], fill=FILL_DONE))
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:K{last}"


def sheet_panduan(wb, rater: str, n_rows: int, sheet_name: str) -> None:
    ws = wb.active
    ws.title = "Panduan"
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 100
    r = 1

    def tulis(a, b=None, bold=False, size=10):
        nonlocal r
        ws.cell(row=r, column=1, value=a).font = Font(name=ARIAL, size=size, bold=bold)
        ws.cell(row=r, column=1).alignment = WRAP
        if b is not None:
            c = ws.cell(row=r, column=2, value=b)
            c.font, c.alignment = F_BODY, WRAP
        r += 1

    tulis(f"Audit label manual — {rater}", bold=True, size=13)
    tulis(f"Versi lembar: {VERSI}")
    tulis(f"Isi sheet \"{sheet_name}\" ({n_rows} citra). Sel kuning diisi; baris berubah hijau setelah Kategori utama terisi.")
    r += 1
    tulis("Yang TIDAK perlu dinilai", None, bold=True)
    tulis("Benar-salahnya kelas penyakit",
          "Kamu tidak perlu tahu nama penyakit atau membedakan daun sehat dan sakit secara medis. Kolom \"Kelas pada label\" "
          "hanya dipakai untuk tahu apakah citra berlabel HEALTHY (semua daun utuh seharusnya berkotak) atau berlabel "
          "penyakit (daun yang mirip daun berkotak seharusnya berkotak).")
    r += 1
    tulis("Langkah", None, bold=True)
    for a, b in (
        ("1", "Klik \"Buka\" di kolom Gambar (atau buka gambar/Annn.jpg). Kiri citra asli, kanan kotak label bernomor."),
        ("2", "Lihat daun-daun yang SUDAH berkotak di panel kanan. Cari di panel kiri daun lain yang tampak sejenis tetapi "
              "tidak berkotak. Perkirakan jumlahnya di kolom \"Daun tanpa kotak\"."),
        ("3", "Periksa setiap kotak: apakah ada kotak di dalam kotak lain, kotak ganda pada satu daun, kotak yang menandai "
              "bukan daun, atau kotak yang tidak pas."),
        ("4", "Pilih Kategori utama. Bila ada lebih dari satu masalah, dahulukan masalah label (4 kategori teratas) daripada "
              "kondisi citra (artefak, padat). Di antara masalah label, pilih yang menyangkut paling banyak daun; sisanya "
              "boleh ditulis di Kategori tambahan."),
        ("5", "Isi Keyakinan penilaian: \"Kurang yakin\" bila kamu ragu dengan jawabanmu sendiri. Ini wajar dan tetap dipakai "
              "dalam analisis (sebagai uji sensitivitas)."),
        ("6", "Lihat PANDUAN_BERGAMBAR.pdf untuk contoh tiap kategori (contoh diambil dari citra di luar sampel audit)."),
        ("7", "Jangan membuka folder _kunci dan jangan membandingkan dengan prediksi model sebelum semua citra dinilai. "
              "Urutan citra sengaja diacak."),
        ("8", "Simpan file secara berkala. Perkiraan waktu: sekitar 1 menit per citra; boleh dicicil."),
    ):
        tulis(f"   {a}", b)
    r += 1
    ws.cell(row=r, column=1, value="Kategori")
    ws.cell(row=r, column=2, value="Gunakan bila")
    style_header(ws, r, 2)
    r += 1
    for k, desc in KATEGORI:
        for j, v in enumerate((k, desc), start=1):
            c = ws.cell(row=r, column=j, value=v)
            c.font, c.border, c.alignment = F_BODY, BORDER, WRAP
        r += 1
    r += 1
    tulis("Kolom lain", None, bold=True)
    for a, b in (
        ("Daun tanpa kotak (perkiraan)", "Perkiraan jumlah daun yang seharusnya berkotak tetapi tidak: 0, 1-2, 3-5, >5, atau Ragu."),
        ("Keyakinan penilaian", "Yakin atau Kurang yakin terhadap penilaian baris ini."),
        ("Kotak presisi?", "Apakah batas kotak pas dengan daunnya: Ya, Sebagian, Tidak, Ragu."),
        ("Catatan", "Opsional. Sebut nomor kotak bila menunjuk kotak tertentu, misalnya \"kotak 3 ada di dalam kotak 1\"."),
    ):
        tulis(a, b)
    r += 1
    tulis("Contoh pengisian (bukan data)", None, bold=True)
    contoh = ["Daun tanpa kotak", "Kotak tidak presisi", "3-5", "Yakin", "Sebagian",
              "Hanya 2 dari sekitar 6 daun berbercak serupa yang berkotak; kotak 2 terlalu longgar."]
    for (hd, _), v in zip(COLS[5:], contoh):
        ws.cell(row=r, column=1, value=hd).font = F_BODY
        c = ws.cell(row=r, column=2, value=v)
        c.font, c.fill = Font(name=ARIAL, size=10, italic=True), FILL_IN
        r += 1


def sheet_progres(wb, sheet_name: str, n_rows: int) -> None:
    ws = wb.create_sheet("Progres")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 12
    last = n_rows + 1
    ws["A1"], ws["B1"] = "Ringkasan pengisian", "Jumlah"
    style_header(ws, 1, 2)
    ws["A2"], ws["B2"] = "Citra dinilai", f"=COUNTA('{sheet_name}'!F2:F{last})"
    ws["A3"], ws["B3"] = "Citra total", n_rows
    ws["A4"], ws["B4"] = "Selesai", "=IF(B3=0,0,B2/B3)"
    ws["B4"].number_format = "0%"
    ws["A6"], ws["B6"] = "Kategori utama", "Jumlah"
    style_header(ws, 6, 2)
    i = 7
    for k, _ in KATEGORI:
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=f"=COUNTIF('{sheet_name}'!F2:F{last},A{i})")
        i += 1
    for row in ws.iter_rows(min_row=2, max_row=i - 1):
        for c in row:
            if not (c.font and c.font.bold):
                c.font = F_BODY
    ws.cell(row=i + 1, column=1, value="Hitungan per strata baru dibuat setelah audit selesai.").font = \
        Font(name=ARIAL, size=9, italic=True)


def buat(path: Path, rater: str, sheet_name: str, rows: list[dict]) -> None:
    wb = Workbook()
    sheet_panduan(wb, rater, len(rows), sheet_name)
    ranges = sheet_daftar(wb)
    sheet_audit(wb, sheet_name, rows, ranges)
    sheet_progres(wb, sheet_name, len(rows))
    wb.move_sheet("Daftar", offset=10)
    wb.active = 1
    wb.save(path)


def sudah_v2(path: Path) -> bool:
    try:
        wb = load_workbook(path, read_only=True)
        v = wb["Panduan"]["A2"].value
        wb.close()
        return isinstance(v, str) and v.startswith("Versi lembar: v2")
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit_label_150")
    ap.add_argument("--bangun-ulang", action="store_true",
                    help="Bangun ulang lembar v2 yang masih kosong (menolak bila sudah ada isian)")
    args = ap.parse_args()
    out = Path(args.out)
    arsip = out / "_arsip_v1"
    arsip.mkdir(exist_ok=True)
    for fname, rater, sheet in (("LEMBAR_AUDIT_LABEL.xlsx", "rater 1", "Audit"),
                                ("LEMBAR_AUDIT_RATER2.xlsx", "rater 2 (opsional, 30 citra)", "Rater2")):
        p = out / fname
        if sudah_v2(p):
            if not args.bangun_ulang:
                print(f"{fname}: sudah v2, dilewati.")
                continue
            rows = baca_v1(p, sheet)  # kolom identitas sama; menolak bila sudah ada isian
            buat(p, rater, sheet, rows)
            print(f"{fname}: {len(rows)} baris -> v2 dibangun ulang (masih kosong)")
            continue
        rows = baca_v1(p, sheet)
        tujuan = arsip / fname.replace(".xlsx", "_v1.xlsx")
        if tujuan.exists():
            raise SystemExit(f"{tujuan} sudah ada; periksa manual.")
        shutil.move(str(p), str(tujuan))
        buat(p, rater, sheet, rows)
        print(f"{fname}: {len(rows)} baris -> v2 (v1 dipindah ke {tujuan})")


if __name__ == "__main__":
    main()

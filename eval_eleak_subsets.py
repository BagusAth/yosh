#!/usr/bin/env python3
"""
E-LEAK (RQ1 bagian A, Pakem v1.3) — evaluasi model pada beberapa subset validasi + laporan selisih
==================================================================================================

Jalankan di .venv-cuda setelah model simulasi dilatih:

  # Model split acak per citra (simulasi kebocoran)
  python eval_eleak_subsets.py --tag sim --weights <best.pt simulasi> --out <folder> \
      --sets val_all=eleak_sim_rq1/data_eleak_sim.yaml val_leaked=eleak_sim_rq1/data_eleak_sim_val_leaked.yaml \
             val_twinval=eleak_sim_rq1/data_eleak_sim_val_twinval.yaml val_single=eleak_sim_rq1/data_eleak_sim_val_single.yaml
  # Model split per grup (EXP-0 seed sama)
  python eval_eleak_subsets.py --tag group --weights <best.pt EXP-0> --out <folder> \
      --sets val_all=cassava-v7-runtime-mirror/data.yaml val_twin=eleak_sim_rq1/data_clean_val_twin.yaml
  # Laporan selisih
  python eval_eleak_subsets.py --report --out <folder>
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import yaml

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
IMG_EXT = {".jpg", ".jpeg", ".png"}
FIELDS = ["tag", "subset", "data_yaml", "weights", "n_images", "precision", "recall", "map50", "map50_95"] + [
    f"ap50_95_{c}" for c in CLASS_NAMES
]


def count_val_images(data_yaml: Path) -> int:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(str(d.get("path", data_yaml.parent)))
    v = Path(str(d["val"]))
    if not v.is_absolute():
        v = base / v
    if str(v).endswith(".txt"):
        return sum(1 for line in v.read_text(encoding="utf-8").splitlines() if line.strip())
    return sum(1 for f in os.listdir(v) if os.path.splitext(f)[1].lower() in IMG_EXT)


def evaluate(args) -> None:
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_p = out / "eleak_subset_metrics.csv"
    new_file = not csv_p.exists()
    with open(csv_p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for spec in args.sets:
            name, yml = spec.split("=", 1)
            yml_p = Path(yml).resolve()
            m = YOLO(args.weights).val(
                data=str(yml_p), split="val", batch=args.batch, imgsz=args.imgsz, device=args.device,
                workers=args.workers, plots=False, project=str(out), name=f"{args.tag}_{name}", exist_ok=True,
            )
            row = {
                "tag": args.tag, "subset": name, "data_yaml": str(yml_p), "weights": args.weights,
                "n_images": count_val_images(yml_p), "precision": f"{float(m.box.mp):.6f}",
                "recall": f"{float(m.box.mr):.6f}", "map50": f"{float(m.box.map50):.6f}", "map50_95": f"{float(m.box.map):.6f}",
            }
            for i, c in enumerate(CLASS_NAMES):
                try:
                    row[f"ap50_95_{c}"] = f"{float(m.box.maps[i]):.6f}"
                except Exception:
                    row[f"ap50_95_{c}"] = ""
            w.writerow(row)
            f.flush()
            print(f"[{args.tag}/{name}] n={row['n_images']} mAP50={row['map50']} mAP50-95={row['map50_95']}")


def report(args) -> None:
    out = Path(args.out)
    rows = list(csv.DictReader(open(out / "eleak_subset_metrics.csv", newline="", encoding="utf-8")))
    get = {(r["tag"], r["subset"]): r for r in rows}

    def val(tag, sub, key="map50_95"):
        r = get.get((tag, sub))
        return float(r[key]) * 100 if r else None

    def delta(a, b):
        return f"{a - b:+.2f}" if (a is not None and b is not None) else "—"

    lines = [
        "# E-LEAK (RQ1 bagian A) — Inflasi metrik akibat split per citra setelah augmentasi",
        f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Model | Subset | Citra | Precision | Recall | mAP@50 | mAP@50–95 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['subset']} | {int(r['n_images']):,} | {float(r['precision'])*100:.2f}% | {float(r['recall'])*100:.2f}% | "
            f"{float(r['map50'])*100:.2f}% | {float(r['map50_95'])*100:.2f}% |"
        )
    lines += [
        "",
        "Selisih mAP@50–95 (poin):",
        f"- Dalam model simulasi, populasi sama: bocor − kembar-di-validasi = {delta(val('sim','val_leaked'), val('sim','val_twinval'))}",
        f"- Angka yang akan dilaporkan peneliti: validasi split acak − validasi split per grup = {delta(val('sim','val_all'), val('group','val_all'))}",
        f"- Antar model, populasi sama: simulasi bocor − split per grup (grup 2 citra) = {delta(val('sim','val_leaked'), val('group','val_twin'))}",
        "",
        "Catatan: subset bocor dan kembar-di-validasi berasal dari populasi yang sama (grup 2 citra hasil augmentasi Roboflow),",
        "sehingga selisih pertama paling bersih mengukur efek kebocoran. Interval kepercayaan bootstrap berpasangan dihitung di Fase 5.",
    ]
    (out / "ELEAK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--weights")
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        if not (args.tag and args.weights and args.sets):
            raise SystemExit("--tag, --weights, dan --sets wajib diisi untuk evaluasi")
        evaluate(args)


if __name__ == "__main__":
    main()

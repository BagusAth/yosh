"""Fase 6: angka val() Ultralytics pada test set sebagai pendamping (bukan metrik utama).

Dipanggil oleh jalankan_fase6.ps1 dengan daftar model "kunci=path". Hasil ditulis per model
(aman dilanjutkan bila terputus): stage_v2_fase6/val_test.json dengan kunci yang sama seperti
analyze_ablation.py ("<seed>|<label>"), ditambah kunci CONV ("conv|EXP-0 60ep", ...).

Pakai: python val_test_fase6.py --data <data.yaml dengan test:> --out <json> --models k1=path1 k2=path2 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--project", default="stage_v2_fase6/val_runs")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO

    out = Path(a.out)
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for item in a.models:
        key, path = item.split("=", 1)
        if key in res:
            print(f"lewati {key} (sudah ada)")
            continue
        m = YOLO(path)
        r = m.val(data=a.data, split="test", imgsz=640, batch=16, conf=0.001, iou=0.7, device=a.device,
                  plots=False, verbose=False, project=a.project,
                  name=key.replace("|", "_").replace(" ", "_"), exist_ok=True)
        res[key] = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map), "weights": path}
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"{key}: mAP50-95 {100 * r.box.map:.2f}")


if __name__ == "__main__":
    main()

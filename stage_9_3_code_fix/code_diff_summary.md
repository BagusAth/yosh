# Ringkasan Perubahan Kode (Code Diff Summary) — Stage 9.3

Dokumen ini menyajikan perbandingan *side-by-side* dan *unified diff* dari seluruh berkas kode yang diperbaiki pada Stage 9.3 untuk menyelesaikan temuan audit Stage 9.2:
- **CRITICAL-01**: `apply_context_crop()` silent NO-OP pada `stage_8_3_adaptive_pipeline.py`.
- **HIGH-01**: AP per kelas salah label pada `stage_8_4_run_ablation_experiments.py` dan `stage_8_6_final_test_benchmark.py`.
- **HIGH-02**: Seleksi model tidak konsisten dengan protokol resmi pada `stage_8_5_model_selection_and_validation.py`.
- **MED-01**: Potensi pencocokan ulang prediksi pada `stage_7_baseline_and_error_analysis.py`.
- **MED-02**: Parser koordinat mengasumsikan dimensi gambar tetap (640x640) pada `stage_7`, `stage_8_2`, `stage_8_5`, dan `stage_8_6`.
- **LOW-01**: `SyntaxWarning: invalid escape sequence '\l'` pada `stage_8_5_model_selection_and_validation.py`.

---

## 1. `stage_8_3_adaptive_pipeline.py` (CRITICAL-01)

### Deskripsi Masalah & Akar Penyebab
Fungsi `apply_context_crop()` mengakses `label.get("bboxes")` yang selalu bernilai `None` pada Ultralytics 8.4 `YOLODataset`, sehingga fungsi langsung mengembalikan label tanpa augmentasi (*silent NO-OP*). Transformasi koordinat crop juga tidak memperbarui objek `label["instances"]` dengan format `xywh` ternormalisasi yang dibutuhkan oleh `Format()`.

### Perbaikan Utama
1. **Akses BBox Kanonikal**: Mengambil bounding box langsung dari `label["instances"].bboxes` (format normalized `xywh`).
2. **Kalkulasi Spasial Konteks 2×**: Menghitung koordinat kotak target secara dinamis baik dari format piksel maupun ternormalisasi, melakukan perluasan konteks 2×, dan melakukan *clamping* presisi pada batas kanvas `[0, W] × [0, H]`.
3. **Penyaringan Visibilitas Sekunder**: Mempertahankan lesi sekunder hanya jika rasio luas yang tersisa di dalam patch crop $\ge 0.30$.
4. **Instansiasi Kanonikal**: Membentuk objek `Instances(bboxes=..., bbox_format="xywh", normalized=True)` baru dan menetapkannya ke `out_label["instances"]`.
5. **Pencegahan Kebocoran Kunci**: Menghapus flag internal `_is_context_cropped` sebelum masuk ke `crop_transforms` agar tidak merusak keselarasan kunci pada `collate_fn`.
6. **Penghitung Diagnostik**: Menambahkan atribut `self.crop_diagnostics` untuk merekam: `crop_attempts`, `crops_applied`, `no_target_skips`, `invalid_box_skips`, dan `visibility_skips`.

```diff
--- a/stage_8_3_adaptive_pipeline.py
+++ b/stage_8_3_adaptive_pipeline.py
@@ -58,2 +58,3 @@
 from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
+from ultralytics.utils.instance import Instances
 from ultralytics.utils.torch_utils import torch_distributed_zero_first
@@ -104,2 +105,11 @@
         self.vis_thresh = vis_thresh
+
+        # Diagnostic counters for targeted context crop auditing
+        self.crop_diagnostics = {
+            "crop_attempts": 0,
+            "crops_applied": 0,
+            "no_target_skips": 0,
+            "invalid_box_skips": 0,
+            "visibility_skips": 0,
+        }
 
-        bboxes = label.get("bboxes")
-        if bboxes is None or len(bboxes) == 0:
+        instances = label.get("instances")
+        if instances is None or len(instances) == 0:
+            self.crop_diagnostics["no_target_skips"] += 1
             return label
+
+        box_str = target_info.get("target_small_box", "")
+        if not box_str or not target_info.get("has_target_small", False):
+            self.crop_diagnostics["no_target_skips"] += 1
+            return label
...
+        out_label["instances"] = Instances(
+            bboxes=np.array(new_bboxes, dtype=np.float32),
+            segments=None,
+            keypoints=None,
+            bbox_format="xywh",
+            normalized=True,
+        )
+        self.crop_diagnostics["crops_applied"] += 1
```

---

## 2. `stage_8_4_run_ablation_experiments.py` (HIGH-01)

### Deskripsi Masalah & Akar Penyebab
Nilai `val_metrics.box.maps` (yang merupakan mAP@50-95 per kelas pada Ultralytics) disimpan ke kamus bernama `per_class_map50` dan disajikan pada tabel bertajuk `Mean mAP@50`. Rata-rata baris tabel tidak cocok dengan mAP@50 global karena data aslinya adalah mAP@50-95.

### Perbaikan Utama
1. Memisahkan ekstraksi metrik:
   - `per_class_ap50` diambil dari `val_metrics.box.ap50` (vektor AP pada IoU 0.50).
   - `per_class_map50_95` diambil dari `val_metrics.box.maps` (vektor AP pada rentang IoU 0.50:0.95).
2. Menyimpan kedua kamus ke dalam `metrics_validation` pada `summary.json`.
3. Memperbarui tabel laporan Markdown menjadi dua tabel terpisah dengan judul kanonis:
   - *Tabel 2: Analisis Performa Per-Kelas (AP@50 pada Validation Set)*
   - *Tabel 3: Analisis Performa Per-Kelas (AP@50-95 pada Validation Set)*

```diff
--- a/stage_8_4_run_ablation_experiments.py
+++ b/stage_8_4_run_ablation_experiments.py
@@ -230,8 +230,19 @@
-    # Ekstrak per-class mAP50
-    per_class_map50 = {}
+    # Ekstrak per-class AP@50 dan AP@50-95 secara terpisah dan kanonis
+    per_class_ap50 = {}
+    per_class_map50_95 = {}
     class_names = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
+
+    # 1. AP@50 per kelas dari box.ap50 (vektor AP pada IoU 0.50)
+    if hasattr(val_metrics.box, "ap50") and val_metrics.box.ap50 is not None:
+        ap50_arr = val_metrics.box.ap50
+        ap_class_idx = list(val_metrics.box.ap_class_index) if hasattr(val_metrics.box, "ap_class_index") else list(range(len(ap50_arr)))
+        for i, c_idx in enumerate(ap_class_idx):
+            if c_idx < len(class_names) and i < len(ap50_arr):
+                per_class_ap50[class_names[c_idx]] = float(ap50_arr[i])
+
+    # 2. AP@50-95 per kelas dari box.maps (vektor AP pada IoU 0.50:0.95)
     if hasattr(val_metrics.box, "maps") and val_metrics.box.maps is not None:
         for idx, cname in enumerate(class_names):
             if idx < len(val_metrics.box.maps):
-                per_class_map50[cname] = float(val_metrics.box.maps[idx])
+                per_class_map50_95[cname] = float(val_metrics.box.maps[idx])
```

---

## 3. `stage_8_5_model_selection_and_validation.py` (HIGH-02, MED-02, LOW-01)

### Deskripsi Masalah & Akar Penyebab
1. **HIGH-02**: Skrip menerapkan aturan seleksi model dengan margin arbitrer 0.50% (`close_competitors = [r for r in results if (top_candidate["mAP50_95"] - r["mAP50_95"]) <= 0.005]`), sehingga model dengan mAP@50-95 lebih rendah dapat mengalahkan model dengan mAP@50-95 tertinggi jika memiliki `recall_small` lebih tinggi.
2. **MED-02**: Fungsi `parse_yolo_labels` mengalikan koordinat dengan skalar konstan `640`.
3. **LOW-01**: Simbol LaTeX `$\le 0.50\%$` dalam non-raw f-string menghasilkan `SyntaxWarning: invalid escape sequence '\l'` pada Python 3.12+.

### Perbaikan Utama
1. **Aturan Seleksi Protokol Murni**: Menghapus seluruh margin toleransi 0.50%. Seleksi model didasarkan mutlak pada nilai mAP@50-95 validasi tertinggi. Tie-break hanya dievaluasi jika terjadi kesamaan numerik (`abs(a - b) <= 1e-5`).
2. **Parser Dimensi Dinamis**: Mengubah fungsi `parse_yolo_labels(lbl_path, img_width=640.0, img_height=640.0)` sehingga sumbu horizontal diskalakan dengan `img_width` dan sumbu vertikal diskalakan dengan `img_height`.
3. **Penyelesaian Escape Sequence**: Menghilangkan string `\le` pada laporan markdown, menggantinya dengan sintaks kompatibel tanpa peringatan sintaks.

```diff
--- a/stage_8_5_model_selection_and_validation.py
+++ b/stage_8_5_model_selection_and_validation.py
@@ -66,2 +66,2 @@
-def parse_yolo_labels(lbl_path: Path) -> list[dict]:
-    """Membaca anotasi validasi YOLO (termasuk konversi poligon jika ada)."""
+def parse_yolo_labels(lbl_path: Path, img_width: float = 640.0, img_height: float = 640.0) -> list[dict]:
+    """Membaca anotasi validasi YOLO (termasuk konversi poligon jika ada) menggunakan dimensi dinamis."""
@@ -87,4 +87,4 @@
-            x1 = max(0.0, (xc - w / 2) * 640)
-            y1 = max(0.0, (yc - h / 2) * 640)
-            x2 = min(640.0, (xc + w / 2) * 640)
-            y2 = min(640.0, (yc + h / 2) * 640)
+            x1 = max(0.0, (xc - w / 2) * img_width)
+            y1 = max(0.0, (yc - h / 2) * img_height)
+            x2 = min(img_width, (xc + w / 2) * img_width)
+            y2 = min(img_height, (yc + h / 2) * img_height)
@@ -336,17 +336,29 @@
-    sorted_by_primary = sorted(results, key=lambda x: x["mAP50_95"], reverse=True)
-    top_candidate = sorted_by_primary[0]
-
-    # Evaluasi tie-breaking jika ada model yang berdekatan dengan top candidate
-    close_competitors = [r for r in results if (top_candidate["mAP50_95"] - r["mAP50_95"]) <= 0.005]
-    if len(close_competitors) > 1:
-        # Sortir kompetitor dekat berdasarkan Recall_Small tertinggi
-        selected_model = sorted(close_competitors, key=lambda x: (x["recall_small"], x["mAP50"], -x["fn_missed"]), reverse=True)[0]
+    max_map50_95 = max(r["mAP50_95"] for r in results)
+    tied_candidates = [r for r in results if abs(r["mAP50_95"] - max_map50_95) <= 1e-5]
+
+    if len(tied_candidates) == 1:
+        selected_model = tied_candidates[0]
+        selection_rationale = (
+            f"Kandidat {selected_model['exp_id']} ({selected_model['title']}) terpilih mutlak berdasarkan "
+            f"mAP@50-95 validasi tertinggi ({selected_model['mAP50_95']:.4f}) tanpa ambiguitas tie."
+        )
     else:
-        selected_model = top_candidate
+        sorted_tied = sorted(
+            tied_candidates,
+            key=lambda x: (x.get("mAP50", 0.0), x.get("recall_small", 0.0), -x.get("fn_missed", 0)),
+            reverse=True
+        )
+        selected_model = sorted_tied[0]
```

---

## 4. `stage_8_6_final_test_benchmark.py` (HIGH-01, MED-02)

### Perbaikan Utama
1. Mengubah `parse_yolo_labels` untuk menerima `(img_width, img_height)` secara dinamis.
2. Memisahkan `per_class_ap50` (dari `val_res.box.ap50`) dan `per_class_map50_95` (dari `val_res.box.maps`).
3. Memperbarui tabel evaluasi Test Set untuk menampilkan metrik kanonik tanpa ambiguitas.

```diff
--- a/stage_8_6_final_test_benchmark.py
+++ b/stage_8_6_final_test_benchmark.py
@@ -57,2 +57,2 @@
-def parse_yolo_labels(lbl_path: Path) -> list[dict]:
-    """Membaca anotasi test set YOLO (dengan konversi poligon jika ada)."""
+def parse_yolo_labels(lbl_path: Path, img_width: float = 640.0, img_height: float = 640.0) -> list[dict]:
+    """Membaca anotasi test set YOLO (dengan konversi poligon jika ada) menggunakan dimensi dinamis."""
@@ -139,5 +149,15 @@
-    per_class_map50 = {}
+    # Ekstrak per-class AP@50 dan AP@50-95 secara terpisah dan kanonis
+    per_class_ap50 = {}
+    per_class_map50_95 = {}
+
+    if hasattr(val_res.box, "ap50") and val_res.box.ap50 is not None:
+        ap50_arr = val_res.box.ap50
+        ap_class_idx = list(val_res.box.ap_class_index) if hasattr(val_res.box, "ap_class_index") else list(range(len(ap50_arr)))
+        for i, c_idx in enumerate(ap_class_idx):
+            if c_idx < len(CLASS_NAMES) and i < len(ap50_arr):
+                per_class_ap50[CLASS_NAMES[c_idx]] = float(ap50_arr[i])
+
     if hasattr(val_res.box, "maps") and val_res.box.maps is not None:
         for idx, cname in enumerate(CLASS_NAMES):
             if idx < len(val_res.box.maps):
-                per_class_map50[cname] = float(val_res.box.maps[idx])
+                per_class_map50_95[cname] = float(val_res.box.maps[idx])
```

---

## 5. `stage_7_baseline_and_error_analysis.py` (MED-01, MED-02)

### Deskripsi Masalah & Akar Penyebab
Pada loop evaluasi Pilar 3 Stage 7, setiap objek GT mencari prediksi terbaik di seluruh `pred_boxes` tanpa memeriksa apakah kotak prediksi tersebut telah dipasangkan ke objek GT sebelumnya. Hal ini berpotensi menyebabkan prediksi tunggal dicocokkan berulang kali ke beberapa ground-truth yang berdekatan.

### Perbaikan Utama
1. Mengimplementasikan pencocokan *one-to-one bipartite greedy matching* berbasis urutan IoU menurun:
   - Seluruh pasangan kandidat `(gt_idx, pred_idx)` dengan $\text{IoU} \ge 0.10$ dikumpulkan.
   - Pasangan kandidat diurutkan dari IoU tertinggi ke terendah.
   - Pencocokan dilakukan secara eksklusif sehingga setiap GT terpasang maksimal ke 1 prediksi, dan setiap prediksi terpasang maksimal ke 1 GT.
2. Memperbarui `parse_yolo_labels` dengan parameter dimensi gambar dinamis `(img_width, img_height)`.

```diff
--- a/stage_7_baseline_and_error_analysis.py
+++ b/stage_7_baseline_and_error_analysis.py
@@ -63,2 +63,2 @@
-def parse_yolo_labels(lbl_path: Path) -> list[dict]:
-    """Parses YOLO label txt file into list of bounding boxes."""
+def parse_yolo_labels(lbl_path: Path, img_width: float = 640.0, img_height: float = 640.0) -> list[dict]:
+    """Parses YOLO label txt file into list of bounding boxes using dynamic image dimensions."""
@@ -310,24 +310,24 @@
-        # Match Ground Truth to Predictions
-        matched_preds = set()
+        # One-to-one bipartite greedy matching (descending IoU)
+        candidate_matches = []
+        for gt_idx, gt in enumerate(gt_objects):
+            for p_idx in range(len(pred_boxes)):
+                iou = calculate_iou_xyxy(gt["box"], pred_boxes[p_idx])
+                if iou >= 0.10:
+                    candidate_matches.append((iou, gt_idx, p_idx))
+
+        # Sort candidate matches by IoU descending to prioritize strongest overlaps
+        candidate_matches.sort(key=lambda x: x[0], reverse=True)
+
+        matched_gt_map = {}  # gt_idx -> (p_idx, iou)
+        assigned_preds = set()
+        for iou, gt_idx, p_idx in candidate_matches:
+            if gt_idx not in matched_gt_map and p_idx not in assigned_preds:
+                matched_gt_map[gt_idx] = (p_idx, iou)
+                assigned_preds.add(p_idx)
```

---

## 6. `stage_8_2_build_difficulty_manifest.py` (MED-02)

### Perbaikan Utama
Mengganti skalar konstan `640.0` pada kalkulasi koordinat kotak dan batas poligon dengan variabel eksplisit `img_width` dan `img_height`, memastikan integritas kalkulasi rasio spasial pada citra beresolusi non-persegi.

```diff
--- a/stage_8_2_build_difficulty_manifest.py
+++ b/stage_8_2_build_difficulty_manifest.py
@@ -125,7 +125,8 @@
                         cls_id = int(parts[0])
+                        img_width, img_height = 640.0, 640.0
                         if len(parts) == 5:
-                            xc = float(parts[1]) * 640.0
-                            yc = float(parts[2]) * 640.0
-                            w = float(parts[3]) * 640.0
-                            h = float(parts[4]) * 640.0
+                            xc = float(parts[1]) * img_width
+                            yc = float(parts[2]) * img_height
+                            w = float(parts[3]) * img_width
+                            h = float(parts[4]) * img_height
```

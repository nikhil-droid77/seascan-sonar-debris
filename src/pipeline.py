r"""Stage 2 - end-to-end pipeline: image -> preprocess -> YOLOv8 -> filter -> geotag -> report.

Example:
  python -m src.pipeline --image data\Sonar_Dataset\test\images\SOME.jpg ^
      --weights runs\detect\baseline\weights\best.pt --lat 15.30 --lon 73.80 --heading 45 --swath 100
"""
import argparse
import json
import os

import cv2
import pandas as pd

from .filtering import filter_detections
from .geotag import SonarMeta, pixel_to_latlon
from .preprocess import preprocess


def detect(model, img, conf=0.05):
    res = model.predict(img, conf=conf, imgsz=416, verbose=False)[0]
    names = res.names
    dets = []
    for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
        dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c, "cls": int(k), "label": names[int(k)]})
    return dets


def run(image_path, weights, meta, use_preprocess=False, conf_thr=0.25, out_dir="outputs"):
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Could not read {image_path}")
    h, w = img.shape[:2]
    model_in = preprocess(img) if use_preprocess else img

    raw = detect(YOLO(weights), model_in)
    kept = filter_detections(raw, w, h, conf_thr=conf_thr)

    rows = []
    for i, d in enumerate(kept, 1):
        lat, lon = pixel_to_latlon((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2, meta, w)
        rows.append({"id": i, "image": os.path.basename(image_path), "label": d["label"],
                     "confidence": round(d["conf"], 3), "lat": round(lat, 6), "lon": round(lon, 6),
                     "x1": round(d["x1"]), "y1": round(d["y1"]), "x2": round(d["x2"]), "y2": round(d["y2"])})
        cv2.rectangle(img, (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"])), (0, 255, 0), 2)
        cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (int(d["x1"]), max(int(d["y1"]) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "report.csv"), index=False)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rows, f, indent=2)
    cv2.imwrite(os.path.join(out_dir, "annotated.png"), img)
    print(f"{len(raw)} raw detections -> {len(rows)} after filtering. Saved to {out_dir}\\")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--lat", type=float, default=15.30)
    p.add_argument("--lon", type=float, default=73.80)
    p.add_argument("--heading", type=float, default=0.0)
    p.add_argument("--swath", type=float, default=100.0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--preprocess", action="store_true", help="apply preprocessing before detection")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    run(a.image, a.weights, SonarMeta(a.lat, a.lon, a.heading, a.swath), a.preprocess, a.conf, a.out)

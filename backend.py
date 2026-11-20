import os
import glob
import random
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("runs", "detect", "stage3_natural", "weights", "best.pt")
model = YOLO(MODEL_PATH)

def run_yolo_inference(img_bgr, conf_thresh, base_lat, base_lon, filename="scan.jpg"):
    H, W = img_bgr.shape[:2]
    results = model.predict(img_bgr, imgsz=320, conf=conf_thresh, verbose=False)[0]

    annotated_bgr = img_bgr.copy()
    GREEN = (60, 232, 120)
    CORAL = (74, 107, 255)
    HAZARDS = {"Mine", "Propeller", "Standing-person", "Hook"}
    detections = []

    for idx, box in enumerate(results.boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        debris_type = model.names[cls_id]
        obj_id = f"D{idx:02d}"
        label = f"{obj_id} {debris_type} {conf:.2f}"

        is_hazard = debris_type in HAZARDS
        box_color = CORAL if is_hazard else GREEN
        status = "CRITICAL" if debris_type in {"Mine", "Standing-person"} else ("WARNING" if is_hazard else "REVIEW")

        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), box_color, 2)
        text_y = max(18, y1 - 6)
        cv2.putText(annotated_bgr, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        cx_norm = ((x1 + x2) / 2) / W
        cy_norm = ((y1 + y2) / 2) / H
        est_lat = round(base_lat + (cy_norm - 0.5) * 0.005, 6)
        est_lon = round(base_lon + (cx_norm - 0.5) * 0.005, 6)

        range_m = round(abs(cx_norm - 0.5) * 100, 1)
        side = "S" if cx_norm >= 0.5 else "P"
        along_m = round(cy_norm * 150, 1)
        w_m = round(((x2 - x1) / W) * 25, 1)
        h_m = round(((y2 - y1) / H) * 25, 1)

        detections.append({
            "id": obj_id,
            "type": debris_type,
            "confidence": round(conf * 100, 1),
            "bbox": [x1, y1, x2, y2],
            "lat": est_lat,
            "lon": est_lon,
            "range": f"{range_m} {side}",
            "along_track": along_m,
            "size": f"{w_m}x{h_m}",
            "status": status
        })

    _, orig_buf = cv2.imencode(".jpg", img_bgr)
    _, ann_buf = cv2.imencode(".jpg", annotated_bgr)

    return {
        "filename": filename,
        "resolution": f"{W} x {H} px",
        "original_image": f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode('utf-8')}",
        "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(ann_buf).decode('utf-8')}",
        "detections": detections
    }

@app.route("/")
def index():
    return jsonify({"status": "SEASCAN YOLOv8 API Online"})

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    conf_thresh = float(request.form.get("conf", 0.30))
    base_lat = float(request.form.get("lat", 13.0827))
    base_lon = float(request.form.get("lon", 80.2707))

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid image format"}), 400

    return jsonify(run_yolo_inference(img_bgr, conf_thresh, base_lat, base_lon, file.filename))

@app.route("/demo-scan", methods=["POST"])
def demo_scan():
    conf_thresh = float(request.form.get("conf", 0.30))
    base_lat = float(request.form.get("lat", 13.0827))
    base_lon = float(request.form.get("lon", 80.2707))

    val_images = glob.glob(os.path.join("data", "images", "stage3_val", "*.jpg"))
    if not val_images:
        return jsonify({"error": "No demo images found"}), 404

    chosen = random.choice(val_images)
    img_bgr = cv2.imread(chosen)
    return jsonify(run_yolo_inference(img_bgr, conf_thresh, base_lat, base_lon, os.path.basename(chosen)))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
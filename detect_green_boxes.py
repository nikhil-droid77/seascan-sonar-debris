import os
import glob
import cv2
from ultralytics import YOLO

MODEL_PATH = r"runs\detect\stage3_natural\weights\best.pt"
SOURCE_DIR = r"data\images\stage3_val"
OUTPUT_DIR = r"runs\detect\green_box_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
GREEN = (0, 255, 0)  # Exact bright green box & floating text style

test_images = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.jpg")))[:25]

for img_path in test_images:
    img = cv2.imread(img_path)
    results = model.predict(img, imgsz=320, conf=0.30, verbose=False)[0]

    detected_list = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = f"{model.names[cls_id]} {conf:.2f}"
        detected_list.append(label)

        # Draw 2px bright green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)

        # Draw floating bright green class + confidence text directly above top-left corner
        text_y = max(18, y1 - 6)
        cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), img)
    print(f"{os.path.basename(img_path)} -> Detected: {', '.join(detected_list) if detected_list else 'None'}")

print(f"\nSaved all green-box results to: {OUTPUT_DIR}")
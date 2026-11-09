import os
from ultralytics import YOLO

# Load your custom trained model
model = YOLO(r"runs\detect\train-4\weights\best.pt")

# Run prediction on seed training images or new test images
results = model.predict(source=r"data\images\train", save=True, imgsz=640, conf=0.25)

print("Inference complete! Results saved in runs/detect/predict")
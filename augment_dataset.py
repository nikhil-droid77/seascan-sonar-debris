import os
import glob
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Path Configuration
IMAGES_DIR = r"data\images\train"
LABELS_DIR = r"data\labels\train"
OUT_IMAGES_DIR = r"data\images\augmented_train"
OUT_LABELS_DIR = r"data\labels\augmented_train"

TARGET_TOTAL_IMAGES = 650

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# Image-only augmentation pipeline (Guaranteed to execute)
img_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT, p=0.6),
    A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=0.7),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
])

# Bounding box augmentation pipeline
bbox_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT, p=0.6),
    A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=0.7),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.1))

def read_yolo_labels(label_path):
    boxes, cat_ids = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cat_ids.append(int(parts[0]))
                    boxes.append([float(x) for x in parts[1:]])
    return boxes, cat_ids

def save_yolo_labels(label_path, boxes, cat_ids):
    with open(label_path, 'w') as f:
        for cat_id, box in zip(cat_ids, boxes):
            f.write(f"{cat_id} {' '.join([f'{x:.6f}' for x in box])}\n")

# Detect all seed images
image_paths = [f for f in glob.glob(os.path.join(IMAGES_DIR, "*.*")) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp'))]

if not image_paths:
    print(f"No seed images found in '{IMAGES_DIR}'.")
else:
    per_image_runs = (TARGET_TOTAL_IMAGES // len(image_paths)) + 1
    generated_count = 0

    print(f"Generating ~{TARGET_TOTAL_IMAGES} synthetic sonar images...")

    for img_path in tqdm(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        bboxes, category_ids = read_yolo_labels(label_path)
        
        for i in range(per_image_runs):
            if generated_count >= TARGET_TOTAL_IMAGES:
                break
                
            out_name = f"sonar_aug_{generated_count+1:04d}"
            
            aug_bboxes, aug_cats = [], []
            if bboxes:
                try:
                    augmented = bbox_transform(image=image, bboxes=bboxes, category_ids=category_ids)
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_cats = augmented['category_ids']
                except Exception:
                    aug_img = img_transform(image=image)['image']
            else:
                aug_img = img_transform(image=image)['image']
            
            cv2.imwrite(os.path.join(OUT_IMAGES_DIR, f"{out_name}.png"), aug_img)
            save_yolo_labels(os.path.join(OUT_LABELS_DIR, f"{out_name}.txt"), aug_bboxes, aug_cats)
            generated_count += 1

    print(f"Success! Generated {generated_count} synthetic image-label pairs in '{OUT_IMAGES_DIR}'.")
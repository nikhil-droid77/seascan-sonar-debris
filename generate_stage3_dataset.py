import os
import glob
import random
import math
import cv2
import numpy as np
from tqdm import tqdm

SEED_DIR = r"data\images\train"
OUT_TRAIN_IMG = r"data\images\stage3_train"
OUT_TRAIN_LBL = r"data\labels\stage3_train"
OUT_VAL_IMG = r"data\images\stage3_val"
OUT_VAL_LBL = r"data\labels\stage3_val"

for d in [OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL]:
    os.makedirs(d, exist_ok=True)

# Exact 11 classes from SeaScan Stage 2 PDF
CLASSES = [
    "Standing-person", "Propeller", "Can", "Tire", "Shampoo-bottle",
    "Drink-carton", "Chain", "Valve", "Hook", "Mine", "Bottle"
]

seed_paths = [f for f in glob.glob(os.path.join(SEED_DIR, "*.*")) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
if not seed_paths:
    raise FileNotFoundError(f"No seed images found in {SEED_DIR}")

def blend_natural_debris(img, cls_id, cx, cy, w, h):
    """Subtly blends textured acoustic echo + shadow matching the side-scan sonar seabed."""
    H, W = img.shape[:2]
    x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
    x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
    bw, bh = max(16, x2 - x1), max(16, y2 - y1)
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2

    # Sample local seafloor color so reflections match gold/sepia or grayscale sonar
    patch = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    local_mean = np.mean(patch, axis=(0, 1)) if patch.size > 0 else np.array([70, 110, 140])
    echo_col = tuple(int(min(255, c * 1.85 + 30)) for c in local_mean)
    mid_col = tuple(int(min(255, c * 1.55 + 20)) for c in local_mean)
    shadow_col = tuple(int(max(4, c * 0.20)) for c in local_mean)

    # 1. Cast soft directional acoustic shadow (like natural rocks/pipes in side-scan sonar)
    sh_layer = np.zeros_like(img, dtype=np.uint8)
    sh_dx = max(8, int(bw * 0.45))
    sh_dy = max(4, int(bh * 0.25))
    pts = np.array([
        [mx - bw//4, my - bh//4],
        [mx + sh_dx, my - bh//4 + sh_dy],
        [mx + sh_dx, my + bh//3 + sh_dy],
        [mx - bw//4, my + bh//3]
    ], np.int32)
    cv2.fillPoly(sh_layer, [pts], shadow_col)
    sh_layer = cv2.GaussianBlur(sh_layer, (13, 13), 0)
    sh_mask = (sh_layer.sum(axis=2) > 8).astype(np.float32)[:, :, None] * 0.50
    img[:] = np.clip(img * (1 - sh_mask) + sh_layer * sh_mask, 0, 255).astype(np.uint8)

    # 2. Build organic acoustic return mask per class
    obj_layer = np.zeros_like(img, dtype=np.uint8)
    r = min(bw, bh) // 2

    if cls_id == 0:    # Standing-person: elongated acoustic return
        cv2.ellipse(obj_layer, (mx, my), (max(4, bw // 5), max(8, bh // 2)), 12, 0, 360, echo_col, -1)
    elif cls_id == 1:  # Propeller: 3-blade hub signature
        for ang in [15, 135, 255]:
            ex = int(mx + r * math.cos(math.radians(ang)))
            ey = int(my + r * math.sin(math.radians(ang)))
            cv2.line(obj_layer, (mx, my), (ex, ey), echo_col, max(3, r // 4))
    elif cls_id == 2:  # Can: small metallic cylinder return
        cv2.ellipse(obj_layer, (mx, my), (bw // 3, bh // 4), -15, 0, 360, echo_col, -1)
    elif cls_id == 3:  # Tire: curved crescent/ring return
        cv2.ellipse(obj_layer, (mx, my), (r, int(r * 0.85)), 0, 20, 310, echo_col, max(3, r // 3))
    elif cls_id == 4:  # Shampoo-bottle: tapered crescent/bottle return
        cv2.ellipse(obj_layer, (mx, my), (int(r * 0.8), int(r * 0.6)), 10, 0, 220, echo_col, max(3, r // 3))
    elif cls_id == 5:  # Drink-carton: angled rectangular return
        cv2.rectangle(obj_layer, (x1 + bw//4, y1 + bh//4), (x2 - bw//4, y2 - bh//4), mid_col, -1)
    elif cls_id == 6:  # Chain: broken linear link reflections
        for s in range(4):
            lx = int(x1 + (s + 0.5) * bw / 4)
            ly = int(y1 + (s + 0.5) * bh / 4)
            cv2.circle(obj_layer, (lx, ly), max(3, r // 4), echo_col, -1)
    elif cls_id == 7:  # Valve: spoked circular return
        cv2.circle(obj_layer, (mx, my), int(r * 0.85), echo_col, 2)
        cv2.line(obj_layer, (mx - r//2, my), (mx + r//2, my), echo_col, 2)
        cv2.line(obj_layer, (mx, my - r//2), (mx, my + r//2), echo_col, 2)
    elif cls_id == 8:  # Hook: curved J-arc reflection
        cv2.ellipse(obj_layer, (mx, my), (r // 2, int(r * 0.85)), 20, 0, 210, echo_col, max(3, r // 3))
    elif cls_id == 9:  # Mine: dense spherical acoustic highlight
        cv2.circle(obj_layer, (mx, my), int(r * 0.65), echo_col, -1)
        for ang in range(0, 360, 60):
            ex = int(mx + int(r * 0.9) * math.cos(math.radians(ang)))
            ey = int(my + int(r * 0.9) * math.sin(math.radians(ang)))
            cv2.line(obj_layer, (mx, my), (ex, ey), mid_col, 2)
    elif cls_id == 10: # Bottle: narrow neck + wider body reflection
        cv2.ellipse(obj_layer, (mx, my + bh//8), (bw // 4, bh // 3), 0, 0, 360, echo_col, -1)
        cv2.line(obj_layer, (mx, my - bh//3), (mx, my), echo_col, max(2, bw // 8))

    # Add sonar speckle noise & soft Gaussian blur so it blends seamlessly into seafloor
    speckle = np.random.normal(1.0, 0.22, obj_layer.shape).astype(np.float32)
    obj_layer = np.clip(obj_layer.astype(np.float32) * speckle, 0, 255).astype(np.uint8)
    obj_layer = cv2.GaussianBlur(obj_layer, (7, 7), 0)

    alpha_mask = (obj_layer.sum(axis=2) > 12).astype(np.float32)[:, :, None] * 0.48
    img[:] = np.clip(img * (1 - alpha_mask) + obj_layer * alpha_mask, 0, 255).astype(np.uint8)

def generate_split(num_images, img_out, lbl_out, prefix):
    class_pool = (list(range(len(CLASSES))) * ((num_images * 3 // len(CLASSES)) + 2))
    random.shuffle(class_pool)
    pool_idx = 0

    for idx in tqdm(range(num_images), desc=f"Generating {prefix}"):
        img = cv2.imread(random.choice(seed_paths)).copy()
        img = cv2.resize(img, (640, 640))

        # Natural side-scan sonar augmentations (flip, brightness/contrast)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.3:
            img = cv2.flip(img, 0)
        gain = random.uniform(0.88, 1.15)
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # Place 3 targets in upper & lower sonar swaths (avoiding the dark middle nadir line)
        swath_slots = [(0.24, 0.24), (0.74, 0.26), (0.26, 0.76), (0.75, 0.75)]
        chosen_slots = random.sample(swath_slots, 3)

        out_name = f"{prefix}_{idx+1:04d}"
        with open(os.path.join(lbl_out, f"{out_name}.txt"), "w") as f:
            for (gx, gy) in chosen_slots:
                cls_id = class_pool[pool_idx]
                pool_idx += 1
                cx = round(gx + random.uniform(-0.07, 0.07), 4)
                cy = round(gy + random.uniform(-0.06, 0.06), 4)
                w = round(random.uniform(0.11, 0.17), 4)
                h = round(random.uniform(0.11, 0.17), 4)

                blend_natural_debris(img, cls_id, cx, cy, w, h)
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        cv2.imwrite(os.path.join(img_out, f"{out_name}.jpg"), img)

generate_split(600, OUT_TRAIN_IMG, OUT_TRAIN_LBL, "sss_train")
generate_split(120, OUT_VAL_IMG, OUT_VAL_LBL, "sss_val")
print("Success! 600 natural side-scan sonar training images and 120 validation images generated.")
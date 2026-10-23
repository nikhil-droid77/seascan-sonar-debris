r"""Stage 2 - preprocessing for side-scan sonar images.

Steps: grayscale -> across-track gain correction -> speckle denoising -> CLAHE contrast.
Run `python -m src.preprocess path\to\image.jpg` to save a before/after picture for your report.
"""
import sys
import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def gain_correct(gray, smooth=51, max_gain=3.0):
    """Even out the brightness falloff across the swath (sonar is brighter near the centre)."""
    col = gray.astype(np.float32).mean(axis=0)
    k = np.ones(smooth, np.float32) / smooth
    col_s = np.convolve(np.pad(col, smooth // 2, mode="edge"), k, mode="valid")[: gray.shape[1]]
    gain = np.clip(col_s.mean() / (col_s + 1e-3), 1 / max_gain, max_gain)
    return np.clip(gray.astype(np.float32) * gain[None, :], 0, 255).astype(np.uint8)


def denoise(gray, method="median", size=5):
    """Reduce speckle noise."""
    if method == "median":
        return cv2.medianBlur(gray, size | 1)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, 7, 40, 7)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return gray


def normalize(gray, clip=2.0):
    """CLAHE local contrast enhancement."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def preprocess(img, method="median"):
    """Full chain. Returns a 3-channel uint8 image (what YOLO expects)."""
    g = normalize(denoise(gain_correct(to_gray(img)), method))
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    out = preprocess(img)
    side = np.hstack([img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), out])
    cv2.imwrite("preprocess_before_after.png", side)
    print("Saved preprocess_before_after.png (left = original, right = preprocessed)")

# SeaScan: Underwater Marine Debris & Anomaly Detection from Side-Scan Sonar

AI pipeline that finds objects (shipwrecks, aircraft, mines, pipelines, fishing nets) in side-scan sonar (SSS) images and
produces geotagged reports. Built for **SIH26057** (Ministry of Earth Sciences) and the course **CSET 214: Data Analysis Using Python**.

**Team:** Rahul Raj, Nikhil Kumar &nbsp;|&nbsp; **Status:** Stage 2 prototype (baseline)

## Pipeline

```
Sonar image -> Preprocessing -> YOLOv8 detector -> Noise filtering -> Geotagging -> CSV / JSON report
```

| Module | File | What it does |
|---|---|---|
| Preprocessing | `src/preprocess.py` | gain correction, speckle denoising (median), CLAHE contrast |
| Detection | `src/pipeline.py` | YOLOv8 inference on a sonar image |
| Noise filtering | `src/filtering.py` | confidence threshold, size filter, non-maximum suppression |
| Geotagging | `src/geotag.py` | pixel -> latitude/longitude from survey metadata |
| Reports | `src/pipeline.py` | `report.csv`, `report.json`, annotated image |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pytest -q
```

## Train a baseline

```bash
yolo detect train model=yolov8n.pt data=data/Sonar_Dataset/data.yaml epochs=5 imgsz=416 fraction=0.25 batch=16 workers=0 device=cpu name=baseline
```

## Run the pipeline

```bash
python -m src.pipeline --image <image.jpg> --weights runs/detect/baseline/weights/best.pt --lat 15.30 --lon 73.80 --heading 45 --swath 100
```

Output goes to `outputs/` (`report.csv`, `report.json`, `annotated.png`).

## Results (baseline)

| Model | Data used | Epochs | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8n | 25% of train split, 416 px, CPU | 5 | _fill in_ | _fill in_ |

_Baseline only. Full training (YOLOv8s, full data, GPU) is planned for Stage 3._

## Dataset and credits

Trained on **Marine SSS Detection V4** (5 classes: Aircraft, Mine, Shipwreck, Pipeline, Fishing_Net; 4,285 train / 535 val / 537 test),
built from the Kaggle *Side Scan Sonar Object Detection Challenge* data and the **DRISHTI** side-scan sonar dataset (CC-BY-SA-4.0).
DRISHTI upstream sources: SubPipe (CC-BY-4.0), AI4Shipwrecks (CC-BY-4.0), Dae Hyeok Lee / Roboflow Universe "Side Scan Sonar (Ship, Plane)" (CC-BY-4.0),
MILCO mine contacts (CC-BY-SA-4.0), and a procedural acoustic generator. The dataset is **not** stored in this repository.

## Limitations (please read)

- The `Fishing_Net` class comes from **synthetic** images, so scores on it do not show real-world ghost-net performance.
- The dataset has **no GPS data**; coordinates in demo reports come from user-supplied example metadata.
- Stage 2 baseline uses a small model on a subset of the data; it is a working prototype, not a final result.

## Roadmap

- **Stage 2 (12-16 Oct 2026):** preprocessing, YOLOv8 baseline, noise filtering, geotagging (this repo)
- **Stage 3 (16-20 Nov 2026):** full training, per-class evaluation, Streamlit dashboard, ONNX export

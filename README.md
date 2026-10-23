# \# SeaScan: Underwater Marine Debris \& Anomaly Detection from Side-Scan Sonar

# 

# AI pipeline that finds marine debris and objects (bottles, cans, tires, chains, mines and others) in sonar images and produces geotagged reports. Built for \*\*SIH26057\*\* (Ministry of Earth Sciences) and the course \*\*CSET 214: Data Analysis Using Python\*\*.

# 

# \*\*Team:\*\* Rahul Raj, Nikhil Kumar \&nbsp;|\&nbsp; \*\*Status:\*\* Stage 2 prototype (baseline)

# 

# \## Pipeline

# 

# ```

# Sonar image -> Preprocessing -> YOLOv8 detector -> Noise filtering -> Geotagging -> CSV / JSON report

# ```

# 

# | Module | File | What it does |

# |---|---|---|

# | Preprocessing | `src/preprocess.py` | gain correction, speckle denoising (median), CLAHE contrast |

# | Detection | `src/pipeline.py` | YOLOv8 inference on a sonar image |

# | Noise filtering | `src/filtering.py` | confidence threshold, size filter, non-maximum suppression |

# | Geotagging | `src/geotag.py` | pixel -> latitude/longitude from survey metadata |

# | Reports | `src/pipeline.py` | `report.csv`, `report.json`, annotated image |

# 

# \## Setup

# 

# ```

# python -m venv .venv

# .venv\\Scripts\\activate

# pip install -r requirements.txt

# pytest -q

# ```

# 

# Put the dataset in `data/Sonar\_Dataset/` (folders `images/` and `labels/`, plus a `data.yaml` whose `path:` matches your computer).

# 

# \## Train a baseline

# 

# ```

# yolo detect train model=yolov8n.pt data=data/Sonar\_Dataset/data.yaml epochs=5 imgsz=416 fraction=0.25 batch=16 workers=0 device=cpu name=baseline

# ```

# 

# \## Run the pipeline

# 

# ```

# python -m src.pipeline --image <image.jpg> --weights runs/detect/baseline/weights/best.pt --conf 0.25 --heading 45

# ```

# 

# Output goes to `outputs/` (`report.csv`, `report.json`, `annotated.png`).

# 

# \## Results (baseline, validation set)

# 

# | Model | Data used | Epochs | mAP50 | mAP50-95 |

# |---|---|---|---|---|

# | YOLOv8n | 407 train images (25% of train split), 202 val images, 416 px, CPU | 5 | 0.515 | 0.273 |

# 

# \- Inference speed: about 20 ms per image on CPU (project target: under 100 ms).

# \- Training took about 6 minutes. The score was still rising at epoch 5, so longer training should help.

# \- Best classes: Standing-person, Propeller, Can, Tire. Weakest: Bottle (0.00 mAP50) and Mine (0.22, with only 6 validation examples).

# \- The target of 80% mAP50 is \*\*not yet reached\*\*. Full training is planned for Stage 3.

# 

# \## Dataset and credits

# 

# The class list matches the Marine Debris Forward-Looking Sonar (Watertank) dataset by M. Valdenegro-Toro (Heriot-Watt University), recorded with an ARIS Explorer 3000 sonar in a water tank. The Mine and Standing-person classes are not part of that dataset, so the data used here appears to be a modified or combined version.

# Exact download source: TO BE CONFIRMED. License: TO BE CONFIRMED.

# 11 classes: Bottle, Can, Chain, Drink-carton, Hook, Propeller, Shampoo-bottle, Standing-person, Tire, Valve, Mine.

# The dataset is \*\*not\*\* stored in this repository.

# \## Limitations (please read)

# 

# \- Baseline only: small model, 25% of the training data, 5 epochs. It is a working prototype, not a final result.

# \- The dataset has \*\*no GPS data\*\*. Coordinates in demo reports come from user-supplied example metadata.

# \- Geotagging assumes a side-scan waterfall image layout, so it needs adapting for other sonar types.

# \- Some classes have very few validation examples, so their scores are unreliable.

\- The debris classes appear to come from forward-looking sonar images recorded in a water tank, not from side-scan sonar at sea, so results may not transfer to real side-scan surveys. Side-scan data is planned for Stage 3.

# 

# \## Roadmap

# 

# \- \*\*Stage 2 (12-16 Oct 2026):\*\* preprocessing, YOLOv8 baseline, noise filtering, geotagging (this repo)

# \- \*\*Stage 3 (16-20 Nov 2026):\*\* full training (YOLOv8s, GPU), per-class evaluation, Streamlit dashboard, ONNX export


# Chapter 5 — Machine Learning Algorithms for Fire Detection

> **Project:** Machine Learning Technique for Enhanced Situational Awareness With Fire Fighter Drones  
> **Institution:** Faculty of Artificial Intelligence, Kafr El-Sheikh University  
> **Academic Year:** 2023 – 2024  
> **Supervisor:** Dr. Zeinab Hassan Ali Hassan

---

## Overview

This chapter covers the complete machine learning pipeline developed to enable real-time fire detection from a drone-mounted camera. The system leverages **YOLOv8** — a state-of-the-art object detection architecture — to identify fire and smoke in live video feeds, and computes the **distance to the detected fire** using triangle similarity geometry. The pipeline spans dataset construction, annotation, augmentation, model training, evaluation, and on-device inference integration.

---

## Objectives

- Build a fire-specific image dataset through manual collection and automated annotation.
- Train a high-accuracy, real-time fire detection model using YOLOv8.
- Integrate the model with the drone's onboard Raspberry Pi camera.
- Estimate the physical distance between the drone camera and the detected fire.
- Validate the model's performance through testing and iterative refinement.

---

## System Description

The detection subsystem operates as a perception layer within the broader drone architecture. The Raspberry Pi 4 captures live video, passes frames through the trained YOLOv8 model, and returns bounding boxes with class confidence scores. When fire is detected, the system uses the bounding box dimensions and known camera parameters to estimate distance in real time.

```
Live Camera Feed
      │
      ▼
 Frame Capture (Raspberry Pi Camera V1.3 — 5MP)
      │
      ▼
 YOLOv8 Inference (Ultralytics)
      │
      ├── Bounding Box Coordinates
      ├── Class Label: "Fire" / "Smoke"
      └── Confidence Score
            │
            ▼
     Distance Estimation (Triangle Similarity)
            │
            ▼
     Alert / Telemetry Output → Ground Station
```

---

## Data Pipeline

### 5.1 Dataset Construction

#### Collection
- Approximately **3,000 fire images** gathered from open sources (Unsplash, iStock).
- Images cover varied environments: indoor fires, wildfires, candlelight, vehicle fires.

#### Annotation
- Primary tool: **LabelImg** — manual bounding box drawing per image.
- Annotation formats supported: **YOLO** and **PascalVOC**.
- Each bounding box labeled with the class `fire`.

#### Data Augmentation & Expansion via Roboflow
Initial training accuracy was insufficient with the manually collected set. The team integrated **Roboflow** to address this:

| Roboflow Capability | Purpose |
|---|---|
| Access to 500+ community datasets | Expanded training diversity |
| Automatic annotation | Eliminated manual labeling bottleneck |
| Built-in augmentation pipeline | Rotation, flip, zoom, shear, brightness shift |
| Export to YOLO format | Direct compatibility with YOLOv8 training |

Five to six Roboflow datasets were merged with the original collection, significantly improving model accuracy and generalization.

---

## Model — YOLOv8

### Why YOLO over Other Pre-trained Models?

| Criterion | Pre-trained Models (general) | YOLO |
|---|---|---|
| Purpose | Classification / Feature extraction | Real-time object detection |
| Speed | Varies | Optimized for high FPS |
| Bounding box output | Not always native | Core output |
| Deployment suitability | Requires adaptation | Production-ready |

YOLO (You Only Look Once) frames detection as a **single regression problem** — one forward pass of the CNN simultaneously predicts bounding box coordinates and class probabilities across the entire image, making it inherently faster than region-proposal approaches such as Faster R-CNN.

### YOLO Architecture

The backbone is a deep **Convolutional Neural Network (CNN)** with 24 convolutional layers:

- The first 20 layers are **pre-trained on ImageNet** (image classification).
- Additional convolutional and fully connected layers are appended for detection.
- The final fully connected layer predicts:
  - Bounding box coordinates `(x, y, w, h)`
  - Objectness confidence score
  - Class probability vector

The image is divided into an **S × S grid**. Each grid cell predicts B bounding boxes and C class probabilities. Final detections are produced after Non-Maximum Suppression (NMS).

### Why YOLOv8 Specifically?

YOLOv8, developed by **Ultralytics**, was chosen for the following reasons:

- **Accuracy:** Benchmarked on Microsoft COCO and Roboflow 100 datasets.
- **Developer Experience:** Clean CLI interface and well-structured Python API.
- **Community Support:** Large active community for troubleshooting and guidance.
- **Multi-task Capability:** Supports detection, segmentation, classification, and pose estimation in a unified framework.
- **Long-term Support:** Ultralytics actively maintains and updates the model.

---

## Development Environment

### IDEs Used

| Tool | Role |
|---|---|
| **Jupyter Notebook** | Local prototyping, visualization, iterative testing |
| **Google Colab** | Cloud-based GPU training (Tesla T4 via CUDA) |

### Programming Language

**Python** was the primary language due to:
- Native support for ML frameworks (PyTorch, OpenCV, Ultralytics)
- Object-oriented design enabling modular pipeline construction
- Cross-platform compatibility
- Extensive open-source ecosystem

### Key Libraries

| Library | Purpose |
|---|---|
| **Ultralytics** | YOLOv8 model loading, training, inference |
| **OpenCV** | Frame capture, image preprocessing, bounding box rendering |
| **Roboflow SDK** | Dataset loading and management |
| **PyTorch** | Deep learning backend for YOLOv8 |
| **Matplotlib** | Training curve visualization |

---

## Technologies Used

| Technology | Version / Details |
|---|---|
| YOLOv8 | Ultralytics YOLOv8 (latest at time of development) |
| Python | 3.x |
| Google Colab | GPU: Tesla T4, CUDA 11.x |
| Roboflow | Fire Detection dataset (multiple merged sources) |
| OpenCV | 4.x |
| LabelImg | Manual annotation |
| Raspberry Pi Camera | Module V1.3 — 5MP, focal length 3.60mm |

---

## Implementation Details

### Step 1 — Environment Setup (Google Colab)
```python
pip install ultralytics
pip install roboflow
```

### Step 2 — Dataset Loading from Roboflow
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("fire-detection-vdmz")
dataset = project.version(1).download("yolov8")
```

### Step 3 — Model Training
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8 nano backbone

model.train(
    data=dataset.location + "/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    optimizer="Adam",
    patience=100,
    cache=True,
    save_period=1
)
```
- **Optimizer:** Adam
- **Epochs:** 30
- **Image Size:** 640 × 640
- **Pretrained Weights:** Transferred from ImageNet-trained backbone

### Step 4 — Inference on Test Images
```python
from ultralytics import YOLO
from IPython.display import Image, display

model = YOLO("/runs/detect/train/weights/best.pt")
results = model.predict(source="test_image.jpg", imgsz=640)

for r in results:
    boxes = r.boxes     # Bounding box outputs
    masks = r.masks     # Segment masks (if applicable)
    probs = r.probs     # Classification probabilities
```

### Step 5 — Batch Validation
```python
import glob
from IPython.display import Image, display

for image_path in glob.glob("/runs/detect/predict/*.jpg")[:1]:
    display(Image(filename=image_path, width=600))
    print("\n")
```

---

## Distance Estimation

To provide actionable spatial data to firefighters, the system estimates the **physical distance between the drone camera and the detected fire** using the **Triangle Similarity Method**.

### Parameters

| Symbol | Description | Value |
|---|---|---|
| `f` | Focal length of camera | 3.60 mm (constant — Raspberry Pi Camera V1.3) |
| `r` | Radius of reference marker in image plane | 10 cm (constant) |
| `R` | Apparent size of object in image (computed per frame) | Varies |
| `d` | Distance from camera to fire object | **Computed output** |

### Governing Equation

The camera is modeled as generating a one-to-one projective relationship between object and image planes:

```math
f / d = r / R
```

Solving for distance:

```math
d = (f × R) / r
```

Where `R` is calculated dynamically from the bounding box dimensions during live video streaming. The focal length `f` and reference radius `r` remain constant for the Raspberry Pi Camera V1.3 module.

---

## Results

| Metric | Outcome |
|---|---|
| Fire detection confidence (test sample) | **0.90** (90%) on sample candle image |
| Fire detection confidence (vehicle fire) | **0.74** (74%) |
| Training convergence | Achieved within 30 epochs |
| Inference environment | Google Colab (Tesla T4 GPU) + Local Jupyter Notebook |
| Model output | Bounding box with class label "Fire" and confidence score |
| Validation | Visual inspection of prediction overlays on held-out images |

Training results including loss curves, precision/recall metrics, and confusion matrix were exported and visualized via `results.png` and `val_batch_pred.jpg`.

---

## Challenges & Solutions

| Problem | Cause | Solution |
|---|---|---|
| Low initial detection accuracy | Small and homogeneous dataset (~3,000 images) | Integrated Roboflow datasets (5–6 additional sources) |
| Manual annotation was time-consuming | Large volume of images requiring bounding boxes | Adopted Roboflow's automatic annotation feature |
| GPU memory constraints | YOLOv8 training on large batches | Used Google Colab with Tesla T4 GPU |
| Inference speed on Raspberry Pi | Limited onboard compute | Optimized model size; considered tiny-YOLO variant for deployment |
| Distance accuracy at varying angles | Camera projection assumptions | Applied Triangle Similarity with fixed reference constants |

---

## Future Improvements

- **Thermal Camera Integration:** Extend detection to infrared feeds for low-visibility and night-time scenarios.
- **Multi-class Detection:** Add `smoke`, `person`, and `vehicle` classes to enrich situational awareness.
- **Model Quantization:** Apply INT8 quantization or TensorRT optimization for faster Raspberry Pi inference.
- **Tracking Module:** Integrate YOLOv8's built-in multi-object tracker to maintain fire identity across frames.
- **Aerial Dataset Collection:** Build a drone-perspective fire dataset for improved aerial detection accuracy.
- **Swarm Coordination:** Feed detection outputs into a multi-drone scheduling algorithm for optimized coverage.

---

## Conclusion

Chapter 5 establishes the complete perception backbone of the firefighting drone system. Through a carefully constructed data pipeline — combining manual collection, LabelImg annotation, Roboflow augmentation, and YOLOv8 fine-tuning — the system achieves reliable real-time fire detection at high confidence scores. The addition of triangle similarity-based distance estimation transforms raw detections into actionable spatial intelligence for emergency responders. This chapter demonstrates a production-grade ML pipeline adapted for resource-constrained, safety-critical autonomous systems.

---

> *Graduation Project — Faculty of Artificial Intelligence, Kafr El-Sheikh University, 2023–2024*

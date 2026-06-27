# Vehicle Detection and Counting System

A real-time vehicle detection and counting system built with Python and deep learning. The system supports three detection models — YOLO11, YOLOv8, and RT-DETR — and uses ByteTrack for multi-object tracking with a crossing-line vehicle counter.

---

## Features

- Real-time vehicle detection using YOLO11, YOLOv8, and RT-DETR
- Multi-object tracking with ByteTrack
- Crossing-line vehicle counter for cars, bikes, and others
- Sequential track ID display
- FPS and live statistics overlay
- Video output saving
- Model benchmarking tool

---

## Project Structure

```
VEHICLE-DETECTION/
├── assets/
│   └── sample_video.mp4
├── configs/
│   └── settings.yaml
├── models/
│   ├── yolov11/
│   ├── yolov8/
│   └── RT-DETR/
├── results/
├── src/
│   ├── app.py
│   ├── benchmark.py
│   ├── detector.py
│   ├── tracker.py
│   └── utils.py
├── yolo11n.pt
├── yolov8n.pt
├── rtdetr-l.pt
├── requirements.txt
└── README.md
```

---

## Models

| Model | Type | Size | Best For |
|-------|------|------|----------|
| YOLO11n | One-stage detector | 5.4 MB | Real-time, best throughput |
| YOLOv8n | One-stage detector | 6.2 MB | Balanced speed and accuracy |
| RT-DETR-L | Transformer-based | 63 MB | High accuracy, offline use |

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Varunbansode0209/VEHICLE-DETECTION.git
cd VEHICLE-DETECTION
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your video**

Place your video file inside the `assets/` folder and update the path in `configs/settings.yaml`:
```yaml
video:
  source: "assets/your_video.mp4"
```

---

## Usage

**Run with YOLO11 (default)**
```bash
python src/app.py
```

**Run with a specific model**
```bash
python src/app.py --model yolov8
python src/app.py --model rtdetr
```

**Run on a custom video**
```bash
python src/app.py --model yolo11 --source assets/sample2.mp4
```

**Run `sample2.mp4` with all three models**
```bash
# YOLO11 (fastest, best real-time throughput)
python src/app.py --model yolo11 --source assets/sample2.mp4

# YOLOv8 (balanced speed and accuracy)
python src/app.py --model yolov8 --source assets/sample2.mp4

# RT-DETR (highest accuracy, transformer-based)
python src/app.py --model rtdetr --source assets/sample2.mp4
```

**Disable display or saving**
```bash
python src/app.py --no-display
python src/app.py --no-save
```

**Run benchmark**
```bash
python src/benchmark.py
```

---

## Configuration

All settings are managed in `configs/settings.yaml`:

```yaml
detection:
  classes: [1, 2, 3, 5, 7]   # bicycle, car, motorcycle, bus, truck
  confidence: 0.45
  iou_threshold: 0.50
  input_size: 640

counting:
  position: 0.55              # counting line position (55% down the frame)
  flash_frames: 8
```

---

## Detection Classes

| Class | Label | Box Colour |
|-------|-------|------------|
| Car | Car | Orange |
| Motorcycle | Bike | Green |
| Bicycle / Bus / Truck | Others | Grey |

---

## Benchmark Results

| Model | FPS | Avg Dets/Frame | Throughput (det/sec) | Verdict |
|-------|-----|----------------|----------------------|---------|
| YOLO11n | 25.4 | 7.03 | 178.6 | ✅ Winner |
| YOLOv8n | 25.4 | 4.53 | 115.1 | |
| RT-DETR-L | 1.9 | 8.78 | 16.7 | |

> Throughput = FPS × Avg Detections per Frame. YOLO11 delivers the highest number of vehicle detections per second, making it the best choice for real-time use.

---

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics
- OpenCV

See `requirements.txt` for full details.

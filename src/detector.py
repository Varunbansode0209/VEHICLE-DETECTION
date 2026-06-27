import cv2
import torch
import yaml
import time
from pathlib import Path
from ultralytics import YOLO


def load_config(path="configs/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class BaseDetector:
    VEHICLE_CLASSES = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, cfg):
        self.conf = cfg["detection"]["confidence"]
        self.iou  = cfg["detection"]["iou_threshold"]
        self.size = cfg["detection"]["input_size"]

    def detect(self, frame):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class YOLO11Detector(BaseDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
        model_path = cfg["models"]["yolo11"]
        self.model = YOLO(model_path)
        self.classes = cfg["detection"]["classes"]
        print(f"[YOLO11] Loaded {model_path}")

    def name(self):
        return "YOLO11n"

    def detect(self, frame):
        results = self.model(
            frame,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.size,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = self.VEHICLE_CLASSES.get(cls, "vehicle")
            detections.append((x1, y1, x2, y2, conf, cls, label))

        return detections


class YOLOv8Detector(BaseDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
        model_path = cfg["models"]["yolov8"]
        self.model = YOLO(model_path)
        self.classes = cfg["detection"]["classes"]
        print(f"[YOLOv8] Loaded {model_path}")

    def name(self):
        return "YOLOv8n"

    def detect(self, frame):
        results = self.model(
            frame,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.size,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = self.VEHICLE_CLASSES.get(cls, "vehicle")
            detections.append((x1, y1, x2, y2, conf, cls, label))

        return detections


class RTDETRDetector(BaseDetector):

    def __init__(self, cfg):
        super().__init__(cfg)
        model_path = cfg["models"]["rtdetr"]
        self.model = YOLO(model_path)
        self.classes = cfg["detection"]["classes"]
        print(f"[RT-DETR] Loaded {model_path}")

    def name(self):
        return "RT-DETR-L"

    def detect(self, frame):
        results = self.model(
            frame,
            classes=self.classes,
            conf=self.conf,
            imgsz=self.size,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = self.VEHICLE_CLASSES.get(cls, "vehicle")
            detections.append((x1, y1, x2, y2, conf, cls, label))

        return detections


def get_detector(model_name: str, cfg: dict) -> BaseDetector:
    options = {
        "yolo11": YOLO11Detector,
        "yolov8": YOLOv8Detector,
        "rtdetr": RTDETRDetector,
    }

    key = model_name.lower()
    if key not in options:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(options)}")
    return options[key](cfg)
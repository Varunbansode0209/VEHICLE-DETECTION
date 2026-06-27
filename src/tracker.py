import cv2
from ultralytics import YOLO


class VehicleTracker:

    VEHICLE_CLASSES = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path: str,
                 conf: float = 0.45,
                 iou: float = 0.5,
                 imgsz: int = 640):
        self.model          = YOLO(model_path)
        self.conf           = conf
        self.iou            = iou
        self.imgsz          = imgsz
        self.classes        = list(self.VEHICLE_CLASSES.keys())
        self._id_map        = {}
        self._next_local_id = 1

    def track(self, frame):
        results = self.model.track(
            frame,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf    = float(box.conf[0])
            cls     = int(box.cls[0])
            label   = self.VEHICLE_CLASSES.get(cls, "vehicle")
            raw_tid = int(box.id[0]) if box.id is not None else -1

            if raw_tid == -1:
                local_tid = -1
            else:
                if raw_tid not in self._id_map:
                    self._id_map[raw_tid] = self._next_local_id
                    self._next_local_id += 1
                local_tid = self._id_map[raw_tid]

            detections.append((x1, y1, x2, y2, local_tid, conf, cls, label))

        return detections


class CrossingCounter:

    def __init__(self, line_y):
        self.line_y           = line_y
        self.previous_centers = {}
        self.counted_ids      = set()
        self.car_count        = 0
        self.bike_count       = 0
        self.others_count     = 0
        self.total_count      = 0

    def update(self, detections):
        for det in detections:
            x1, y1, x2, y2, tid, conf, cls, label = det

            if tid == -1:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if tid in self.previous_centers:
                _, prev_y = self.previous_centers[tid]

                crossed = (
                    prev_y < self.line_y
                    and cy  >= self.line_y
                )

                skipped_over = (
                    prev_y < (self.line_y - 30)
                    and cy  > (self.line_y + 30)
                )

                if (crossed or skipped_over) and tid not in self.counted_ids:
                    self.counted_ids.add(tid)

                    if cls == 2:
                        self.car_count  += 1
                    elif cls == 3:
                        self.bike_count += 1
                    else:
                        self.others_count += 1

                    self.total_count += 1

            else:
                if cy > self.line_y and tid not in self.counted_ids:
                    self.counted_ids.add(tid)

                    if cls == 2:
                        self.car_count  += 1
                    elif cls == 3:
                        self.bike_count += 1
                    else:
                        self.others_count += 1

                    self.total_count += 1

            self.previous_centers[tid] = (cx, cy)

    def draw_line(self, frame):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.line_y), (w, self.line_y), (0, 0, 255), 3)
        return frame

    def get_counts(self):
        return {
            "cars":   self.car_count,
            "bikes":  self.bike_count,
            "others": self.others_count,
            "total":  self.total_count,
        }

    def reset(self):
        self.previous_centers.clear()
        self.counted_ids.clear()
        self.car_count    = 0
        self.bike_count   = 0
        self.others_count = 0
        self.total_count  = 0
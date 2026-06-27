import cv2
import time

CLASS_COLORS = {
    2: (0, 200, 255),
    3: (0, 255, 0),
}

OTHERS_COLOR   = (180, 180, 180)
DEFAULT_COLOR  = (200, 200, 200)
CENTROID_COLOR = (0, 0, 255)

OTHER_CLASSES = {1: "Bicycle", 5: "Bus", 7: "Truck"}


class FPSCounter:

    def __init__(self, smoothing=30):
        self._times     = []
        self._smoothing = smoothing

    def tick(self):
        self._times.append(time.time())
        if len(self._times) > self._smoothing:
            self._times.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed


def draw_detections(frame, detections, show_track_id=True, centroid_radius=5):
    for det in detections:
        if len(det) == 8:
            x1, y1, x2, y2, tid, conf, cls, label = det
        else:
            x1, y1, x2, y2, conf, cls, label = det
            tid = -1

        if cls in CLASS_COLORS:
            color = CLASS_COLORS[cls]
            short_map = {"car": "Car", "motorcycle": "Bike"}
            short = short_map.get(label, label.title())
        elif cls in OTHER_CLASSES:
            color = OTHERS_COLOR
            short = OTHER_CLASSES[cls]
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_track_id and tid >= 0:
            tag = f"{short} #{tid}  {conf:.2f}"
        else:
            tag = f"{short}  {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

        cv2.putText(
            frame, tag,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), centroid_radius, CENTROID_COLOR, -1)

    return frame


def draw_counting_line(frame, line_px, orientation="horizontal", flash=False):
    h, w = frame.shape[:2]
    color = (0, 255, 0) if flash else (0, 255, 255)

    if orientation == "horizontal":
        cv2.line(frame, (0, line_px), (w, line_px), color, 3, cv2.LINE_AA)
        cv2.putText(
            frame, "COUNT LINE",
            (w - 150, line_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA,
        )
    else:
        cv2.line(frame, (line_px, 0), (line_px, h), color, 3, cv2.LINE_AA)
        cv2.putText(
            frame, "COUNT LINE",
            (line_px + 5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA,
        )

    return frame


def draw_overlay(frame, fps, model_name, active_count,
                 cars_total=0, motorcycles_total=0,
                 others_total=0, crossed_total=0):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Model: {model_name}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w // 2 - 40, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 128), 1, cv2.LINE_AA)

    cv2.putText(frame, f"In Frame: {active_count}",
                (w - 170, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    col_w = w // 4
    entries = [
        (f"Cars: {cars_total}",         CLASS_COLORS[2]),
        (f"Bikes: {motorcycles_total}",  CLASS_COLORS[3]),
        (f"Others: {others_total}",      OTHERS_COLOR),
        (f"Total: {crossed_total}",      (0, 255, 255)),
    ]
    for i, (txt, col) in enumerate(entries):
        cv2.putText(frame, txt,
                    (8 + i * col_w, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    return frame
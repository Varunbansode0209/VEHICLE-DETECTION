import sys
import argparse
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detector import get_detector, load_config
from tracker import VehicleTracker, CrossingCounter
from utils import FPSCounter, draw_detections, draw_counting_line, draw_overlay


def run(model_name: str, source, save: bool, display: bool):
    cfg     = load_config()
    fps_ctr = FPSCounter()
    ccfg    = cfg["counting"]

    model_path_map = {
        "yolo11": cfg["models"]["yolo11"],
        "yolov8": cfg["models"]["yolov8"],
        "rtdetr": cfg["models"]["rtdetr"],
    }
    display_name_map = {
        "yolo11": "YOLO11n + ByteTrack",
        "yolov8": "YOLOv8n + ByteTrack",
        "rtdetr": "RT-DETR-L + ByteTrack",
    }

    runner       = VehicleTracker(
        model_path=model_path_map[model_name],
        conf=cfg["detection"]["confidence"],
        iou=cfg["detection"]["iou_threshold"],
        imgsz=cfg["detection"]["input_size"],
    )
    display_name = display_name_map[model_name]

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flash_ttl = 0
    line_y    = int(frame_h * ccfg["position"])
    counter   = CrossingCounter(line_y=line_y)

    writer = None
    if save:
        out_path = Path(cfg["video"]["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (frame_w, frame_h),
        )

    print(f"\n[INFO] Running {display_name} on '{source}'")
    print(f"[INFO] Counting line at y={counter.line_y}px")
    print("[INFO] Press 'Q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = runner.track(frame)
        fps_ctr.tick()

        prev_total = counter.total_count
        counter.update(detections)

        if counter.total_count > prev_total:
            flash_ttl = ccfg["flash_frames"]
        elif flash_ttl > 0:
            flash_ttl -= 1

        draw_detections(frame, detections, show_track_id=True)

        draw_counting_line(
            frame,
            line_px=counter.line_y,
            orientation="horizontal",
            flash=(flash_ttl > 0),
        )

        counts = counter.get_counts()

        draw_overlay(
            frame,
            fps=fps_ctr.fps,
            model_name=display_name,
            active_count=len(detections),
            cars_total=counts.get("cars", 0),
            motorcycles_total=counts.get("bikes", 0),
            others_total=counts.get("others", 0),
            crossed_total=counts.get("total", 0),
        )

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow(f"Vehicle Detection - {display_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Average FPS : {fps_ctr.fps:.1f}")

    c = counter.get_counts()
    print(f"[DONE] Cars crossed   : {c['cars']}")
    print(f"[DONE] Bikes crossed  : {c['bikes']}")
    print(f"[DONE] Others crossed : {c['others']}")
    print(f"[DONE] Total crossed  : {c['total']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Vehicle Detection")

    parser.add_argument(
        "--model",
        default="yolo11",
        choices=["yolo11", "yolov8", "rtdetr"],
    )
    parser.add_argument("--source",     default=None)
    parser.add_argument("--no-save",    dest="save",    action="store_false")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(save=True, display=True)

    args    = parser.parse_args()
    cfg     = load_config()
    source  = args.source or cfg["video"]["source"]
    save    = args.save    and cfg["video"]["save_output"]
    display = args.display and cfg["video"]["display"]

    run(args.model, source, save, display)
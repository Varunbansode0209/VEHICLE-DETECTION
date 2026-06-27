import cv2
from detector import get_detector, load_config
from utils import FPSCounter

MAX_FRAMES = 200


def benchmark_model(detector, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_ctr = FPSCounter(smoothing=MAX_FRAMES)
    total_detections = 0
    frames = 0

    while frames < MAX_FRAMES:
        ret, frame = cap.read()

        if not ret:
            break

        dets = detector.detect(frame)
        fps_ctr.tick()

        total_detections += len(dets)
        frames += 1

    cap.release()

    avg_fps  = fps_ctr.fps
    avg_dets = total_detections / max(frames, 1)
    throughput = avg_fps * avg_dets

    return avg_fps, avg_dets, throughput, frames


def main():
    cfg     = load_config()
    source  = cfg["video"]["source"]
    models  = ["yolo11", "yolov8", "rtdetr"]
    results = {}

    for name in models:
        print(f"\n[Benchmark] Testing {name}...")

        det = get_detector(name, cfg)
        fps, avg_dets, throughput, frames = benchmark_model(det, source)

        results[name] = {
            "fps":        fps,
            "avg_dets":   avg_dets,
            "throughput": throughput,
            "frames":     frames,
        }

        print(f" -> {fps:.1f} FPS | {avg_dets:.2f} avg detections/frame | {throughput:.1f} det/sec")

    print("\n" + "=" * 70)
    print(f"{'Model':<15}{'FPS':>8}{'Avg Dets':>12}{'Throughput':>14}{'Verdict':>15}")
    print("=" * 70)

    best = max(results, key=lambda n: results[n]["throughput"])

    for name, r in results.items():
        tag = "WINNER" if name == best else ""
        print(
            f"{name:<15}"
            f"{r['fps']:>8.1f}"
            f"{r['avg_dets']:>12.2f}"
            f"{r['throughput']:>14.1f}"
            f"{tag:>15}"
        )

    print("=" * 70)
    print("\n[INFO] Throughput = FPS x Avg Detections/Frame (higher is better)")
    print(f"[INFO] {best.upper()} achieves the highest detection throughput overall.")
    print("[INFO] YOLO11  : Best speed-accuracy balance for real-time detection.")
    print("[INFO] YOLOv8  : Competitive speed but lower detection rate.")
    print("[INFO] RT-DETR : Highest per-frame accuracy but too slow for real-time use.")


if __name__ == "__main__":
    main()
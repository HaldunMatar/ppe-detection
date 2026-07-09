"""
Run the trained PPE-detection model on a video file with PER-PERSON tracking.

Each person gets a tracked id. A violation (missing required PPE) is recorded
once per person per cooldown window (default 60s); if the same person is still
violating after the cooldown, a new record is saved. Every recorded violation
saves the full frame with a red box around that person, and a row in a CSV log.

Usage:
    python detect_video.py <input_video> [output_video] [--conf 0.4]
                           [--cooldown 60] [--require helmet,vest]

Outputs (next to the input video):
    <name>_detected.mp4          annotated video
    <name>_violations/           full-frame snapshots of each recorded violation
    <name>_violations/log.csv    one row per recorded violation
"""
import os
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
import sys
import csv
import argparse
import cv2
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Model-Deployment"))
from person_tracker import PersonViolationTracker, build_detections, draw_violation_box

MODEL_PATH = os.path.join(
    HERE,
    "Model-Training/Outputs/runs/detect/yolov8s_ppe_css_80_epochs/weights/best.pt",
)


def main():
    ap = argparse.ArgumentParser(description="Per-person PPE detection on a video file.")
    ap.add_argument("input", help="Path to the input video (e.g. clip.mp4)")
    ap.add_argument("output", nargs="?", default=None, help="Annotated output video path (optional)")
    ap.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default 0.4)")
    ap.add_argument("--cooldown", type=float, default=60,
                    help="Seconds before the SAME person is recorded again (default 60)")
    ap.add_argument("--require", default="helmet,vest",
                    help="Comma list of required PPE: helmet,vest,mask (default helmet,vest)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input video not found: {args.input}")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model weights not found: {MODEL_PATH}")

    out_path = args.output or (os.path.splitext(args.input)[0] + "_detected.mp4")
    viol_dir = os.path.splitext(args.input)[0] + "_violations"
    os.makedirs(viol_dir, exist_ok=True)
    csv_path = os.path.join(viol_dir, "log.csv")

    required = {k: (k in {p.strip() for p in args.require.split(",")}) for k in ("helmet", "vest", "mask")}
    tracker = PersonViolationTracker(cooldown=args.cooldown, required_ppe=required)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on device: {device}")
    print(f"Required PPE: {[k for k, v in required.items() if v]} | per-person cooldown: {args.cooldown}s")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["record_id", "person_id", "missing_ppe", "frame", "video_time_sec", "snapshot"])

    print(f"Processing {total or '?'} frames ({width}x{height} @ {fps:.0f} fps)...")
    i = 0
    records = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # video time is our clock, so cooldown is measured in video seconds
        video_time = i / fps
        result = model.track(frame, persist=True, conf=args.conf, device=device, verbose=False)[0]
        detections = build_detections(result)
        person_results = tracker.process(detections, now=video_time)

        annotated = result.plot()
        for pr in person_results:
            if pr["is_violation"]:
                draw_violation_box(annotated, pr)
            if pr["should_record"]:
                records += 1
                snap_path = os.path.join(viol_dir, f"{pr['record_id']}.jpg")
                cv2.imwrite(snap_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                csv_writer.writerow([
                    pr["record_id"], pr["track_id"], "|".join(pr["missing_ppe"]),
                    i, f"{video_time:.2f}", snap_path,
                ])

        writer.write(annotated)
        i += 1
        if i % 30 == 0 or i == total:
            print(f"  frame {i}/{total or '?'}  ({records} violation records so far)")

    cap.release()
    writer.release()
    csv_file.close()
    print(f"\nDone. {records} violation record(s).")
    print(f"  annotated video : {out_path}")
    print(f"  violation images: {viol_dir}/")
    print(f"  log             : {csv_path}")


if __name__ == "__main__":
    main()

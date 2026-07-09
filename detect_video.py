"""
Run the trained PPE-detection model on a video file and save an annotated copy.

Usage:
    python detect_video.py <input_video> [output_video] [--conf 0.4]

If no output path is given, saves alongside the input as "<name>_detected.mp4".
"""
import os
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
import sys
import argparse
import cv2
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    HERE,
    "Model-Training/Outputs/runs/detect/yolov8s_ppe_css_80_epochs/weights/best.pt",
)


def main():
    ap = argparse.ArgumentParser(description="Run PPE detection on a video file.")
    ap.add_argument("input", help="Path to the input video (e.g. clip.mp4)")
    ap.add_argument("output", nargs="?", default=None,
                    help="Path for the annotated output video (optional)")
    ap.add_argument("--conf", type=float, default=0.4,
                    help="Confidence threshold (default 0.4)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input video not found: {args.input}")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model weights not found: {MODEL_PATH}")

    out_path = args.output or (os.path.splitext(args.input)[0] + "_detected.mp4")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on device: {device}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print(f"Processing {total or '?'} frames ({width}x{height} @ {fps:.0f} fps)...")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, device=device, conf=args.conf, verbose=False)
        writer.write(res[0].plot())
        i += 1
        if i % 30 == 0 or i == total:
            print(f"  frame {i}/{total or '?'}")

    cap.release()
    writer.release()
    print(f"\nDone. Annotated video saved to:\n  {out_path}")


if __name__ == "__main__":
    main()

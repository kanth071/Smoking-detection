"""
Extract frames from a video so you can ANNOTATE them (in CVAT / Roboflow / labelImg)
and turn raw footage into a trainable dataset.

A detector cannot be trained on a video directly — it needs frames with
bounding-box labels. This is step 1 of that process.

Usage:
  python training/extract_frames.py --video clip.mp4 --out dataset/images/train --every 15
"""
import argparse
import os
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="dataset/images/train")
    ap.add_argument("--every", type=int, default=15, help="save 1 frame every N frames")
    ap.add_argument("--prefix", default="frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    i = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % args.every == 0:
            path = os.path.join(args.out, f"{args.prefix}_{saved:05d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        i += 1
    cap.release()
    print(f"Saved {saved} frames to {args.out}. Now annotate them (cigarette boxes) "
          f"and export in YOLO format.")


if __name__ == "__main__":
    main()

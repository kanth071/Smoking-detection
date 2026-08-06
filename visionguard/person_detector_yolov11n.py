# =============================================================================
# YOLOv11n Person Detection + ByteTrack Tracking
# Optimized for PERSON CLASS ONLY — filters out chairs, objects, noise
# =============================================================================
"""
KEY IMPROVEMENTS:
  • YOLOv11n: Faster nano model, still highly accurate for persons
  • CLASS FILTERING: Only detects class 0 (person) — rejects chairs, tables, etc.
  • CONFIDENCE BOOST: Raised to 0.50 to reduce false positives
  • IOU FILTERING: Strict IOU=0.50 to avoid overlapping detections
  • TRACKING: ByteTrack maintains persistent person IDs across frames

Requires:
  pip install -U ultralytics opencv-python

Run:
  python person_detector_yolov11n.py  # uses webcam (source 0)
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────
# CONFIG: Fine-tuned for person-only detection
# ─────────────────────────────────────────────────────────────────────
PERSON_MODEL = "yolo11n.pt"      # YOLOv11 Nano (pretrained COCO, auto-downloads)
SOURCE       = 0                 # 0 = webcam | "video.mp4" | "rtsp://..."
PERSON_CONF  = 0.50              # ↑ RAISED: 0.50 filters weak detections (chairs, objects)
IOU_THRESH   = 0.50              # ↓ STRICT: removes overlapping bboxes
IMG_SIZE     = 416               # YOLOv11n optimal: 416 (faster) or 640 (more accurate)
TRACKER      = "bytetrack.yaml"  # ByteTrack: persistent person IDs
SHOW_WINDOW  = True
PERSON_CLASS = 0                 # COCO class 0 = person (all others rejected)

# ─────────────────────────────────────────────────────────────────────
# HELPER: Filter non-person detections
# ─────────────────────────────────────────────────────────────────────
def filter_persons_only(results):
    """
    Extract ONLY person detections (class 0).
    Rejects chairs, tables, bottles, backpacks, etc.
    """
    if results is None or len(results) == 0:
        return None
    
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    
    # Filter: keep only class 0 (person)
    person_mask = r.boxes.cls == PERSON_CLASS
    
    if not person_mask.any():
        return None  # No persons detected
    
    # Create filtered result with only persons
    r.boxes.xyxy = r.boxes.xyxy[person_mask]
    r.boxes.conf = r.boxes.conf[person_mask]
    r.boxes.cls = r.boxes.cls[person_mask]
    if r.boxes.id is not None:
        r.boxes.id = r.boxes.id[person_mask]
    
    return r


# ─────────────────────────────────────────────────────────────────────
# MAIN: Person detection loop
# ─────────────────────────────────────────────────────────────────────
def run():
    print(f"Loading {PERSON_MODEL} (YOLOv11 Nano — COCO pretrained)...")
    print(f"  ✓ Confidence threshold: {PERSON_CONF} (high = fewer false positives)")
    print(f"  ✓ IOU threshold: {IOU_THRESH} (strict = cleaner boxes)")
    print(f"  ✓ Detecting PERSON CLASS ONLY (class 0) — chairs/objects rejected")
    print()
    
    model = YOLO(PERSON_MODEL)
    cap = cv2.VideoCapture(SOURCE)
    
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE!r}")
    
    frame_count = 0
    print("Running YOLOv11n person detection + ByteTrack. Press 'q' to quit.\n")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_count += 1
        
        # ─── DETECT & TRACK ───────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=PERSON_CONF,           # High confidence = fewer false positives
            iou=IOU_THRESH,             # Strict IOU = cleaner boxes
            imgsz=IMG_SIZE,
            verbose=False,
            device=0,                   # GPU (change to 'cpu' if no GPU)
        )
        
        # ─── FILTER: PERSONS ONLY ─────────────────────────────────
        r = filter_persons_only(results)
        
        person_count = 0
        if r is not None and r.boxes is not None:
            xyxys = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else range(len(xyxys)))
            
            person_count = len(xyxys)
            
            for (x1, y1, x2, y2), conf, tid in zip(xyxys, confs, ids):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # ─── DRAW BOUNDING BOX ────────────────────────────
                # Green box = detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ─── DRAW LABEL ───────────────────────────────────
                label = f"ID#{int(tid)} {conf:.2f}"
                cv2.putText(
                    frame, label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                
                # ─── OPTIONAL: Draw pose/center point ──────────────
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        
        # ─── DISPLAY STATS ─────────────────────────────────────────
        stats = f"Frame {frame_count} | Persons: {person_count}"
        cv2.putText(
            frame, stats,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # ─── SHOW ──────────────────────────────────────────────────
        if SHOW_WINDOW:
            cv2.imshow("YOLOv11n Person Detection (Class 0 Only)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("\n[INFO] Stopping detection...")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Processed {frame_count} frames")


if __name__ == "__main__":
    run()

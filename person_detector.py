"""
=============================================================================
 FILE 1 of 2 — person_detector.py
 Person detection + ByteTrack tracking using YOLO26s
=============================================================================
IMPORTANT: person detection needs NO training from you. YOLO26s is already
trained on the COCO dataset, which includes the "person" class. The pretrained
`yolo26s.pt` IS your "perfectly trained" person model — you just load and use it.
It auto-downloads on first run.

Requires Ultralytics:
    pip install -U ultralytics

Run:
    python person_detector.py            # uses webcam (source 0)
=============================================================================
"""
import cv2
from ultralytics import YOLO

# ── CONFIG ──────────────────────────────────────────────
PERSON_MODEL = "yolo26s.pt"     # pretrained COCO (person = class 0). Auto-downloads.
SOURCE       = 0                # 0 = webcam | "video.mp4" | "rtsp://..."
PERSON_CONF  = 0.40             # min confidence
IOU_THRESH   = 0.45
IMG_SIZE     = 640
TRACKER      = "bytetrack.yaml" # native ByteTrack -> persistent person IDs
SHOW_WINDOW  = True


def run():
    print(f"Loading {PERSON_MODEL} (YOLO26s, pretrained on COCO — no training needed)...")
    model = YOLO(PERSON_MODEL)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE!r}")

    print("Running person detection + ByteTrack. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # class 0 = person. YOLO26 is NMS-free, but the .track() API is unchanged.
        results = model.track(
            frame, persist=True, tracker=TRACKER, classes=[0],
            conf=PERSON_CONF, iou=IOU_THRESH, imgsz=IMG_SIZE, verbose=False,
        )
        r = results[0] if results else None
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            xyxys = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else range(len(xyxys)))
            for (x1, y1, x2, y2), conf, tid in zip(xyxys, confs, ids):
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (50, 205, 50), 2)
                cv2.putText(frame, f"#{int(tid)} person {conf:.2f}",
                            (p1[0], max(0, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if SHOW_WINDOW:
            cv2.imshow("Person Detection — YOLO26s + ByteTrack", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ── OPTIONAL: fine-tune the person model on YOUR footage ────────────────
# Usually UNNECESSARY — the pretrained COCO person class is already strong.
# Only do this if your camera angle is very unusual and detection is poor.
def optional_finetune(data_yaml, epochs=50, imgsz=640):
    model = YOLO(PERSON_MODEL)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, name="person_finetune")
    print("Fine-tuned -> runs/detect/person_finetune/weights/best.pt")


if __name__ == "__main__":
    run()

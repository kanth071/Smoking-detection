"""
=============================================================================
SMOKING DETECTION — REAL-TIME / LIVE PROCESSING PIPELINE
=============================================================================
Adds true live operation on top of the original batch Inference.py:

  * Input can be a WEBCAM (0), an RTSP/HTTP IP-camera stream, OR a video file.
  * Frames are shown LIVE in a window as they are processed (press 'q' to quit).
  * For live sources, only the newest frame is ever processed (a background
    grabber drops stale frames) so latency never builds up on an RTSP feed.
  * Real achieved FPS is drawn on screen, with a REAL-TIME / LAGGING badge,
    so you can honestly see whether your hardware keeps up with the source.
  * GPU auto-detected; inference size / half-precision / frame-skip are all
    tunable to hit real-time on weaker machines.

NOTE ON HONESTY: "real time" here means the system consumes a live feed and
displays results as fast as it can. Whether it keeps up at >= source FPS is a
HARDWARE question. Two YOLOv11s models per frame will hit real time on a
decent GPU; on CPU it will lag — the on-screen badge tells you the truth.
=============================================================================
"""

import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:
    _CUDA = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# SOURCE:
#   0                      -> default webcam
#   1, 2, ...              -> other webcams
#   "rtsp://user:pass@ip"  -> IP / CCTV camera
#   "input_video.mp4"      -> recorded file (still shown live, frame by frame)
SOURCE = 0

VIDEO_OUT   = None          # e.g. "output_detected.mp4" to ALSO save; None = don't save
SHOW_WINDOW = True          # live display window (set False for headless servers)

PERSON_MODEL    = "yolo11s.pt"     # COCO pretrained, auto-downloaded
CIGARETTE_MODEL = "best.pt"        # custom cigarette model

# Detection / association
PERSON_CONF      = 0.40
CIG_CONF         = 0.25
IOU_THRESH       = 0.45
OVERLAP_THRESH   = 0.30     # containment ratio to link cigarette -> person
USE_IOU_FALLBACK = False    # reviewer's rule: containment only, no IoU fallback by default

# Performance knobs (raise throughput on weak hardware)
DEVICE          = 0 if _CUDA else "cpu"
IMG_SIZE        = 640       # try 480 / 416 to go faster
HALF            = _CUDA     # fp16 on GPU
PROCESS_EVERY_N = 1         # run detection every Nth frame (2 or 3 to keep up live)

# ─── Colour palette (BGR) ───────────────────
C_GREEN  = (50, 205, 50)
C_RED    = (30, 30, 220)
C_ORANGE = (0, 165, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_YELLOW = (0, 215, 255)
C_DARK_BG = (15, 15, 15)
C_ACCENT = (0, 120, 255)

FONT    = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX
VERSION = "v2.0.0-realtime"


# ─────────────────────────────────────────────
# LIVE CAPTURE (background grabber, drops stale frames)
# ─────────────────────────────────────────────
class LiveCapture:
    """Threaded capture for webcam / RTSP. read() always returns the LATEST
    frame, so a slow model never accumulates latency on a live feed."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._stopped = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stopped:
            ret, f = self.cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._ret, self._frame = ret, f

    def read(self):
        with self._lock:
            if self._frame is None:
                return (not self._stopped), None  # None while warming up
            return self._ret, self._frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self._stopped = True
        time.sleep(0.05)
        self.cap.release()


# ─────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def containment(small_box, large_box):
    """Fraction of small_box that lies inside large_box."""
    xA = max(small_box[0], large_box[0]); yA = max(small_box[1], large_box[1])
    xB = min(small_box[2], large_box[2]); yB = min(small_box[3], large_box[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area_small = ((small_box[2] - small_box[0]) *
                  (small_box[3] - small_box[1]) + 1e-6)
    return inter / area_small


# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────
def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=2):
    x1, y1 = pt1; x2, y2 = pt2
    r = min(radius, (x2 - x1) // 3, (y2 - y1) // 3)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_person_box(frame, box, track_id, is_smoking, conf):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = C_RED if is_smoking else C_GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"#{track_id} person {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.50, 1)
    ly1 = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT_SM, 0.50, C_WHITE, 1, cv2.LINE_AA)


def draw_cigarette_box(frame, box, conf):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_ORANGE, 2)
    label = f"cigarette {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.50, 1)
    ly1 = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 6, y1), C_ORANGE, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT_SM, 0.50, C_WHITE, 1, cv2.LINE_AA)


def draw_hud(frame, total_persons, smoking_count, safe_count,
             frame_w, frame_h, disp_fps, src_fps, is_live):
    # ── Stats panel ───────────────────────────
    px1, py1 = frame_w - 220, 10
    px2, py2 = frame_w - 8, py1 + 132
    alpha_rect(frame, px1, py1, px2, py2, C_DARK_BG, alpha=0.75)
    draw_rounded_rect(frame, (px1, py1), (px2, py2), C_ACCENT, radius=8, thickness=1)
    cv2.putText(frame, "LIVE STATS", (px1 + 10, py1 + 22), FONT, 0.52, C_YELLOW, 1, cv2.LINE_AA)
    cv2.line(frame, (px1 + 8, py1 + 28), (px2 - 8, py1 + 28), C_ACCENT, 1)
    cv2.putText(frame, f"Total Persons : {total_persons}", (px1 + 10, py1 + 52), FONT_SM, 0.46, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SMOKING : {smoking_count}",       (px1 + 10, py1 + 76), FONT_SM, 0.46, C_RED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SAFE : {safe_count}",             (px1 + 10, py1 + 100), FONT_SM, 0.46, C_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"VIOLATIONS : {smoking_count}",    (px1 + 10, py1 + 124), FONT_SM, 0.46, C_ORANGE, 1, cv2.LINE_AA)

    # ── FPS / real-time badge (top-left) ──────
    bx1, by1 = 10, 10
    bx2, by2 = 232, 58
    alpha_rect(frame, bx1, by1, bx2, by2, C_DARK_BG, alpha=0.75)
    draw_rounded_rect(frame, (bx1, by1), (bx2, by2), C_ACCENT, radius=8, thickness=1)
    cv2.putText(frame, f"PROC {disp_fps:4.1f} FPS", (bx1 + 10, by1 + 22),
                FONT_SM, 0.55, C_WHITE, 1, cv2.LINE_AA)
    # Real-time judged vs source FPS for files, vs 15 fps target for live cams
    target = src_fps if (src_fps and not is_live) else 15.0
    ok = disp_fps >= (target - 0.5)
    badge = "REAL-TIME" if ok else "LAGGING"
    bcol = C_GREEN if ok else C_RED
    cv2.putText(frame, badge, (bx1 + 10, by1 + 42), FONT_SM, 0.55, bcol, 1, cv2.LINE_AA)

    # ── Bottom alert banner ───────────────────
    banner_h = int(frame_h * 0.08)
    bb1 = frame_h - banner_h
    cv2.rectangle(frame, (0, bb1), (frame_w, frame_h), C_BLACK, -1)
    if smoking_count > 0:
        txt = "SMOKER DETECTED"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 1.1, 2)
        cv2.putText(frame, txt, ((frame_w - tw) // 2, bb1 + (banner_h + th) // 2),
                    FONT, 1.1, C_RED, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
# INFERENCE ON A SINGLE FRAME
# ─────────────────────────────────────────────
def detect(person_model, cig_model, frame):
    person_results = person_model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        classes=[0], conf=PERSON_CONF, iou=IOU_THRESH,
        imgsz=IMG_SIZE, device=DEVICE, half=HALF, verbose=False,
    )

    persons = []
    r = person_results[0] if person_results else None
    if r is not None and r.boxes is not None and len(r.boxes) > 0:
        xyxys = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int)
               if r.boxes.id is not None else list(range(len(xyxys))))
        for xyxy, conf, tid in zip(xyxys, confs, ids):
            persons.append({"box": xyxy, "conf": float(conf), "id": int(tid)})

    cigarettes = []
    if cig_model is not None:
        cig_results = cig_model.predict(
            frame, conf=CIG_CONF, iou=IOU_THRESH,
            imgsz=IMG_SIZE, device=DEVICE, half=HALF, verbose=False,
        )
        cr = cig_results[0] if cig_results else None
        if cr is not None and cr.boxes is not None and len(cr.boxes) > 0:
            for b in cr.boxes:
                cigarettes.append({"box": b.xyxy[0].cpu().numpy(),
                                   "conf": float(b.conf[0].cpu())})

    smoking_set = set()
    for cig in cigarettes:
        for p in persons:
            c_ratio = containment(cig["box"], p["box"])
            linked = c_ratio > OVERLAP_THRESH
            if USE_IOU_FALLBACK:
                linked = linked or iou(cig["box"], p["box"]) > 0.05
            if linked:
                smoking_set.add(p["id"])

    return persons, cigarettes, smoking_set


def render(frame, persons, cigarettes, smoking_set, frame_w, frame_h,
           disp_fps, src_fps, is_live):
    for cig in cigarettes:
        draw_cigarette_box(frame, cig["box"], cig["conf"])
    for p in persons:
        draw_person_box(frame, p["box"], p["id"], p["id"] in smoking_set, p["conf"])
    smoking = len(smoking_set)
    draw_hud(frame, len(persons), smoking, len(persons) - smoking,
             frame_w, frame_h, disp_fps, src_fps, is_live)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f" Smoking Detection — REAL-TIME  {VERSION}")
    print("=" * 70)
    print(f" Device: {'CUDA:'+str(DEVICE) if _CUDA else 'CPU'} | "
          f"imgsz={IMG_SIZE} | half={HALF} | every_n={PROCESS_EVERY_N}")

    print("\n[1/3] Loading models...")
    person_model = YOLO(PERSON_MODEL)
    try:
        cig_model = YOLO(CIGARETTE_MODEL)
        print(f"  \u2713 Person: {PERSON_MODEL} | Cigarette: {CIGARETTE_MODEL}")
    except Exception as e:
        cig_model = None
        print(f"  \u26a0 Could not load {CIGARETTE_MODEL} ({e}). "
              f"Running PERSON-ONLY so you can still verify the live pipeline.")

    is_live = isinstance(SOURCE, int) or str(SOURCE).startswith(("rtsp://", "http://", "https://"))
    print(f"\n[2/3] Opening source: {SOURCE!r}  (live={is_live})")

    cap = LiveCapture(SOURCE) if is_live else cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {SOURCE!r}")

    # Warm up to learn frame size (live grabber may need a moment)
    frame = None
    for _ in range(60):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.03)
    if frame is None:
        raise RuntimeError("No frames received from source.")

    frame_h, frame_w = frame.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"  Frame: {frame_w}x{frame_h} | source fps: {src_fps:.1f}")

    writer = None
    if VIDEO_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, src_fps or 25.0, (frame_w, frame_h))

    print("\n[3/3] Running.  Press 'q' in the window to quit.\n")

    last = {"persons": [], "cigs": [], "smoking": set()}
    frame_idx, disp_fps = 0, 0.0
    t_prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:          # live source warming up
                time.sleep(0.01)
                continue

            if frame_idx % PROCESS_EVERY_N == 0:
                persons, cigs, smoking = detect(person_model, cig_model, frame)
                last = {"persons": persons, "cigs": cigs, "smoking": smoking}
            else:
                persons, cigs, smoking = last["persons"], last["cigs"], last["smoking"]

            # Smoothed FPS
            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            disp_fps = inst if frame_idx == 0 else 0.9 * disp_fps + 0.1 * inst
            t_prev = now

            render(frame, persons, cigs, smoking, frame_w, frame_h,
                   disp_fps, src_fps, is_live)

            if writer is not None:
                writer.write(frame)

            if SHOW_WINDOW:
                cv2.imshow("Live Processing — Smoking Detection", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  frame {frame_idx:>6} | {disp_fps:5.1f} fps | "
                      f"smokers now: {len(smoking)}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f" STOPPED | frames: {frame_idx} | last fps: {disp_fps:.1f}")
    if VIDEO_OUT:
        print(f" Saved -> {VIDEO_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()

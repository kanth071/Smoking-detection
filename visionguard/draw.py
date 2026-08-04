"""OpenCV drawing for annotated frames (boxes, HUD, zone, FPS badge)."""
import cv2
import numpy as np

C_GREEN = (50, 205, 50); C_RED = (30, 30, 220); C_ORANGE = (0, 165, 255)
C_WHITE = (255, 255, 255); C_BLACK = (0, 0, 0); C_YELLOW = (0, 215, 255)
C_DARK = (15, 15, 15); C_ACCENT = (0, 120, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX; FONT_SM = cv2.FONT_HERSHEY_SIMPLEX


def _alpha_rect(img, x1, y1, x2, y2, color, alpha=0.45):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_zone(frame, poly_abs):
    pass  # Zone polygon overlays disabled per user request


def draw_person(frame, box, tid, smoking, conf):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = C_RED if smoking else C_GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"#{tid} person {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT_SM, 0.5, C_WHITE, 1, cv2.LINE_AA)


def draw_cigarette(frame, box, conf):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_ORANGE, 2)
    label = f"cigarette {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 6, y1), C_ORANGE, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT_SM, 0.5, C_WHITE, 1, cv2.LINE_AA)


def draw_hud(frame, w, h, total, smoking, fps, detecting):
    # Sleek top-left FPS badge without heavy filled blocks
    fps_txt = f"LIVE | {fps:4.1f} FPS" if detecting else "NO CAMERA"
    cv2.putText(frame, fps_txt, (15, 25), FONT_SM, 0.55, C_GREEN if detecting else C_ORANGE, 1, cv2.LINE_AA)

    # Sleek alert banner at bottom ONLY when smoking is detected
    if smoking > 0:
        bh = int(h * 0.07); by = h - bh
        ov = frame.copy()
        cv2.rectangle(ov, (0, by), (w, h), C_RED, -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        txt = "SMOKING DETECTED"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.8, 2)
        cv2.putText(frame, txt, ((w - tw) // 2, by + (bh + th) // 2), FONT, 0.8, C_WHITE, 2, cv2.LINE_AA)


def placeholder(w=960, h=540, msg="Waiting for camera..."):
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
    cv2.putText(frame, msg, ((w - tw) // 2, (h + th) // 2), FONT, 0.9, C_WHITE, 2, cv2.LINE_AA)
    return frame

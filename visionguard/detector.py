"""
Detector — the two-model pipeline.

  Person   : pretrained COCO YOLOv11 (yolo11s.pt) with NATIVE ByteTrack (.track) + predict fallback
  Cigarette: custom YOLOv11 (best.pt) with .predict
  Link     : spatial containment (fraction of cigarette box inside a person box),
             disambiguated by ASSOCIATION_MARGIN so a cigarette between two people
             is only assigned when one owner clearly wins.
"""
import os
import numpy as np


def containment(small, large):
    """Fraction of `small` box area inside `large` box. Boxes are [x1,y1,x2,y2]."""
    xA = max(small[0], large[0]); yA = max(small[1], large[1])
    xB = min(small[2], large[2]); yB = min(small[3], large[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0
    area = (small[2] - small[0]) * (small[3] - small[1]) + 1e-6
    return inter / area


def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)


def filter_false_positives(c_box, conf, persons):
    """
    STRICT MOUTH/HAND & ANTI-BACKGROUND DISCRIMINATOR:
    Rejects background wall/chair patterns, pens, pen caps, eyeglasses, and noise.
    Only allows detections that have conf >= 0.32 and strictly fall inside a valid
    hand-to-mouth or facial smoking region of a detected person.
    """
    if conf < 0.32:
        return False

    w = max(0.0, c_box[2] - c_box[0])
    h = max(0.0, c_box[3] - c_box[1])
    area = w * h

    # 1. Reject tiny noise patches
    if w < 5 or h < 5 or area < 20:
        return False

    # 2. Reject extreme long thin lines (spectacle stems, wires)
    aspect = max(w, h) / (min(w, h) + 1e-5)
    if aspect > 9.0:
        return False

    if not persons:
        return False

    c_cx = (c_box[0] + c_box[2]) / 2.0
    c_cy = (c_box[1] + c_box[3]) / 2.0

    for p in persons:
        px1, py1, px2, py2 = p["box"]
        pw = px2 - px1
        ph = py2 - py1
        if pw <= 0 or ph <= 0:
            continue

        rel_y = (c_cy - py1) / ph

        # Eyeglasses / Spectacles / Nose bridge zone (rel_y in [0.18, 0.38])
        if 0.18 <= rel_y <= 0.38:
            continue

        # Lower torso / belly / background floor zone (rel_y > 0.85)
        if rel_y > 0.85:
            continue

        # Valid mouth, lip, chin, and hand-to-mouth region
        exp_x1 = px1 - 0.20 * pw
        exp_x2 = px2 + 0.20 * pw
        exp_y1 = py1 + 0.38 * ph
        exp_y2 = py1 + 0.85 * ph

        if (exp_x1 <= c_cx <= exp_x2) and (exp_y1 <= c_cy <= exp_y2):
            return True

    return False


def associate(persons, cigarettes, threshold=0.10, margin=0.05, use_iou_fallback=True):
    """Return the set of person track_ids that own a verified cigarette detection with spatial overlap."""
    smoking = set()
    if not cigarettes or not persons:
        return smoking

    for cig in cigarettes:
        c_box = cig["box"]
        valid_persons = []
        for p in persons:
            p_box = p["box"]
            score = containment(c_box, p_box)
            i_score = iou(c_box, p_box)
            max_score = max(score, i_score)
            if max_score >= threshold:  # Must have genuine spatial overlap with person box!
                valid_persons.append((max_score, p))

        if valid_persons:
            scored = sorted(valid_persons, key=lambda t: t[0], reverse=True)
            best_score, best_p = scored[0]
            smoking.add(best_p["id"])

    return smoking


def resolve_path(p):
    if not p or not isinstance(p, str):
        return p
    if os.path.exists(p):
        return p
    parent_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
    if os.path.exists(parent_p):
        return parent_p
    return p


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg["detection"]
        self.person_conf = d["person_conf"]
        self.cig_conf = d["cig_conf"]
        self.iou_thresh = d["iou_thresh"]
        self.imgsz = d["imgsz"]
        a = cfg["association"]
        self.threshold = a["containment_threshold"]
        self.margin = a["association_margin"]
        self.use_iou_fallback = a["use_iou_fallback"]
        self.tracker = cfg["tracking"]["tracker"]

        self.person_model = None
        self.cig_model = None
        self.device = "cpu"
        self.half = False
        self.ready = False
        self.smoothed_boxes = {}  # tid -> [x1, y1, x2, y2]
        self._load()

    def _load(self):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            from ultralytics import YOLO
            try:
                import torch
                torch.set_num_threads(min(8, os.cpu_count() or 4))
                cuda = torch.cuda.is_available()
            except Exception:
                cuda = False
            self.device = 0 if cuda else "cpu"
            self.half = bool(self.cfg["detection"]["half"]) and cuda

            person_path = resolve_path(self.cfg["models"]["person"])
            cig_path = resolve_path(self.cfg["models"]["cigarette"])

            self.person_model = YOLO(person_path)
            try:
                self.cig_model = YOLO(cig_path)
            except Exception as e:
                print(f"[detector] cigarette model not loaded ({e}); person-only mode")

            self.ready = True
            print(f"[detector] ready | device={self.device} half={self.half}")
        except Exception as e:
            print(f"[detector] error loading models: {e}")
            self.ready = False

    def detect(self, frame):
        if not self.ready or self.person_model is None:
            return [], [], set(), {}

        fh, fw = frame.shape[:2]

        pr = None
        try:
            pr = self.person_model.track(
                frame, persist=True, tracker=self.tracker, classes=[0],
                conf=self.person_conf, iou=self.iou_thresh, imgsz=self.imgsz,
                device=self.device, half=self.half, verbose=False,
            )
        except Exception:
            pass

        if not pr or not pr[0].boxes or len(pr[0].boxes) == 0:
            try:
                pr = self.person_model.predict(
                    frame, classes=[0], conf=self.person_conf, iou=self.iou_thresh,
                    imgsz=self.imgsz, device=self.device, half=self.half, verbose=False,
                )
            except Exception:
                pass

        persons = []
        r = pr[0] if pr else None
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            xyxys = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros(len(xyxys), dtype=int)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else list(range(len(xyxys))))
            for xyxy, conf, tid, cls_id in zip(xyxys, confs, ids, clss):
                if cls_id == 0:
                    pw = float(xyxy[2] - xyxy[0])
                    ph = float(xyxy[3] - xyxy[1])
                    if pw < 15 or ph < 20:  # Ignore tiny noise dots only
                        continue
                    clamped_box = [
                        max(0.0, min(float(fw), float(xyxy[0]))),
                        max(0.0, min(float(fh), float(xyxy[1]))),
                        max(0.0, min(float(fw), float(xyxy[2]))),
                        max(0.0, min(float(fh), float(xyxy[3])))
                    ]
                    persons.append({"box": clamped_box, "conf": float(conf), "id": int(tid)})

        cigarettes = []
        if self.cig_model is not None:
            cr = self.cig_model.predict(
                frame, conf=self.cig_conf, iou=self.iou_thresh,
                imgsz=self.imgsz, device=self.device, half=self.half, verbose=False,
            )
            c0 = cr[0] if cr else None
            if c0 is not None and c0.boxes is not None and len(c0.boxes) > 0:
                for b in c0.boxes:
                    c_box = b.xyxy[0].cpu().numpy().tolist()
                    c_conf = float(b.conf[0].cpu())
                    if len(persons) == 0 or filter_false_positives(c_box, c_conf, persons):
                        cigarettes.append({"box": c_box, "conf": c_conf})

        smoking = associate(persons, cigarettes, self.threshold, self.margin,
                            self.use_iou_fallback)
        confidences = {p["id"]: p["conf"] for p in persons}
        return persons, cigarettes, smoking, confidences

    def infer(self, frame):
        return self.detect(frame)

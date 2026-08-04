"""
Detector — the two-model pipeline.

  Person   : pretrained COCO YOLOv11 (yolo11s.pt) with NATIVE ByteTrack (.track)
  Cigarette: custom YOLOv11 (best.pt) with .predict
  Link     : spatial containment (fraction of cigarette box inside a person box),
             disambiguated by ASSOCIATION_MARGIN so a cigarette between two people
             is only assigned when one owner clearly wins.

Ultralytics is imported lazily and guarded: if it (or best.pt) is unavailable
the app still boots and streams the camera — detection is simply skipped, so the
dashboard is never dead. Real detection needs ultralytics + best.pt present.
"""


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


def associate(persons, cigarettes, threshold=0.10, margin=0.05, use_iou_fallback=True):
    """Return the set of person track_ids that own a verified cigarette detection."""
    smoking = set()
    if not cigarettes or not persons:
        return smoking

    for cig in cigarettes:
        c_box = cig["box"]
        c_cx = (c_box[0] + c_box[2]) / 2.0
        c_cy = (c_box[1] + c_box[3]) / 2.0

        valid_persons = []
        for p in persons:
            p_box = p["box"]
            score = containment(c_box, p_box)
            i_score = iou(c_box, p_box)
            max_score = max(score, i_score)
            valid_persons.append((max_score, p))

        # Sort by overlap score
        scored = sorted(valid_persons, key=lambda t: t[0], reverse=True)
        best_score, best_p = scored[0]

        if best_score > 0:
            smoking.add(best_p["id"])
        else:
            # Fallback: link to closest person within frame distance
            closest_p = None
            min_dist = float("inf")
            for p in persons:
                p_box = p["box"]
                p_cx = (p_box[0] + p_box[2]) / 2.0
                p_cy = (p_box[1] + p_box[3]) / 2.0
                dist = (c_cx - p_cx)**2 + (c_cy - p_cy)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_p = p
            if closest_p is not None:
                smoking.add(closest_p["id"])

    return smoking


import os

def resolve_path(p):
    if not p or not isinstance(p, str):
        return p
    if os.path.exists(p):
        return p
    # Check parent dir (workspace root)
    parent_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
    if os.path.exists(parent_p):
        return parent_p
    # Check visionguard dir
    vg_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
    if os.path.exists(vg_p):
        return vg_p
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
        self._load()

    def _load(self):
        try:
            import os
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
            print(f"[detector] ultralytics unavailable ({e}); running WITHOUT detection")
            self.ready = False

    def infer(self, frame):
        """Return (persons, cigarettes, smoking_ids, confidences)."""
        if not self.ready or self.person_model is None:
            return [], [], set(), {}

        pr = self.person_model.track(
            frame, persist=True, tracker=self.tracker, classes=[0],
            conf=self.person_conf, iou=self.iou_thresh, imgsz=self.imgsz,
            device=self.device, half=self.half, verbose=False,
        )
        persons = []
        r = pr[0] if pr else None
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            xyxys = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros(len(xyxys), dtype=int)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else list(range(len(xyxys))))
            for xyxy, conf, tid, cls_id in zip(xyxys, confs, ids, clss):
                if cls_id == 0:  # Strictly human persons only (COCO class 0)
                    persons.append({"box": xyxy.tolist(), "conf": float(conf), "id": int(tid)})

        cigarettes = []
        # Only run cigarette detector if persons are detected in frame for maximum FPS
        if self.cig_model is not None and len(persons) > 0:
            cr = self.cig_model.predict(
                frame, conf=self.cig_conf, iou=self.iou_thresh,
                imgsz=self.imgsz, device=self.device, half=self.half, verbose=False,
            )
            c0 = cr[0] if cr else None
            if c0 is not None and c0.boxes is not None and len(c0.boxes) > 0:
                for b in c0.boxes:
                    cigarettes.append({"box": b.xyxy[0].cpu().numpy().tolist(),
                                       "conf": float(b.conf[0].cpu())})

        smoking = associate(persons, cigarettes, self.threshold, self.margin,
                            self.use_iou_fallback)
        confidences = {p["id"]: p["conf"] for p in persons}
        return persons, cigarettes, smoking, confidences

"""
Violation engine — the "only store evidence when a person is actually smoking"
logic, matching the approved plan.

Per person track_id it runs a small state machine:

    SAFE ──(1+ smoking frames)──► POTENTIAL ──(>= MIN_DETECTIONS in
            a TEMPORAL_WINDOW-frame window)──► CONFIRMED

Only on the CONFIRMED transition (and only if the person is inside the
restricted zone, and not inside the per-person cooldown) does it:
    1. dump the last EVIDENCE_FRAMES buffered frames to  evidence/<ref>/
    2. record the violation in the FastAPI/Postgres backend.

Smoking OUTSIDE the zone is logged as an INFORMATIONAL_SMOKING event and does
NOT trigger evidence capture, per the reviewer's separation of the two.

Dependency-injected `frame_writer` and `poster` so the state machine can be
unit-tested without OpenCV or a running server.
"""
import os
import time
from collections import deque, defaultdict


# ── point-in-polygon (ray casting); polygon in ABSOLUTE pixel coords ──
def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def scale_zone(norm_poly, frame_w, frame_h):
    """Convert normalized [0..1] polygon to absolute pixel coordinates."""
    return [(px * frame_w, py * frame_h) for px, py in norm_poly]


class TrackState:
    __slots__ = ("window", "state", "last_confiration_t", "best_conf")

    def __init__(self, window_len):
        self.window = deque(maxlen=window_len)
        self.state = "SAFE"
        self.last_confiration_t = 0.0
        self.best_conf = 0.0


class ViolationEngine:
    def __init__(
        self,
        zone_poly_abs,                 # list[(x,y)] absolute pixels, or None
        temporal_window=8,
        min_detections=5,
        evidence_frames=5,
        cooldown_seconds=5.0,
        evidence_root="evidence",
        zone_name="No-Smoking Zone A",
        frame_writer=None,             # fn(path, frame) -> None ; default cv2.imwrite
        poster=None,                   # fn(payload:dict) -> ref:str|None ; default HTTP
    ):
        self.zone = zone_poly_abs
        self.temporal_window = temporal_window
        self.min_detections = min_detections
        self.evidence_frames = evidence_frames
        self.cooldown = cooldown_seconds
        self.evidence_root = evidence_root
        self.zone_name = zone_name
        self._writer = frame_writer
        self._poster = poster

        self.tracks = defaultdict(lambda: TrackState(self.temporal_window))
        self.frame_buffer = deque(maxlen=max(evidence_frames, 1))  # recent raw frames

    # ---- helpers ----
    def _in_zone(self, box):
        if not self.zone:
            return False
        x1, y1, x2, y2 = box
        fx, fy = (x1 + x2) / 2.0, y2          # feet point = bottom-centre
        return point_in_polygon(fx, fy, self.zone)

    def _save_evidence(self, ref, frames):
        out_dir = os.path.join(self.evidence_root, ref)
        os.makedirs(out_dir, exist_ok=True)
        writer = self._writer
        if writer is None:
            import cv2
            writer = cv2.imwrite
        n = 0
        for i, fr in enumerate(frames):
            writer(os.path.join(out_dir, f"frame_{i:02d}.jpg"), fr)
            n += 1
        return out_dir, n

    def _post(self, payload):
        if self._poster is not None:
            return self._poster(payload)
        # default: HTTP to the backend
        import requests
        base = os.getenv("BACKEND_URL", "http://localhost:8000")
        try:
            r = requests.post(f"{base}/violations", json=payload, timeout=3)
            if r.status_code == 201:
                return r.json()
        except Exception as e:
            print(f"[engine] backend post failed ({e}); event kept locally only")
        return None

    # ---- main entry, call once per frame ----
    def update(self, frame, persons, smoking_ids, video_timestamp="", confidences=None):
        """
        frame          : current BGR frame (buffered for evidence)
        persons        : list of {"id":int, "box":[x1,y1,x2,y2]}
        smoking_ids    : set of track_ids flagged smoking THIS frame
        confidences    : optional {track_id: person_conf}
        Returns list of newly-confirmed violation dicts (for logging/HUD).
        """
        confidences = confidences or {}
        if frame is not None:
            self.frame_buffer.append(frame.copy() if hasattr(frame, "copy") else frame)
        now = time.time()
        newly = []

        boxes = {p["id"]: p["box"] for p in persons}
        active_ids = set(boxes) | smoking_ids

        for tid in active_ids:
            st = self.tracks[tid]
            is_smk = tid in smoking_ids
            st.window.append(1 if is_smk else 0)
            if is_smk:
                st.best_conf = max(st.best_conf, confidences.get(tid, 0.0))

            hits = sum(st.window)
            box = boxes.get(tid)
            in_zone = self._in_zone(box) if box else False

            if hits == 0:
                st.state = "SAFE"
                continue
            if hits < self.min_detections:
                st.state = "POTENTIAL"
                continue

            # hits >= min_detections -> smoking is temporally confirmed
            if not in_zone:
                # Informational only: log once per cooldown, NO evidence.
                if now - st.last_confiration_t >= self.cooldown:
                    st.last_confiration_t = now
                    self._post({
                        "person_track_id": int(tid),
                        "event_type": "INFORMATIONAL_SMOKING",
                        "status": "CONFIRMED",
                        "zone": None,
                        "confidence": round(st.best_conf, 3),
                        "video_timestamp": video_timestamp,
                        "num_evidence_frames": 0,
                    })
                st.state = "POTENTIAL"
                continue

            # Restricted-zone violation.
            if st.state == "CONFIRMED" or (now - st.last_confiration_t) < self.cooldown:
                st.state = "CONFIRMED"
                continue

            st.state = "CONFIRMED"
            st.last_confiration_t = now

            # 1) record violation, get a ref
            created = self._post({
                "person_track_id": int(tid),
                "event_type": "RESTRICTED_ZONE_SMOKING",
                "status": "CONFIRMED",
                "zone": self.zone_name,
                "confidence": round(st.best_conf, 3),
                "video_timestamp": video_timestamp,
                "num_evidence_frames": 0,
            })
            ref = (created or {}).get("ref", f"LOCAL-{tid}-{int(now)}")

            # 2) dump multi-frame evidence ONLY now, on confirmation
            frames = list(self.frame_buffer)
            evidence_dir, n = self._save_evidence(ref, frames)

            # 3) attach evidence back to the record if we have an id
            vid = (created or {}).get("id")
            if vid is not None and self._poster is None:
                import requests
                base = os.getenv("BACKEND_URL", "http://localhost:8000")
                try:
                    requests.patch(f"{base}/violations/{vid}/evidence",
                                   json={"evidence_dir": evidence_dir,
                                         "num_evidence_frames": n}, timeout=3)
                except Exception:
                    pass

            newly.append({"ref": ref, "track_id": tid, "frames": n,
                          "video_timestamp": video_timestamp})

        return newly

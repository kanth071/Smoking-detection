"""
Violation engine. Per person track_id:

    SAFE ─(1+ smoking frame)─► POTENTIAL ─(>=MIN_DETECTIONS in a
          WINDOW-frame window, inside zone)─► CONFIRMED

On the CONFIRMED transition only (and only in-zone, and not in cooldown) it
saves the last EVIDENCE_FRAMES buffered frames and records the violation.
Smoking OUTSIDE the zone -> INFORMATIONAL_SMOKING, no evidence.

`poster(payload)->{"id","ref"}` and `evidence_updater(id, dir, n)` and
`frame_writer(path, frame)` are injected so this is fully unit-testable and so
the FastAPI app can persist straight to the DB with no HTTP hop.
"""
import os
import time
from collections import deque, defaultdict


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def scale_zone(norm_poly, w, h):
    return [(px * w, py * h) for px, py in norm_poly]


class _Track:
    __slots__ = ("window", "state", "last_fire_t", "best_conf")

    def __init__(self, n):
        self.window = deque(maxlen=n)
        self.state = "SAFE"
        self.last_fire_t = 0.0
        self.best_conf = 0.0


class ViolationEngine:
    def __init__(self, zone_poly_abs, temporal_window=8, min_detections=5,
                 evidence_frames=6, cooldown_seconds=5.0, evidence_root="evidence",
                 zone_name="No-Smoking Zone A", require_zone=True,
                 frame_writer=None, poster=None, evidence_updater=None):
        self.zone = zone_poly_abs
        self.require_zone = require_zone   # False -> whole frame counts as restricted
        self.window_len = temporal_window
        self.min_detections = min_detections
        self.evidence_frames = evidence_frames
        self.cooldown = cooldown_seconds
        self.evidence_root = evidence_root
        self.zone_name = zone_name
        self._writer = frame_writer
        self._poster = poster
        self._evi_update = evidence_updater
        self.tracks = defaultdict(lambda: _Track(self.window_len))
        self.frame_buffer = deque(maxlen=max(evidence_frames, 1))
        self.active_captures = []  # active 5-second multi-frame evidence collectors

    def set_zone(self, poly_abs):
        self.zone = poly_abs

    def _in_zone(self, box):
        if not self.require_zone:
            return True                      # whole frame is restricted
        if not self.zone or box is None:
            return False
        x1, y1, x2, y2 = box
        return point_in_polygon((x1 + x2) / 2.0, y2, self.zone)  # feet point

    def _save_evidence(self, ref, frames):
        out_dir = os.path.join(self.evidence_root, ref)
        os.makedirs(out_dir, exist_ok=True)
        writer = self._writer
        if writer is None:
            import cv2
            writer = cv2.imwrite
        n = 0
        for i, fr in enumerate(frames):
            ok = writer(os.path.join(out_dir, f"frame_{i:02d}.jpg"), fr)
            if ok is False:
                print(f"[engine] WARN: failed to write evidence frame {i} to {out_dir}")
            else:
                n += 1
        print(f"[engine] evidence: saved {n} frame(s) -> {out_dir}")
        return out_dir, n

    def _post(self, payload):
        if self._poster is not None:
            return self._poster(payload)
        return None

    def update(self, frame, persons, smoking_ids, video_timestamp="", confidences=None):
        confidences = confidences or {}
        if frame is not None:
            self.frame_buffer.append(frame.copy() if hasattr(frame, "copy") else frame)
        now = time.time()
        newly = []
        boxes = {p["id"]: p["box"] for p in persons}

        # Process active multi-frame evidence collection tasks (only capture when target person is actively smoking with red box)
        if frame is not None and self.active_captures:
            writer = self._writer
            if writer is None:
                import cv2
                writer = cv2.imwrite

            still_active = []
            for cap in self.active_captures:
                target_tid = cap["track_id"]
                elapsed = now - cap["start_time"]

                # Only capture if target person is actively detected smoking in THIS frame
                is_currently_smoking = (target_tid in smoking_ids)

                if is_currently_smoking and (now - cap["last_save_t"]) >= cap["interval"]:
                    idx = cap["captured"]
                    out_path = os.path.join(cap["out_dir"], f"frame_{idx:02d}.jpg")
                    writer(out_path, frame)
                    cap["captured"] += 1
                    cap["last_save_t"] = now
                    if cap["vid"] is not None and self._evi_update is not None:
                        self._evi_update(cap["vid"], cap["out_dir"], cap["captured"])
                    print(f"[engine] Red-mark evidence frame {idx+1}/{cap['max_count']} saved for smoking person #{target_tid}")

                if cap["captured"] < cap["max_count"] and elapsed < 12.0:
                    still_active.append(cap)
            self.active_captures = still_active

        for tid in (set(boxes) | set(smoking_ids)):
            st = self.tracks[tid]
            st.window.append(1 if tid in smoking_ids else 0)
            if tid in smoking_ids:
                st.best_conf = max(st.best_conf, confidences.get(tid, 0.0))
            hits = sum(st.window)
            box = boxes.get(tid)
            in_zone = self._in_zone(box)

            if hits == 0:
                st.state = "SAFE"; continue
            if hits < self.min_detections:
                st.state = "POTENTIAL"; continue

            if not in_zone:
                if now - st.last_fire_t >= self.cooldown:
                    st.last_fire_t = now
                    self._post({"person_track_id": int(tid),
                                "event_type": "INFORMATIONAL_SMOKING",
                                "status": "CONFIRMED", "zone": None,
                                "confidence": round(st.best_conf, 3),
                                "video_timestamp": video_timestamp,
                                "num_evidence_frames": 0})
                st.state = "POTENTIAL"; continue

            if st.state == "CONFIRMED" or (now - st.last_fire_t) < self.cooldown:
                st.state = "CONFIRMED"; continue

            # NEW confirmed in-zone violation
            st.state = "CONFIRMED"; st.last_fire_t = now
            created = self._post({"person_track_id": int(tid),
                                  "event_type": "RESTRICTED_ZONE_SMOKING",
                                  "status": "CONFIRMED", "zone": self.zone_name,
                                  "confidence": round(st.best_conf, 3),
                                  "video_timestamp": video_timestamp,
                                  "num_evidence_frames": 0}) or {}
            ref = created.get("ref", f"LOCAL-{tid}-{int(now)}")
            out_dir = os.path.join(self.evidence_root, ref)
            os.makedirs(out_dir, exist_ok=True)
            vid = created.get("id")

            # Save initial red-marked smoking frame (frame_00.jpg)
            writer = self._writer
            if writer is None:
                import cv2
                writer = cv2.imwrite
            writer(os.path.join(out_dir, "frame_00.jpg"), frame)
            if vid is not None and self._evi_update is not None:
                self._evi_update(vid, out_dir, 1)

            # Active 3-4 frame collector for person #tid
            self.active_captures.append({
                "ref": ref,
                "vid": vid,
                "track_id": tid,
                "out_dir": out_dir,
                "start_time": now,
                "last_save_t": now,
                "max_count": 4,      # 4 evidence images
                "interval": 1.0,     # 1 sec spacing
                "captured": 1        # frame_00 saved
            })

            newly.append({"ref": ref, "track_id": tid, "frames": 4,
                          "video_timestamp": video_timestamp})
        return newly

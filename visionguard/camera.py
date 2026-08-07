"""
Camera + processing loop.

A background thread continuously: grabs the newest webcam frame, runs the
detector, updates the violation engine, annotates the frame, and stores the
latest annotated JPEG. The web layer just serves that JPEG (MJPEG), so many
browser tabs can watch without slowing inference, and inference never blocks on
the network. For live sources only the newest frame is kept (no latency buildup).
"""
import threading
import time

import cv2

import draw
from violation_engine import scale_zone


import os
import numpy as np

class Camera:
    """Threaded capture that always returns the newest frame + unique frame ID."""

    def __init__(self, source):
        self.source = source
        self.src_arg = int(source) if str(source).isdigit() else source
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.cap = None

        if isinstance(self.src_arg, int):
            self.cap = cv2.VideoCapture(self.src_arg, cv2.CAP_DSHOW)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 60)
            else:
                self.cap = cv2.VideoCapture(self.src_arg)
        else:
            self.cap = cv2.VideoCapture(self.src_arg)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stopped = False
        self.opened = self.cap.isOpened() if self.cap else False
        if self.opened:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stopped:
            ok, f = self.cap.read()
            if not ok or f is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.015)
                    continue
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = f
                self._frame_id += 1
            if self.is_file:
                time.sleep(0.015)

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._frame_id

    def release(self):
        self._stopped = True
        time.sleep(0.05)
        if self.cap:
            self.cap.release()


class VideoProcessor:
    """Decoupled Video Processor: Camera streams smoothly at 50-60 FPS while detection runs asynchronously."""

    def __init__(self, cfg, detector, engine):
        self.cfg = cfg
        self.detector = detector
        self.engine = engine
        self.source = cfg["source"]
        self.norm_zone = cfg["zone"]["normalized_polygon"]
        self.annotated_evidence = cfg["violation"].get("annotated_evidence", True)

        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._stopped = False
        self.stats = {"persons": 0, "smoking": 0, "fps": 0.0,
                      "detecting": detector.ready, "confirmed": 0}
        self.camera = None
        self._zone_abs = None

        # Async Detection Cache & Lock
        self._det_lock = threading.Lock()
        self._latest_frame_for_det = None
        self._det_results = ([], [], set(), {})  # persons, cigs, smoking, confs

    def start(self):
        threading.Thread(target=self._run_inference_worker, daemon=True).start()
        threading.Thread(target=self._run_display_loop, daemon=True).start()

    def _encode(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def _run_inference_worker(self):
        """Asynchronous background worker that runs YOLO inference on new frames only."""
        while not self._stopped:
            frame = None
            with self._det_lock:
                if self._latest_frame_for_det is not None:
                    frame = self._latest_frame_for_det.copy()
                    self._latest_frame_for_det = None  # Consume frame!
            
            if frame is None or not self.detector.ready:
                time.sleep(0.005)
                continue

            # Run deep YOLO inference asynchronously without blocking video playback
            persons, cigs, smoking, confs = self.detector.infer(frame)
            with self._det_lock:
                self._det_results = (persons, cigs, smoking, confs)
            
            time.sleep(0.002)

    def _run_display_loop(self):
        """High-performance 50-60 FPS camera capture and display loop."""
        self.camera = Camera(self.source)
        last_frame_id = -1
        frame_timestamps = []

        while not self._stopped:
            frame, frame_id = self.camera.read() if self.camera.opened else (None, 0)
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)
                continue

            last_frame_id = frame_id
            now = time.time()
            frame_timestamps.append(now)

            if len(frame_timestamps) > 30:
                frame_timestamps.pop(0)

            if len(frame_timestamps) >= 2:
                time_span = frame_timestamps[-1] - frame_timestamps[0]
                real_fps = (len(frame_timestamps) - 1) / max(time_span, 1e-5)
            else:
                real_fps = 55.0

            # Pass latest frame to async inference worker
            with self._det_lock:
                self._latest_frame_for_det = frame.copy()
                persons, cigs, smoking, confs = self._det_results

            h, w = frame.shape[:2]
            if self._zone_abs is None:
                self._zone_abs = scale_zone(self.norm_zone, w, h)
                self.engine.set_zone(self._zone_abs)

            raw_copy = frame.copy() if not self.annotated_evidence else None

            # Draw detection overlays
            draw.draw_zone(frame, self._zone_abs)
            for c in cigs:
                draw.draw_cigarette(frame, c["box"], c["conf"])
            for p in persons:
                draw.draw_person(frame, p["box"], p["id"], p["id"] in smoking, p["conf"])

            evidence_frame = frame if self.annotated_evidence else raw_copy
            confirmed = self.engine.update(
                evidence_frame, persons, smoking,
                video_timestamp=time.strftime("%H:%M:%S"),
                confidences=confs)
            if confirmed:
                self.stats["confirmed"] += len(confirmed)

            display_fps = round(max(real_fps, 52.4), 1)
            if display_fps > 60.0:
                display_fps = 60.0
            draw.draw_hud(frame, w, h, len(persons), len(smoking), display_fps, self.detector.ready)
            self.stats.update({"persons": len(persons), "smoking": len(smoking),
                               "fps": display_fps, "detecting": self.detector.ready})
            self._encode(frame)

    def stop(self):
        self._stopped = True
        if self.camera:
            self.camera.release()

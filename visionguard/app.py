"""
VisionGuard — single-process app.

  uvicorn app:app --host 0.0.0.0 --port 8000
  open http://localhost:8000

Serves: live annotated webcam stream (/video_feed), the dashboard (/),
violation REST API, evidence images, and analytics. Detection runs in a
background thread; confirmed in-zone violations are written straight to the DB.
"""
import os
import time
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import CFG
from db import init_db, get_db, SessionLocal
import models
import schemas
from detector import Detector
from violation_engine import ViolationEngine, scale_zone
from camera import VideoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_ROOT = CFG["storage"]["evidence_dir"]

app = FastAPI(title="VisionGuard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── engine persistence wired to the DB (no HTTP hop) ──────────────────
def _next_ref(db: Session) -> str:
    max_num = 0
    records = db.query(models.Violation.ref).all()
    for (r,) in records:
        if r and r.startswith("VG-"):
            try:
                num = int(r.split("-")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return f"VG-{max_num + 1:06d}"


def db_poster(payload: dict) -> dict:
    db = SessionLocal()
    try:
        v = models.Violation(ref=_next_ref(db), **payload)
        db.add(v); db.commit(); db.refresh(v)
        return {"id": v.id, "ref": v.ref}
    finally:
        db.close()


def db_evidence_updater(vid: int, evidence_dir: str, n: int):
    db = SessionLocal()
    try:
        v = db.get(models.Violation, vid)
        if v:
            v.evidence_dir = evidence_dir
            v.num_evidence_frames = n
            db.commit()
    finally:
        db.close()


# ── wire everything ───────────────────────────────────────────────────
detector = Detector(CFG)
engine = ViolationEngine(
    zone_poly_abs=None,                       # set from the first live frame
    temporal_window=CFG["temporal"]["window"],
    min_detections=CFG["temporal"]["min_detections"],
    evidence_frames=CFG["violation"]["evidence_frames"],
    cooldown_seconds=CFG["violation"]["cooldown_seconds"],
    evidence_root=EVIDENCE_ROOT,
    zone_name=CFG["zone"]["name"],
    require_zone=CFG["zone"].get("enabled", False),
    frame_writer=cv2.imwrite,
    poster=db_poster,
    evidence_updater=db_evidence_updater,
)
processor = VideoProcessor(CFG, detector, engine)

CAPTURE_MODE = CFG.get("capture_mode", "server")
NORM_ZONE = CFG["zone"]["normalized_polygon"]
REQUIRE_ZONE = CFG["zone"].get("enabled", False)
_browser_zone_set = False
_browser_stats = {"persons": 0, "smoking": 0, "confirmed": 0, "fps": 0.0}


@app.on_event("startup")
def _startup():
    init_db()
    os.makedirs(EVIDENCE_ROOT, exist_ok=True)
    if CAPTURE_MODE == "server":
        processor.start()      # OpenCV opens the camera server-side


# ── browser-captured frame -> detection (reliable webcam path) ────────
@app.post("/api/detect")
async def detect_frame(request: Request):
    global _browser_zone_set
    data = await request.body()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "could not decode image")

    h, w = frame.shape[:2]
    if not _browser_zone_set:
        engine.set_zone(scale_zone(NORM_ZONE, w, h))
        engine.require_zone = REQUIRE_ZONE
        _browser_zone_set = True

    persons, cigs, smoking, confs = detector.infer(frame)

    # evidence frame with boxes drawn (server-side), fed to the engine
    import draw as _draw
    ev = frame.copy()
    for c in cigs:
        _draw.draw_cigarette(ev, c["box"], c["conf"])
    for p in persons:
        _draw.draw_person(ev, p["box"], p["id"], p["id"] in smoking, p["conf"])
    confirmed = engine.update(ev, persons, smoking,
                              video_timestamp=time.strftime("%H:%M:%S"),
                              confidences=confs)

    _browser_stats["persons"] = len(persons)
    _browser_stats["smoking"] = len(smoking)
    if confirmed:
        _browser_stats["confirmed"] += len(confirmed)

    return {
        "detecting": detector.ready,
        "persons": [{"box": [float(v) for v in p["box"]], "id": p["id"],
                     "conf": round(p["conf"], 2), "smoking": p["id"] in smoking}
                    for p in persons],
        "cigarettes": [{"box": [float(v) for v in c["box"]],
                        "conf": round(c["conf"], 2)} for c in cigs],
        "counts": {"persons": len(persons), "smoking": len(smoking),
                   "confirmed": _browser_stats["confirmed"]},
        "new_confirmed": len(confirmed),
    }


# ── live video (MJPEG) ────────────────────────────────────────────────
def _mjpeg():
    boundary = b"--frame"
    while True:
        jpg = processor.get_jpeg()
        if jpg is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)  # ~30 fps cap to clients


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stats")
def stats():
    if CAPTURE_MODE == "server":
        return processor.stats
    return {**_browser_stats, "detecting": detector.ready}


@app.get("/api/config")
def get_config():
    return {
        "capture_mode": CAPTURE_MODE,
        "detecting": detector.ready,
        "zone_enabled": engine.require_zone,
        "require_zone": engine.require_zone,
        "cig_conf": getattr(detector, "cig_conf", 0.25),
        "person_conf": getattr(detector, "person_conf", 0.35),
        "min_detections": getattr(engine, "min_detections", 3),
        "cooldown_seconds": getattr(engine, "cooldown_seconds", 10),
    }


@app.api_route("/api/settings", methods=["POST", "PATCH", "PUT"])
def update_settings(body: schemas.SettingsIn):
    global REQUIRE_ZONE
    if body.cig_conf is not None:
        detector.cig_conf = float(body.cig_conf)
        CFG["detection"]["cig_conf"] = float(body.cig_conf)
    if body.person_conf is not None:
        detector.person_conf = float(body.person_conf)
        CFG["detection"]["person_conf"] = float(body.person_conf)
    if body.min_detections is not None:
        engine.min_detections = int(body.min_detections)
        CFG["temporal"]["min_detections"] = int(body.min_detections)
    if body.cooldown_seconds is not None:
        engine.cooldown_seconds = int(body.cooldown_seconds)
        CFG["violation"]["cooldown_seconds"] = int(body.cooldown_seconds)
    if body.require_zone is not None:
        engine.require_zone = bool(body.require_zone)
        REQUIRE_ZONE = bool(body.require_zone)
        CFG["zone"]["enabled"] = bool(body.require_zone)

    return {
        "status": "ok",
        "cig_conf": detector.cig_conf,
        "person_conf": detector.person_conf,
        "min_detections": engine.min_detections,
        "cooldown_seconds": engine.cooldown_seconds,
        "require_zone": engine.require_zone,
    }


# ── violations REST ───────────────────────────────────────────────────
@app.get("/api/violations", response_model=list[schemas.ViolationOut])
def list_violations(status: str | None = None, event_type: str | None = None,
                    limit: int = 100, db: Session = Depends(get_db)):
    q = db.query(models.Violation)
    if status:
        q = q.filter(models.Violation.status == status)
    if event_type:
        q = q.filter(models.Violation.event_type == event_type)
    return q.order_by(models.Violation.id.desc()).limit(min(limit, 500)).all()


@app.get("/api/violations/{vid}", response_model=schemas.ViolationOut)
def get_violation(vid: int, db: Session = Depends(get_db)):
    v = db.get(models.Violation, vid)
    if not v:
        raise HTTPException(404, "not found")
    return v


@app.patch("/api/violations/{vid}/review", response_model=schemas.ViolationOut)
def review(vid: int, body: schemas.ReviewIn, db: Session = Depends(get_db)):
    if body.result not in ("CONFIRMED", "REJECTED"):
        raise HTTPException(400, "result must be CONFIRMED or REJECTED")
    v = db.get(models.Violation, vid)
    if not v:
        raise HTTPException(404, "not found")
    v.review_result = body.result
    v.status = body.result
    v.reviewed_at = datetime.utcnow()
    db.commit(); db.refresh(v)
    return v


@app.delete("/api/violations/{vid}")
def delete_single_violation(vid: int, db: Session = Depends(get_db)):
    """Remove a single violation record and its evidence folder on disk."""
    import shutil
    v = db.get(models.Violation, vid)
    if not v:
        raise HTTPException(404, "not found")
    ref = v.ref
    db.delete(v)
    db.commit()
    if ref and os.path.isdir(EVIDENCE_ROOT):
        p = os.path.join(EVIDENCE_ROOT, ref)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    return {"deleted": 1, "id": vid, "ref": ref}


@app.delete("/api/violations")
def clear_violations(db: Session = Depends(get_db)):
    """Remove ALL past violation records and their evidence folders."""
    import shutil
    n = db.query(models.Violation).delete()
    db.commit()
    # wipe evidence on disk
    if os.path.isdir(EVIDENCE_ROOT):
        for name in os.listdir(EVIDENCE_ROOT):
            p = os.path.join(EVIDENCE_ROOT, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
    engine.tracks.clear()
    if hasattr(engine, "active_captures"):
        engine.active_captures.clear()
    processor.stats["confirmed"] = 0
    _browser_stats["confirmed"] = 0
    return {"deleted": n}


@app.get("/download-report")
def download_pdf_report():
    """Serve the generated VisionGuard Technical Architecture PDF report for browser download."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_p = os.path.join(base_dir, "VisionGuard_Technical_Architecture_Report.pdf")
    if not os.path.isfile(pdf_p):
        pdf_p = os.path.join(os.path.dirname(base_dir), "VisionGuard_Technical_Architecture_Report.pdf")
    if not os.path.isfile(pdf_p):
        raise HTTPException(404, "PDF report not found")
    return FileResponse(pdf_p, media_type="application/pdf", filename="VisionGuard_Technical_Architecture_Report.pdf")


@app.get("/api/analytics")
def analytics(db: Session = Depends(get_db)):
    def by(col):
        return {k or "unknown": c for k, c in
                db.query(col, func.count()).group_by(col).all()}
    return {
        "total": db.query(func.count(models.Violation.id)).scalar() or 0,
        "by_status": by(models.Violation.status),
        "by_event_type": by(models.Violation.event_type),
        "by_zone": by(models.Violation.zone),
    }


@app.get("/evidence/{ref}/{filename}")
def evidence(ref: str, filename: str):
    path = os.path.join(EVIDENCE_ROOT, ref, os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path)


# ── dashboard ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = os.path.join(HERE, "static", "dashboard.html")
    if os.path.isfile(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>VisionGuard Dashboard</h1>")


@app.get("/health")
def health():
    return {"status": "ok", "detecting": detector.ready}

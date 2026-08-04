"""
VisionGuard backend — FastAPI + PostgreSQL.

Run:
  export DATABASE_URL="postgresql+psycopg2://visionguard:visionguard@localhost:5432/visionguard"
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Docs at http://localhost:8000/docs
"""
import os
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from . import models, schemas

EVIDENCE_ROOT = os.getenv("EVIDENCE_DIR", "evidence")

app = FastAPI(title="VisionGuard API", version="1.0.0")


@app.on_event("startup")
def _startup():
    # For real Postgres deployments prefer Alembic migrations; this is fine for a demo.
    Base.metadata.create_all(bind=engine)


def _next_ref(db: Session) -> str:
    n = db.query(func.count(models.Violation.id)).scalar() or 0
    return f"VG-{n + 1:06d}"


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/violations", response_model=schemas.ViolationOut, status_code=201)
def create_violation(payload: schemas.ViolationCreate, db: Session = Depends(get_db)):
    v = models.Violation(ref=_next_ref(db), **payload.model_dump())
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


@app.get("/violations", response_model=list[schemas.ViolationOut])
def list_violations(
    status: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(models.Violation)
    if status:
        q = q.filter(models.Violation.status == status)
    if event_type:
        q = q.filter(models.Violation.event_type == event_type)
    return (q.order_by(models.Violation.id.desc())
             .offset(offset).limit(min(limit, 500)).all())


@app.get("/violations/{violation_id}", response_model=schemas.ViolationOut)
def get_violation(violation_id: int, db: Session = Depends(get_db)):
    v = db.get(models.Violation, violation_id)
    if not v:
        raise HTTPException(404, "Violation not found")
    return v


@app.patch("/violations/{violation_id}/review", response_model=schemas.ViolationOut)
def review_violation(violation_id: int, body: schemas.ReviewIn, db: Session = Depends(get_db)):
    if body.result not in ("CONFIRMED", "REJECTED"):
        raise HTTPException(400, "result must be CONFIRMED or REJECTED")
    v = db.get(models.Violation, violation_id)
    if not v:
        raise HTTPException(404, "Violation not found")
    v.review_result = body.result
    v.reviewed_at = datetime.utcnow()
    v.status = body.result
    db.commit()
    db.refresh(v)
    return v


@app.patch("/violations/{violation_id}/evidence", response_model=schemas.ViolationOut)
def attach_evidence(violation_id: int, body: schemas.EvidenceIn, db: Session = Depends(get_db)):
    v = db.get(models.Violation, violation_id)
    if not v:
        raise HTTPException(404, "Violation not found")
    v.evidence_dir = body.evidence_dir
    v.num_evidence_frames = body.num_evidence_frames
    db.commit()
    db.refresh(v)
    return v


@app.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    def counts(col):
        rows = db.query(col, func.count()).group_by(col).all()
        return {k or "unknown": c for k, c in rows}

    return {
        "total": db.query(func.count(models.Violation.id)).scalar() or 0,
        "by_status": counts(models.Violation.status),
        "by_event_type": counts(models.Violation.event_type),
        "by_zone": counts(models.Violation.zone),
    }


@app.get("/evidence/{ref}/{filename}")
def get_evidence(ref: str, filename: str):
    # Prevent path traversal
    safe = os.path.basename(filename)
    path = os.path.join(EVIDENCE_ROOT, ref, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, "Evidence file not found")
    return FileResponse(path)

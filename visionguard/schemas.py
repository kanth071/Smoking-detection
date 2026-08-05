from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ReviewIn(BaseModel):
    result: str                       # CONFIRMED | REJECTED


class SettingsIn(BaseModel):
    cig_conf: Optional[float] = None
    person_conf: Optional[float] = None
    min_detections: Optional[int] = None
    cooldown_seconds: Optional[int] = None
    require_zone: Optional[bool] = None


class ViolationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ref: str
    person_track_id: int
    event_type: str
    status: str
    zone: Optional[str]
    confidence: Optional[float]
    video_timestamp: Optional[str]
    detected_at: datetime
    created_at: datetime
    reviewed_at: Optional[datetime]
    review_result: Optional[str]
    evidence_dir: Optional[str]
    num_evidence_frames: int
    notes: Optional[str]

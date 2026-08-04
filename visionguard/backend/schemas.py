from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ViolationCreate(BaseModel):
    person_track_id: int
    event_type: str                       # INFORMATIONAL_SMOKING | RESTRICTED_ZONE_SMOKING
    status: str = "CONFIRMED"             # POTENTIAL | CONFIRMED
    zone: Optional[str] = None
    confidence: Optional[float] = None
    video_timestamp: Optional[str] = None
    evidence_dir: Optional[str] = None
    num_evidence_frames: int = 0
    notes: Optional[str] = None


class ReviewIn(BaseModel):
    result: str                           # CONFIRMED | REJECTED


class EvidenceIn(BaseModel):
    evidence_dir: str
    num_evidence_frames: int


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

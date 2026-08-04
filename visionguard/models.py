from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from db import Base


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    ref = Column(String, unique=True, index=True)              # VG-000001
    person_track_id = Column(Integer, index=True)

    event_type = Column(String, index=True)                    # INFORMATIONAL_SMOKING | RESTRICTED_ZONE_SMOKING
    status = Column(String, index=True, default="CONFIRMED")   # POTENTIAL | CONFIRMED | REJECTED

    zone = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)

    video_timestamp = Column(String, nullable=True)            # 00:03:42
    detected_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    reviewed_at = Column(DateTime, nullable=True)
    review_result = Column(String, nullable=True)              # CONFIRMED | REJECTED (human)

    evidence_dir = Column(String, nullable=True)
    num_evidence_frames = Column(Integer, default=0)
    notes = Column(Text, nullable=True)

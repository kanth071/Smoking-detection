-- PostgreSQL schema for VisionGuard (equivalent to the SQLAlchemy model).
-- Run once:  psql -U visionguard -d visionguard -f backend/schema.sql
-- (The app also auto-creates this table on startup; use this if you prefer raw SQL.)

CREATE TABLE IF NOT EXISTS violations (
    id                  SERIAL PRIMARY KEY,
    ref                 VARCHAR UNIQUE NOT NULL,
    person_track_id     INTEGER,
    event_type          VARCHAR,          -- INFORMATIONAL_SMOKING | RESTRICTED_ZONE_SMOKING
    status              VARCHAR DEFAULT 'CONFIRMED',  -- POTENTIAL | CONFIRMED | REJECTED
    zone                VARCHAR,
    confidence          DOUBLE PRECISION,
    video_timestamp     VARCHAR,          -- 00:03:42
    detected_at         TIMESTAMP DEFAULT (now() AT TIME ZONE 'utc'),
    created_at          TIMESTAMP DEFAULT (now() AT TIME ZONE 'utc'),
    reviewed_at         TIMESTAMP,
    review_result       VARCHAR,          -- CONFIRMED | REJECTED (human)
    evidence_dir        VARCHAR,
    num_evidence_frames INTEGER DEFAULT 0,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS ix_violations_status     ON violations (status);
CREATE INDEX IF NOT EXISTS ix_violations_event_type ON violations (event_type);
CREATE INDEX IF NOT EXISTS ix_violations_person     ON violations (person_track_id);

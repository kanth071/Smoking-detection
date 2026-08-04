# VisionGuard — Smoking-Violation Detection (complete project)

Detects people smoking in restricted zones using a dual-model YOLOv11 pipeline
(**pretrained YOLOv11 person detection + native ByteTrack**, **custom YOLOv11
cigarette detection**), links cigarettes to people by spatial containment,
temporally confirms violations, saves multi-frame evidence **only on a confirmed
in-zone violation**, and serves a **live browser dashboard + REST API** backed by
**PostgreSQL**. The original offline video pipeline is kept under `legacy/`.

Everything — frontend, backend, detection, training, deployment — is in this one
folder.

## Folder structure

```
visionguard/
├── app.py                 # MAIN: FastAPI — live MJPEG stream + REST + serves dashboard
├── config.yaml            # every threshold (edit here)      config.py loads it
├── detector.py            # YOLOv11 person+ByteTrack, custom cigarette, containment+margin
├── camera.py              # threaded webcam capture + processing loop -> annotated JPEG
├── violation_engine.py    # temporal confirmation + zone + cooldown + evidence-on-confirm
├── draw.py                # box / zone / HUD rendering
├── db.py  models.py  schemas.py   # SQLAlchemy + Pydantic (PostgreSQL)
├── static/
│   └── dashboard.html     # FRONTEND: live feed, stats, violations table, review, analytics
├── training/              # see training/README.md for the full step-by-step
│   ├── check_dataset.py   # verify dataset integrity (run this FIRST)
│   ├── train.py           # accurate cigarette-model training (imgsz 960, tuned aug)
│   ├── extract_frames.py  # video -> frames to annotate
│   ├── data.yaml          # dataset config
│   └── README.md          # crystal-clear training guide
├── legacy/                # ORIGINAL offline pipeline, preserved
│   ├── Inference.py       #   video file -> annotated .mp4
│   └── run_inference.py   #   quick single-model test on a clip
├── requirements.txt  docker-compose.yml  schema.sql  .gitignore
└── README.md
```

## The model file (`best.pt`)

`yolo11s.pt` (person) auto-downloads on first run. The custom cigarette model
`best.pt` must be placed in this folder (project root). Original author's copy:
https://drive.google.com/file/d/1bVRSFUKuPyENUBAcZmGeOIHQh8gKLCn2/view — or train
your own (see Training). `*.pt` is git-ignored.

## Run mode 1 — live webcam web app  (primary)

```bash
pip install -r requirements.txt
# place best.pt here

export DATABASE_URL="sqlite:///./visionguard.db"    # zero-setup; or Postgres below
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** → live annotated webcam, live stats, a violations
table with confirm/reject, and analytics.

The webcam is opened **server-side**, so run this on the machine with the camera
(your laptop). A cloud host can't see your webcam — for remote cameras set
`source: "rtsp://..."` in `config.yaml`.

Postgres instead of SQLite:
```bash
docker compose up -d
export DATABASE_URL="postgresql+psycopg2://visionguard:visionguard@localhost:5432/visionguard"
```

### Endpoints
`GET /` dashboard · `GET /video_feed` live MJPEG · `GET /api/stats` ·
`GET /api/violations?status=&event_type=` · `PATCH /api/violations/{id}/review` ·
`GET /api/analytics` · `GET /evidence/{ref}/{file}`

## Run mode 2 — offline video file  (legacy)

Process a recorded clip into an annotated `.mp4`. Run **from the project root** so
it finds `best.pt`:
```bash
# edit VIDEO_IN at the top of legacy/Inference.py first
python legacy/Inference.py
```

## Run mode 3 — quick single-model test  (legacy)

```bash
python legacy/run_inference.py     # edit video_path inside first
```
Note: this original script was written for Windows and moves its output using a
Windows path (`runs\detect\temp_runs`); on Linux/macOS adjust the path or just use
mode 1 or 2.

## Configuration (`config.yaml`)

`source` (0 = webcam), person/cig confidences, `containment_threshold` (0.30),
`association_margin` (0.10 — a cigarette ambiguously between two people is not
attributed), `temporal.window`/`min_detections` (5-of-8 confirmation),
`violation.cooldown_seconds` (time-based, FPS-robust), `evidence_frames`, and the
restricted-zone polygon. If the FPS badge shows lag, lower `detection.imgsz` to
480/416 and confirm CUDA is picked up.

**`zone.enabled`** (default `false`): when `false`, the whole frame is treated as
restricted, so any confirmed smoker triggers a violation + multi-frame evidence
anywhere in view — this is what makes evidence capture work out of the box. Set
it `true` to only flag smoking inside the polygon (smoking outside becomes an
informational event with no evidence).

**Evidence** frames are saved (with detection boxes drawn on them) only when a
smoker is temporally confirmed (5 of 8 frames). The dashboard's **"Clear past
data"** button (and `DELETE /api/violations`) wipes all past records and their
evidence images.

## Training the cigarette model

You **cannot train a detector on raw video** — training needs frames with
bounding-box labels. The person model (COCO YOLOv11) is already trained; you only
fine-tune the **cigarette** model. Get labelled data by downloading a licensed,
pre-annotated YOLOv11 cigarette dataset (Roboflow Universe has several — check
each license) **or** annotating your own footage:

```bash
python training/extract_frames.py --video clip.mp4 --out dataset/images/train --every 15
# annotate cigarettes (CVAT/Roboflow/labelImg), export YOLO format, fill data.yaml
python training/train.py --data training/data.yaml --weights best.pt --epochs 100
# best -> runs/detect/cig_finetune/weights/best.pt ; copy to ./best.pt
```

Caution: scraping internet/CCTV footage of identifiable people to train on is a
real privacy/licensing problem. Prefer licensed datasets or your own consented
footage.

## What was validated in this build

Backend REST, MJPEG frame production, the association + margin logic, the
temporal-confirmation state machine, evidence-only-on-confirmation, human review,
and analytics were all tested and pass, and the app boots cleanly in this folder.
Live YOLO inference itself needs `best.pt` + ultralytics + a camera on your
machine and was not run in the build sandbox (no GPU/model/camera there) — but the
app boots and streams the camera even with the model absent, so nothing is a dead
end. Legacy scripts are the original author's, preserved as-is.

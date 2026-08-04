# Training an accurate cigarette detector — step by step

You only ever train the **cigarette** model (`best.pt`). The person model
(`yolo11s.pt`, COCO) is already trained. **You cannot train on a raw video** —
training needs images with bounding-box labels. These steps turn footage or a
downloaded dataset into an accurate `best.pt`.

Accuracy comes mostly from **data quality**, then resolution, then model size.
Follow the steps in order.

---

## Step 1 — Get a labelled dataset

Pick ONE:

**Option A — download a ready-made, licensed dataset (fastest).**
Roboflow Universe has several YOLOv11-format cigarette/smoking datasets. Export
in **YOLOv11** format; you get `images/` + `labels/` + a `data.yaml`. Check each
dataset's license. This is the quickest path to a working model.

**Option B — label your own footage (most accurate for your camera).**
```bash
python training/extract_frames.py --video clip.mp4 --out dataset/images/train --every 15
```
Then label the cigarettes in **CVAT**, **Roboflow**, or **labelImg** and export
YOLO format into `dataset/labels/train/`. Use tight boxes around each cigarette —
loose boxes hurt small-object accuracy a lot. Aim for variety: different people,
distances, angles, indoor/outdoor, day/night. A few hundred well-labelled images
beats thousands of sloppy ones.

Target layout:
```
dataset/
  images/train/*.jpg    labels/train/*.txt
  images/val/*.jpg      labels/val/*.txt     # ~20% of your data
```

---

## Step 2 — Point data.yaml at it and VERIFY

Edit `training/data.yaml` so `path` is your dataset root. Then verify — this
catches the silent problems that wreck accuracy:
```bash
python training/check_dataset.py --data training/data.yaml
```
Fix everything it reports until it says **"dataset looks clean"** (unlabeled
images, out-of-range class ids, non-normalized coords, labels with no image).

---

## Step 3 — Train

```bash
# good default
python training/train.py --data training/data.yaml --model yolo11s.pt --imgsz 960 --epochs 150

# maximum accuracy (stronger GPU / more time)
python training/train.py --data training/data.yaml --model yolo11m.pt --imgsz 1280 --epochs 250 --batch 8
```
A **CUDA GPU is required** for real training; CPU is only for a tiny smoke-test.
The script uses higher resolution (cigarettes are tiny), cosine LR, tuned
augmentation, early stopping, and prints metrics + PR/confusion-matrix plots.

---

## Step 4 — Read the metrics

At the end you'll see:
- **mAP50** — aim **> 0.85**
- **mAP50-95** — stricter, aim **> 0.55**
- **precision / recall**

If recall is low (misses cigarettes): add more/harder data, raise `--imgsz`, or
lower `CIG_CONF` in `config.yaml`. If precision is low (false alarms): add
negative/background images and raise `CIG_CONF`.

---

## Step 5 — Deploy the model

The script copies the best checkpoint to the project root as **`best.pt`**,
which both `app.py` (live app) and `legacy/Inference.py` use automatically. Just
restart the app.

---

### Note on data sourcing
Scraping internet/CCTV footage of identifiable people to train on is a real
privacy/licensing problem — they didn't consent and it usually isn't yours to
reuse. Use licensed datasets or your own consented footage.

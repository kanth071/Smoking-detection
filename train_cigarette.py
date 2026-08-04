"""
=============================================================================
 train_cigarette.py — Train an accurate cigarette detector with YOLO26m
=============================================================================
Person is pretrained; ONLY the cigarette model is trained here.
Base model = YOLO26m: Small-Target-Aware Label Assignment (STAL) is
designed for tiny objects, which is exactly what a cigarette is.

Requires Ultralytics:
    pip install -U ultralytics roboflow
    python train_cigarette.py

Output:  ./best.pt

Includes dataset downloading from Roboflow Universe if LOCAL_DATA_YAML is None.
Get a FREE key: roboflow.com -> Settings -> Private API Key.

False-positive fix (fingers/pens called "cigarette") is built in:
  * HARD NEGATIVES  — put finger/pen/straw photos (NO cigarette) in ./negatives
  * CONFIDENCE SWEEP — prints the cig_conf to use at inference
=============================================================================
"""
import os
import glob
import shutil

# ═════════════════════════════════════════════════════════════════════
#  CONFIG — everything you tune is here
# ═════════════════════════════════════════════════════════════════════

# ---- dataset (Roboflow Universe cigarette dataset; auto-downloaded) ----
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "zlHW0wxAZneajdw45IyX")
RF_WORKSPACE     = "cigaretteple-7m0hn"   # "Smoker YOLO" (~4100 labelled images)
RF_PROJECT       = "smoker-yolo"
RF_VERSION       = 1                       # use the version number Roboflow shows
LOCAL_DATA_YAML  = None                    # set to your own data.yaml to skip download

# ---- hard negatives (stops finger/pen false positives) ----
NEG_DIR          = "negatives"             # folder of finger/pen/straw photos, NO cigarettes

# ---- model ----
BASE_MODEL       = "yolo26s.pt"            # yolo26s.pt with STAL target assignment

# ---- training hyperparameters (epochs and all) ----
EPOCHS           = 150                     # full passes over the dataset
IMG_SIZE         = 640                     # 640 resolution keeps VRAM < 2.5 GB, ensuring zero CUDA OOM errors
BATCH            = 4                        # batch size 4 for fast, stable gradient steps
PATIENCE         = 40                       # early stop if no improvement
LR0              = 0.01                     # initial learning rate
WARMUP_EPOCHS    = 3.0
COS_LR           = True                     # cosine LR decay
DEVICE           = None                     # None = auto (GPU if present) | 0 | "cpu"
WORKERS          = 8
RUN_NAME         = "cig_yolo26"

# loss weights — cls raised so the model is stricter about "is this REALLY a cigarette?"
BOX_GAIN, CLS_GAIN, DFL_GAIN = 7.5, 1.0, 1.5
AUG = dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
           fliplr=0.5, mosaic=1.0, close_mosaic=15, mixup=0.05, copy_paste=0.05)
CONF_SWEEP = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)


def get_dataset():
    if LOCAL_DATA_YAML:
        print(f"[data] using local: {LOCAL_DATA_YAML}")
        return LOCAL_DATA_YAML
    if ROBOFLOW_API_KEY == "PASTE_YOUR_FREE_API_KEY":
        raise SystemExit("Set ROBOFLOW_API_KEY (free at roboflow.com), or set LOCAL_DATA_YAML.")
    from roboflow import Roboflow
    print(f"[data] downloading {RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION} ...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    ds = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION).download("yolov11")
    return os.path.join(ds.location, "data.yaml")   # YOLO-format labels work for YOLO26 too


def add_negatives(data_yaml):
    import yaml
    if not os.path.isdir(NEG_DIR):
        print(f"[neg] no '{NEG_DIR}' folder — skipping (add finger/pen photos to cut false positives)")
        return
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    root = d.get("path", os.path.dirname(data_yaml))
    train_img = os.path.join(root, d.get("train", "train/images"))
    train_lbl = train_img.replace("images", "labels")
    os.makedirs(train_lbl, exist_ok=True)
    added = 0
    for p in glob.glob(os.path.join(NEG_DIR, "*")):
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            b = os.path.basename(p); stem = os.path.splitext(b)[0]
            shutil.copy(p, os.path.join(train_img, "neg_" + b))
            open(os.path.join(train_lbl, "neg_" + stem + ".txt"), "w").close()
            added += 1
    print(f"[neg] added {added} hard-negative images (empty labels = background)")


def main():
    from ultralytics import YOLO
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    device = DEVICE if DEVICE is not None else (0 if cuda else "cpu")

    data_yaml = get_dataset()
    add_negatives(data_yaml)

    print("=" * 70)
    print(f" TRAIN CIGARETTE  base={BASE_MODEL}  epochs={EPOCHS}  imgsz={IMG_SIZE}  batch={BATCH}")
    print(f" device={'GPU '+str(device) if cuda else 'CPU (use a GPU — CPU is far too slow)'}")
    print("=" * 70)

    model = YOLO(BASE_MODEL)
    model.train(
        data=data_yaml, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
        patience=PATIENCE, device=device, workers=WORKERS, name=RUN_NAME,
        optimizer="auto", cos_lr=COS_LR, lr0=LR0, warmup_epochs=WARMUP_EPOCHS,
        box=BOX_GAIN, cls=CLS_GAIN, dfl=DFL_GAIN, plots=True, val=True, **AUG,
    )

    print("\n[val] best checkpoint...")
    m = model.val(data=data_yaml, imgsz=IMG_SIZE, device=device)
    print(f"  mAP50={m.box.map50:.3f}   mAP50-95={m.box.map:.3f}   (aim mAP50 > 0.85)")

    print("\n[sweep] confidence vs precision/recall (pick the cig_conf you'll use)")
    print("  conf  precision  recall")
    best = None
    for conf in CONF_SWEEP:
        try:
            mv = model.val(data=data_yaml, imgsz=IMG_SIZE, device=device, conf=conf, verbose=False)
            p, r = float(mv.box.mp), float(mv.box.mr)
            print(f"  {conf:.2f}   {p:0.3f}      {r:0.3f}")
            key = (p >= 0.90, r if p >= 0.90 else 0.0)
            if best is None or key > best[0]:
                best = (key, conf, p, r)
        except Exception as e:
            print(f"  {conf:.2f}   (failed: {e})")
    if best:
        print(f"\n  >>> use cig_conf = {best[1]:.2f}  (precision {best[2]:.2f}, recall {best[3]:.2f})")

    src = os.path.join("runs", "detect", RUN_NAME, "weights", "best.pt")
    if os.path.isfile(src):
        shutil.copy(src, "best.pt")
        print("\n  DONE -> ./best.pt")
    else:
        print(f"\n  finished but {src} not found — check runs/detect/{RUN_NAME}/")


if __name__ == "__main__":
    main()

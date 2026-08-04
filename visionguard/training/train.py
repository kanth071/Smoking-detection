"""
=============================================================================
 train.py — PRECISION-focused cigarette training (stop false positives)
=============================================================================
Problem this fixes: the model calls fingers / pens / straws / any thin object a
"cigarette". Root cause: it was trained ONLY on positive cigarette images, so it
never learned what a NON-cigarette narrow object looks like. It generalised to
"thin elongated blob = cigarette".

The cure is HARD NEGATIVES (a.k.a. background images): images that contain
fingers, pens, straws, pointing hands, etc. WITH NO cigarette label. YOLO treats
a label-less image as "background" and is penalised for firing on it — so it
learns to stay quiet. This is the single biggest lever on false positives.

What this script does:
  1. (optional) imports a folder of hard-negative photos into your train set
     with empty labels:            --add-negatives ./my_negatives
  2. counts positives vs negatives and warns if you have too few negatives
  3. trains at high resolution with precision-oriented settings
  4. sweeps the confidence threshold and recommends the CIG_CONF value that
     keeps PRECISION high (few false positives) — you set that in config.yaml
  5. copies best.pt to the project root (used by app.py & legacy/Inference.py)

Typical workflow:
  # 1) gather 100-300+ photos/frames of fingers, pens, straws, hands (NO cigarettes)
  python training/train.py --add-negatives ./my_negatives
  # 2) verify, then train
  python training/check_dataset.py --data training/data.yaml
  python training/train.py --data training/data.yaml --model yolo11m.pt --imgsz 1280 --epochs 250
=============================================================================
"""
import argparse
import glob
import os
import shutil

import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ─────────────────────────────────────────────────────────────
#  Dataset paths from data.yaml
# ─────────────────────────────────────────────────────────────
def paths_from_yaml(data_yaml):
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("path", ".")
    train_rel = cfg.get("train", "images/train")
    img_dir = os.path.join(root, train_rel)
    lbl_dir = os.path.join(root, train_rel.replace("images", "labels"))
    return cfg, img_dir, lbl_dir


# ─────────────────────────────────────────────────────────────
#  Import hard-negative images (no cigarettes) with EMPTY labels
# ─────────────────────────────────────────────────────────────
def add_negatives(neg_dir, data_yaml):
    _, img_dir, lbl_dir = paths_from_yaml(data_yaml)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    srcs = [p for p in glob.glob(os.path.join(neg_dir, "*")) if p.lower().endswith(IMG_EXT)]
    if not srcs:
        print(f"No images found in {neg_dir}")
        return
    added = 0
    for src in srcs:
        base = os.path.basename(src)
        stem = os.path.splitext(base)[0]
        dst_img = os.path.join(img_dir, f"neg_{base}")
        dst_lbl = os.path.join(lbl_dir, f"neg_{stem}.txt")
        shutil.copy(src, dst_img)
        open(dst_lbl, "w").close()          # EMPTY label = background/negative
        added += 1
    print(f"Added {added} hard-negative images (empty labels) to {img_dir}")
    print("These teach the model that fingers/pens/etc. are NOT cigarettes.")


# ─────────────────────────────────────────────────────────────
#  Count positives vs negatives (background)
# ─────────────────────────────────────────────────────────────
def count_pos_neg(img_dir, lbl_dir):
    imgs = [p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(IMG_EXT)]
    pos = neg = 0
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        if os.path.isfile(lp) and os.path.getsize(lp) > 0:
            pos += 1
        else:
            neg += 1                        # empty or missing label = background
    return pos, neg


def parse_args():
    ap = argparse.ArgumentParser(description="Precision-focused cigarette trainer")
    ap.add_argument("--data", default="training/data.yaml")
    ap.add_argument("--add-negatives", default=None,
                    help="folder of finger/pen/straw photos (NO cigarettes) to import, then exit")
    ap.add_argument("--model", default="yolo11m.pt",
                    help="yolo11s.pt (fast) | yolo11m.pt (recommended) | yolo11l.pt | best.pt (continue)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="high res helps tiny cigarettes; 960 if GPU is limited")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="cig_precision")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-sweep", action="store_true", help="skip the confidence sweep")
    return ap.parse_args()


def main():
    args = parse_args()

    # Mode: just import negatives and stop (no ML deps needed)
    if args.add_negatives:
        add_negatives(args.add_negatives, args.data)
        return

    # Heavy deps only needed for actual training
    from ultralytics import YOLO
    try:
        import torch
        CUDA = torch.cuda.is_available()
    except Exception:
        CUDA = False

    device = args.device if args.device is not None else (0 if CUDA else "cpu")
    cfg, img_dir, lbl_dir = paths_from_yaml(args.data)
    pos, neg = count_pos_neg(img_dir, lbl_dir)

    print("=" * 70)
    print(" PRECISION-FOCUSED CIGARETTE TRAINING")
    print("=" * 70)
    print(f" train positives (with cigarettes): {pos}")
    print(f" train negatives (background)      : {neg}")
    ratio = neg / max(pos, 1)
    if neg == 0:
        print("\n  ⚠  ZERO negatives. THIS is why fingers/pens are detected as cigarettes.")
        print("     Add hard negatives:  python training/train.py --add-negatives ./my_negatives")
        print("     Aim for negatives ≈ 15-30% of positives.\n")
    elif ratio < 0.10:
        print(f"\n  ⚠  Only {ratio*100:.0f}% negatives — add more finger/pen/straw images"
              f" to cut false positives.\n")
    else:
        print(f"  ✓ negatives ≈ {ratio*100:.0f}% of positives — good for suppressing false alarms")

    print(f"\n base model : {args.model}   imgsz: {args.imgsz}   epochs: {args.epochs}")
    print(f" device     : {'GPU '+str(device) if CUDA else 'CPU (use a GPU for real training)'}")
    print("=" * 70)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        name=args.name,
        patience=args.patience,

        optimizer="auto",
        cos_lr=True,
        lr0=0.01,
        warmup_epochs=3.0,

        # precision-oriented loss weighting: emphasise correct classification
        # and tight boxes so weak/ambiguous narrow blobs are less likely to pass
        box=7.5,
        cls=1.0,          # raised from default 0.5 -> stronger "is this really the class?"
        dfl=1.5,

        # augmentation: keep it varied but not so wild that thin shapes get invented
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=15,   # disable mosaic for the last 15 epochs
        mixup=0.05, copy_paste=0.05,

        plots=True, val=True,
    )

    print("\n[validation] best checkpoint metrics...")
    m = model.val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"  mAP50    : {m.box.map50:.4f}   mAP50-95: {m.box.map:.4f}")

    # ── Confidence sweep: recommend the CIG_CONF that keeps precision high ──
    if not args.no_sweep:
        print("\n[confidence sweep] finding a threshold that minimises false positives")
        print("  conf   precision  recall     F1")
        best = None
        for conf in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
            try:
                mv = model.val(data=args.data, imgsz=args.imgsz, device=device,
                               conf=conf, verbose=False)
                p, r = float(mv.box.mp), float(mv.box.mr)
                f1 = 2 * p * r / (p + r + 1e-9)
                print(f"  {conf:.2f}   {p:0.3f}      {r:0.3f}     {f1:0.3f}")
                # prefer precision >= 0.90 with the highest recall; else best F1
                key = (p >= 0.90, r if p >= 0.90 else 0.0, f1)
                if best is None or key > best[0]:
                    best = (key, conf, p, r, f1)
            except Exception as e:
                print(f"  {conf:.2f}   (sweep step failed: {e})")
        if best:
            _, conf, p, r, f1 = best
            print("\n  ➜ RECOMMENDED  CIG_CONF: %.2f   (precision %.2f, recall %.2f)" % (conf, p, r))
            print("    Set this in config.yaml under detection.cig_conf to stop weak false positives.")

    best_pt = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    if os.path.isfile(best_pt):
        shutil.copy(best_pt, "best.pt")
        print(f"\n  Copied {best_pt} -> ./best.pt")
    print("=" * 70)
    print("\nIf it STILL fires on fingers after this:")
    print("  1) add MORE hard negatives of exactly the objects it confuses")
    print("     (photograph the real fingers/pens in your camera's setting)")
    print("  2) raise detection.cig_conf in config.yaml toward the recommended value")
    print("  3) the live app already requires 5-of-8 frames + person association,")
    print("     which suppresses one-off false positives — keep that on")


if __name__ == "__main__":
    main()

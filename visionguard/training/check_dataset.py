"""
check_dataset.py — verify a YOLO dataset BEFORE training.

Bad data is the #1 cause of a weak model. This checks the common problems that
quietly destroy accuracy: images with no label file, label files with no image,
empty labels, class indices out of range, and malformed lines. Run it until it
reports 0 problems.

    python training/check_dataset.py --data training/data.yaml
"""
import argparse
import os
import glob
import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def check_split(img_dir, lbl_dir, nc):
    problems = []
    imgs = [p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(IMG_EXT)]
    lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))
    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in lbls}

    n_boxes = 0
    empty = 0

    # images without a label (fine only if intentional "background" images)
    for stem in img_stems - lbl_stems:
        problems.append(f"image without label: {stem}")

    # labels without an image (always a mistake)
    for stem in lbl_stems - img_stems:
        problems.append(f"label without image: {stem}")

    # validate label contents
    for lp in lbls:
        with open(lp) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            empty += 1
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                problems.append(f"{os.path.basename(lp)}: bad line '{ln}' (need 5 values)")
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                problems.append(f"{os.path.basename(lp)}: non-numeric '{ln}'")
                continue
            if not (0 <= cls < nc):
                problems.append(f"{os.path.basename(lp)}: class {cls} out of range 0..{nc-1}")
            if any(not (0.0 <= v <= 1.0) for v in vals):
                problems.append(f"{os.path.basename(lp)}: coords not normalized 0..1 '{ln}'")
            n_boxes += 1

    return {"images": len(imgs), "labels": len(lbls), "boxes": n_boxes,
            "empty_labels": empty, "problems": problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data.yaml")
    args = ap.parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("path", ".")
    nc = int(cfg.get("nc", 1))

    total_problems = 0
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            print(f"[{split}] not defined in data.yaml — skipping")
            continue
        img_dir = os.path.join(root, rel)
        lbl_dir = os.path.join(root, rel.replace("images", "labels"))
        if not os.path.isdir(img_dir):
            print(f"[{split}] image dir missing: {img_dir}")
            total_problems += 1
            continue
        r = check_split(img_dir, lbl_dir, nc)
        print(f"\n[{split}]  images={r['images']}  labels={r['labels']}  "
              f"boxes={r['boxes']}  empty_labels={r['empty_labels']}")
        for p in r["problems"][:25]:
            print("   -", p)
        if len(r["problems"]) > 25:
            print(f"   ... and {len(r['problems']) - 25} more")
        total_problems += len(r["problems"])

    print("\n" + "=" * 50)
    if total_problems == 0:
        print("✓ dataset looks clean — ready to train")
    else:
        print(f"✗ {total_problems} problem(s) found — fix these before training")


if __name__ == "__main__":
    main()

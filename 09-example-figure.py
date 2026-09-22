#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 09 (optional) - Example figure like Fig. 1 of the paper.

Builds a grid with one column group per scene class (Flood, Landslide,
Mudslide - taken from the ERA backgrounds) and three rows:
  Original   the curated background
  Combine    the synthetic image produced by 05-combine.py
  Annotated  the same image with the YOLO label boxes (real + pasted people)

Usage
  python 09-example-figure.py                      # out/diagram.png
  python 09-example-figure.py --per-class 2 --cell 320 --seed 3
  python 09-example-figure.py --weights out/runs/yolo11n_v4/weights/best.pt
      # "Annotated" row shows detector predictions with confidence instead
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config as C
from pipeline_utils import read_csv, read_yolo_labels

CLASSES = ["Flood", "Landslide", "Mudslide"]
ROWS = ["Original", "Combine", "Annotated"]


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default(size=size)


def fit(img: Image.Image, cell: int) -> Image.Image:
    """Center-crop to square then resize to cell x cell."""
    s = min(img.size)
    l, t = (img.width - s) // 2, (img.height - s) // 2
    return img.crop((l, t, l + s, t + s)).resize((cell, cell), Image.LANCZOS), (l, t, s)


def annotate(img: Image.Image, boxes, color, labels=None):
    """Boxes (and optional confidence text) with stroke/font scaled to the image."""
    d = ImageDraw.Draw(img)
    width = max(1, img.width // 320)
    fnt = font(max(10, img.width // 45))
    for i, b in enumerate(boxes):
        d.rectangle(list(b), outline=color, width=width)
        if labels:
            tw = d.textlength(labels[i], font=fnt)
            th = fnt.size + 2
            x, y = b[0], b[1] - th if b[1] - th >= 0 else b[3]
            d.rectangle((x, y, x + tw + 2, y + th), fill=color)
            d.text((x + 1, y), labels[i], fill="white", font=fnt)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--cell", type=int, default=420, help="cell size in px")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--weights", default=None, help="draw detector predictions (with confidence) instead of labels")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(C.OUT_DIR / "diagram.png"))
    args = ap.parse_args()
    random.seed(args.seed)

    rows = read_csv(C.OUT_MANIFEST)
    picks = {}
    for cls in CLASSES:
        cand = [r for r in rows if r["background"].startswith(f"ERA_{cls}_") and int(r["n_pasted"]) > 0]
        # one synthetic image per distinct background, prefer shots that also contain real people
        by_bg = {}
        for r in cand:
            by_bg.setdefault(r["background"], r)
        cand = list(by_bg.values())
        random.shuffle(cand)
        cand.sort(key=lambda r: -int(r["n_original"]) if random.random() < 0.5 else 0)
        picks[cls] = cand[: args.per_class]
        if len(picks[cls]) < args.per_class:
            raise SystemExit(f"not enough generated images for {cls}")

    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    cell, gap, pad = args.cell, 4, 6
    header_h, label_w = 34, 34
    n_cols = len(CLASSES) * args.per_class
    W = label_w + n_cols * (cell + gap) + (len(CLASSES) - 1) * pad
    H = header_h + len(ROWS) * (cell + gap)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_head = font(max(18, cell // 16))

    for ci, cls in enumerate(CLASSES):
        x0 = label_w + ci * (args.per_class * (cell + gap) + pad)
        x1 = x0 + args.per_class * (cell + gap) - gap
        d.rectangle((x0, 0, x1, header_h - 4), fill=(235, 235, 235))
        tw = d.textlength(cls, font=f_head)
        d.text(((x0 + x1 - tw) / 2, 7), cls, fill="black", font=f_head)

        for k, r in enumerate(picks[cls]):
            x = x0 + k * (cell + gap)
            orig = Image.open(C.BG_IMAGES / r["background"]).convert("RGB")
            comb = Image.open(C.OUT_IMAGES / r["image"]).convert("RGB")
            w, h = comb.size
            if model is not None:
                res = model.predict(comb, conf=args.conf, imgsz=C.TRAIN["imgsz"], verbose=False)[0]
                boxes = [tuple(int(round(v)) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
                labels = [f"{c:.2f}" for c in res.boxes.conf.cpu().numpy()]
            else:
                _, boxes = read_yolo_labels(C.OUT_LABELS / r["image"].replace(".jpg", ".txt"), w, h)
                labels = None
            ann = annotate(comb.copy(), boxes, (255, 0, 0), labels)
            _, gt = read_yolo_labels(C.OUT_LABELS / r["image"].replace(".jpg", ".txt"), w, h)
            r["_summary"] = f"{len(boxes)} detections / {len(gt)} labels" if model else f"{len(gt)} labels"

            for ri, img in enumerate((orig.resize((w, h)), comb, ann)):
                y = header_h + ri * (cell + gap)
                tile, _ = fit(img, cell)
                canvas.paste(tile, (x, y))

    for ri, name in enumerate(ROWS):
        y = header_h + ri * (cell + gap)
        lab = Image.new("RGB", (cell, label_w - 6), (235, 235, 235))
        ld = ImageDraw.Draw(lab)
        tw = ld.textlength(name, font=f_head)
        ld.text(((cell - tw) / 2, 4), name, fill="black", font=f_head)
        canvas.paste(lab.rotate(90, expand=True), (0, y))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"{args.out}  ({W}x{H})")
    for cls in CLASSES:
        for r in picks[cls]:
            print(f"  {cls:9s} {r['background']:24s} -> {r['image']}  real={r['n_original']} "
                  f"pasted={r['n_pasted']}  {r['_summary']}")


if __name__ == "__main__":
    main()

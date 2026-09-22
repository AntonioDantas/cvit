#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 03 - Annotate people already present in the backgrounds (paper
Sec. 3.1.3, "The final step is to annotate the images that contain people
already identified in the background ...").

Some aerial disaster shots already contain rescuers or victims. Instead of
discarding them, their boxes are kept as ground truth and the combination
step (05) treats them as occupied space, so pasted figures never overlap
them and both real and synthetic people end up in the same label file.

Input : backgrounds/images/*
Output: backgrounds/labels/<stem>.txt   YOLO "0 cx cy w h" (only when people found)
        backgrounds/annotation_report.csv  one row per image (n_persons)
        backgrounds/preview/labels/*       optional boxes drawn on N images

Detector: Ultralytics YOLOv8x (COCO) restricted to class 0 = person,
confidence >= BG_PERSON_MIN_CONF, inference at BG_PERSON_IMGSZ.
The script is resumable (images in the report are skipped).
"""

import argparse

from PIL import Image
from tqdm import tqdm

import config as C
from pipeline_utils import (append_csv, draw_boxes, list_images, read_csv,
                            torch_device, xyxy_to_yolo, yolo_weights)

FIELDS = ["name", "n_persons", "max_conf"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=C.BG_PERSON_MIN_CONF)
    ap.add_argument("--imgsz", type=int, default=C.BG_PERSON_IMGSZ)
    ap.add_argument("--preview", type=int, default=30, help="draw boxes for the first N annotated images")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch_device()
    model = yolo_weights(C.YOLO_COCO_WEIGHTS)
    print(f"{C.YOLO_COCO_WEIGHTS} on {device}, conf>={args.conf}, imgsz={args.imgsz}")

    C.BG_LABELS.mkdir(parents=True, exist_ok=True)
    report = C.BG_DIR / "annotation_report.csv"
    done = {r["name"] for r in read_csv(report)}
    images = list_images(C.BG_IMAGES)[: args.limit]
    todo = [p for p in images if p.name not in done]
    print(f"{len(images)} backgrounds, {len(done)} already annotated, {len(todo)} to process")

    previews = 0
    for p in tqdm(todo, desc="Annotating"):
        res = model.predict(str(p), classes=[0], conf=args.conf, imgsz=args.imgsz,
                            device=device, verbose=False)[0]
        w, h = res.orig_shape[1], res.orig_shape[0]
        boxes = [tuple(int(round(v)) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
        confs = res.boxes.conf.cpu().numpy().tolist()

        lbl = C.BG_LABELS / f"{p.stem}.txt"
        if boxes:
            lbl.write_text("\n".join(xyxy_to_yolo(0, b, w, h) for b in boxes) + "\n", encoding="utf-8")
            if previews < args.preview:
                dst = C.BG_PREVIEW / "labels" / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                draw_boxes(Image.open(p), boxes, color=(0, 128, 255), width=3).save(dst, quality=90)
                previews += 1
        elif lbl.exists():
            lbl.unlink()
        append_csv(report, dict(name=p.name, n_persons=len(boxes),
                                max_conf=f"{max(confs):.3f}" if confs else ""), FIELDS)

    rows = read_csv(report)
    with_people = [r for r in rows if int(r["n_persons"]) > 0]
    total = sum(int(r["n_persons"]) for r in rows)
    print(f"\n{len(rows)} backgrounds, {len(with_people)} contain people, "
          f"{total} real person boxes reused as labels")
    print(f"Labels : {C.BG_LABELS}")
    print(f"Report : {report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 04 - Extraction and treatment of human figures (paper Sec. 3.1.2).

Input : persons/raw/*.jpg   Multiple Poses Human Body (MPHB) dataset,
                            26,675 images of bent / kneeling / sitting / standing / lying people.
Output: persons/valid/<stem>.png   transparent, tightly cropped cut-outs
        persons/report.csv         one row per raw image (status, conf, size)

Two automated sub-steps, "ensuring that only clippings containing visible and intact humans were passed to the combination stage":

  1. Background removal - rembg with the U2-Net "u2net_human_seg" model
     isolates the central, highlighted person; the alpha channel is then trimmed to its bounding box (as corte.py did). Frames are shrunk to PERSON_MAX_SIDE_IN px first (U2-Net infers at 320 px; the final pasted patches are <= 81 px, so nothing is lost).
  2. Identification and clipping - the cut-out is composited over white and
     checked with a COCO person detector (YOLOv8x). It is accepted only if
        - at least PERSON_MIN_VISIBLE_PIXELS pixels are opaque,
        - at least PERSON_MIN_TRANSPARENT_FRAC of the pixels are fully
          transparent (i.e. the background was really removed),
        - a person is detected with confidence >= PERSON_MIN_CONF (0.50),
        - and no other COCO class is detected (only people in the patch).
     The final patch is re-cropped to the detected box.

Resumable: images already in persons/report.csv are skipped.
Speed    : ~0.4 s/image on an Apple M4 (rembg on CoreML, detector batched
           on MPS); the whole MPHB takes ~3 h. Use --limit for a quick check.
"""

import argparse
import os
from collections import Counter

import numpy as np
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

import config as C
from pipeline_utils import append_csv, list_images, read_csv, torch_device, yolo_weights

FIELDS = ["name", "status", "conf", "width", "height"]


def shrink(img: Image.Image, max_side: int) -> Image.Image:
    s = max_side / max(img.size)
    if s >= 1:
        return img
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)


def remove_background(img: Image.Image, session) -> Image.Image | None:
    """rembg + trim to the non-transparent bounding box."""
    rgba = remove(shrink(img.convert("RGBA"), C.PERSON_MAX_SIDE_IN), session=session)
    bbox = rgba.getchannel("A").getbbox()
    return rgba.crop(bbox) if bbox else None


def alpha_checks(rgba: Image.Image) -> str | None:
    a = np.asarray(rgba)[:, :, 3]
    if int((a > 0).sum()) < C.PERSON_MIN_VISIBLE_PIXELS:
        return "too_small"
    if float((a == 0).mean()) < C.PERSON_MIN_TRANSPARENT_FRAC:
        return "no_transparency"
    return None


def on_white(rgba: Image.Image) -> Image.Image:
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    return white.convert("RGB")


def judge(res):
    """(status, conf, box) from one Ultralytics result."""
    if len(res.boxes) == 0:
        return "no_person", 0.0, None
    cls = res.boxes.cls.cpu().numpy().astype(int)
    if (cls != 0).any():
        return "other_objects", 0.0, None
    confs = res.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    box = tuple(int(round(v)) for v in res.boxes.xyxy[best].cpu().numpy())
    return "ok", float(confs[best]), box


def flush(batch, model, device, conf):
    """Run the detector on a batch of (path, rgba) and write results."""
    if not batch:
        return
    results = model.predict([on_white(r) for _, r in batch], conf=conf, device=device, verbose=False)
    for (p, rgba), res in zip(batch, results):
        status, score, box = judge(res)
        row = dict(name=p.name, status=status, conf="", width="", height="")
        if status == "ok":
            patch = rgba.crop(box)
            patch.save(C.PERSONS_VALID / f"{p.stem}.png", format="PNG")
            row.update(conf=f"{score:.3f}", width=patch.width, height=patch.height)
        append_csv(C.PERSONS_REPORT, row, FIELDS)
    batch.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=C.PERSON_MIN_CONF)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N raw images")
    args = ap.parse_args()

    device = torch_device()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", str(C.MODELS_DIR / "rembg"))
    session = new_session(C.REMBG_MODEL, providers=C.REMBG_PROVIDERS)
    model = yolo_weights(C.YOLO_COCO_WEIGHTS)
    print(f"rembg={C.REMBG_MODEL} ({session.inner_session.get_providers()[0]})  "
          f"detector={C.YOLO_COCO_WEIGHTS} on {device}  conf>={args.conf}")

    C.PERSONS_VALID.mkdir(parents=True, exist_ok=True)
    done = {r["name"] for r in read_csv(C.PERSONS_REPORT)}
    raws = list_images(C.PERSONS_RAW)[: args.limit]
    todo = [p for p in raws if p.name not in done]
    print(f"{len(raws)} raw images, {len(done)} already processed, {len(todo)} to go")

    batch = []
    for p in tqdm(todo, desc="Persons"):
        try:
            rgba = remove_background(Image.open(p), session)
        except Exception as e:
            tqdm.write(f"[ERR] {p.name}: {e}")
            append_csv(C.PERSONS_REPORT, dict(name=p.name, status="error", conf="", width="", height=""), FIELDS)
            continue
        early = "empty_mask" if rgba is None else alpha_checks(rgba)
        if early:
            append_csv(C.PERSONS_REPORT, dict(name=p.name, status=early, conf="", width="", height=""), FIELDS)
            continue
        batch.append((p, rgba))
        if len(batch) >= C.PERSON_BATCH:
            flush(batch, model, device, args.conf)
    flush(batch, model, device, args.conf)

    rows = read_csv(C.PERSONS_REPORT)
    counts = Counter(r["status"] for r in rows)
    print(f"\n{len(rows)} processed:")
    for k, v in counts.most_common():
        print(f"  {k:16s} {v}")
    print(f"Valid cut-outs : {C.PERSONS_VALID}  ({counts.get('ok', 0)} png)")
    print(f"Report         : {C.PERSONS_REPORT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 05 - Combination strategy: cluster-based C2A synthesis (paper Sec. 3.1.3
and 4.1).

Input : backgrounds/images/*      curated aerial disaster shots (step 01)
        backgrounds/cloud.txt     forbidden sky band per image (step 02)
        backgrounds/labels/*.txt  people ALREADY present in the shot (step 03)
        persons/valid/*.png       validated transparent cut-outs (step 04)
Output: out/images/img_XXXXXX.jpg   synthetic images
        out/labels/img_XXXXXX.txt   YOLO labels = original people + pasted people
        out/manifest.csv            provenance of every generated image
        out/debug/                  triples (background / composite / boxes)
                                    for the first --debug-samples images;
                                    blue boxes = real people kept from the
                                    background, red boxes = pasted instances

For every background, `variants_per_bg` images are produced. Each one starts
from the ORIGINAL label boxes of the background (they are copied to the new
label file and treated as occupied space), then new instances are inserted:

  mode "fill"     (v1, v2)  grid over the whole allowed area followed by a
                            random fill up to max_per_bg (~100 / image).
  mode "clusters" (v3-v5)   N ~ U[clusters] rectangles (10-50 % of the
                            image) are drawn under the sky line; each has an
                            internal rows x cols grid and every cell receives
                            1..max_per_cell figures, with a per-image cap
                            max_per_bg (~30 / image) so the budget is spread
                            over the clusters.

Backgrounds larger than BG_MAX_SIDE (1280 px) are shrunk first (training
uses 640 px). Every pasted figure is scaled to 1-4 % of the shorter side, rotated in
[-90, 90] and - when `augment` is on - may receive side crop (occlusion),
brightness matching, blur, noise and JPEG compression (Sec. 4.1).

Usage
  python 05-combine.py                      # v4 (published)
  python 05-combine.py --version v1         # any preset in config.VERSIONS
  python 05-combine.py --limit 30 --clean   # quick check on 30 backgrounds
"""

import argparse
import csv
import io
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat
from tqdm import tqdm

import config as C
from pipeline_utils import (draw_boxes, list_images, read_cloud_file,
                            read_yolo_labels, xyxy_to_yolo, yolo_to_xyxy)

Box = Tuple[int, int, int, int]


# ------------------------------------------------------------ geometry
def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps_any(box: Box, others: List[Box], margin: int) -> bool:
    nb = (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)
    return any(iou(nb, (o[0] - margin, o[1] - margin, o[2] + margin, o[3] + margin)) > 0 for o in others)


def downscale_max(im: Image.Image, max_side: int) -> Image.Image:
    if max(im.size) <= max_side:
        return im
    r = max_side / max(im.size)
    return im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.BICUBIC)


def scaled_size(bg_w, bg_h, pw, ph, cfg) -> Tuple[int, int]:
    """Target size so that the patch's shorter side is 1-4 % of the shorter background side."""
    target = max(2, int(min(bg_w, bg_h) * random.uniform(cfg["min_scale"], cfg["max_scale"])))
    aspect = pw / ph
    if aspect >= 1:
        return target, max(2, int(target / aspect))
    return max(2, int(target * aspect)), target


def try_place(bg_w, bg_h, person: Image.Image, placed: List[Box], cfg, y_min: int,
              region: Box | None, attempts: int = 40):
    """Scale + rotate the patch and find a free spot inside `region` (or the whole allowed area)."""
    for _ in range(attempts):
        tw, th = scaled_size(bg_w, bg_h, person.width, person.height, cfg)
        patch = person.resize((tw, th), Image.BICUBIC)
        patch = patch.rotate(random.uniform(-cfg["max_rotation"], cfg["max_rotation"]),
                             expand=True, resample=Image.BICUBIC)
        rw, rh = patch.size
        if rw >= bg_w or rh >= bg_h:
            continue
        if region is None:
            x0, x1, y0, y1 = 0, bg_w - rw, y_min, bg_h - rh
        else:
            x0, x1 = max(0, region[0]), min(bg_w - rw, region[2] - rw)
            y0, y1 = max(y_min, region[1]), min(bg_h - rh, region[3] - rh)
        if x1 < x0 or y1 < y0:
            continue
        x, y = random.randint(x0, x1), random.randint(y0, y1)
        box = (x, y, x + rw, y + rh)
        if not overlaps_any(box, placed, cfg["margin"]):
            return patch, box
    return None


# ------------------------------------------------- instance augmentation
def with_alpha(orig: Image.Image, rgb: Image.Image) -> Image.Image:
    r, g, b = rgb.resize(orig.size, Image.BICUBIC).split()
    return Image.merge("RGBA", (r, g, b, orig.getchannel("A")))


def side_crop(p: Image.Image, max_frac: float) -> Image.Image:
    w, h = p.size
    side = random.choice(["left", "right", "top", "bottom"])
    n = int((w if side in ("left", "right") else h) * max_frac)
    if n < 1:
        return p
    cut = random.randint(1, n)
    boxes = {"left": (cut, 0, w, h), "right": (0, 0, w - cut, h),
             "top": (0, cut, w, h), "bottom": (0, 0, w, h - cut)}
    b = boxes[side]
    return p.crop(b) if b[2] - b[0] >= 2 and b[3] - b[1] >= 2 else p


def match_brightness(p: Image.Image, canvas: Image.Image, box: Box, cfg) -> Image.Image:
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = max(x1 + 1, min(canvas.width, box[2])), max(y1 + 1, min(canvas.height, box[3]))
    lum_bg = sum(ImageStat.Stat(canvas.crop((x1, y1, x2, y2)).convert("RGB")).mean) / 3
    rgb = p.convert("RGB")
    lum_p = sum(ImageStat.Stat(rgb).mean) / 3
    if lum_p < 1e-3:
        return p
    f = 1 + cfg["color_match_strength"] * (lum_bg / lum_p - 1)
    f = max(cfg["color_match_min_factor"], min(cfg["color_match_max_factor"], f))
    return with_alpha(p, ImageEnhance.Brightness(rgb).enhance(f))


def add_noise(p: Image.Image, var: float, alpha: float) -> Image.Image:
    n = Image.effect_noise(p.size, var)
    return with_alpha(p, ImageChops.blend(p.convert("RGB"), Image.merge("RGB", (n, n, n)), alpha))


def jpeg(p: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    p.convert("RGB").save(buf, format="JPEG", quality=max(1, min(95, quality)))
    buf.seek(0)
    return with_alpha(p, Image.open(buf).convert("RGB"))


def augment(p: Image.Image, box: Box, canvas: Image.Image, cfg) -> Tuple[Image.Image, Box]:
    x1, y1 = box[0], box[1]
    if random.random() < cfg["random_crop_prob"]:
        p = side_crop(p, cfg["random_crop_max_fraction"])
    box = (x1, y1, x1 + p.width, y1 + p.height)
    if random.random() < cfg["color_match_prob"]:
        p = match_brightness(p, canvas, box, cfg)
    if random.random() < cfg["blur_prob"]:
        p = p.filter(ImageFilter.GaussianBlur(random.uniform(cfg["blur_radius_min"], cfg["blur_radius_max"])))
    if random.random() < cfg["noise_prob"]:
        p = add_noise(p, random.uniform(cfg["noise_variance_min"], cfg["noise_variance_max"]),
                      random.uniform(cfg["noise_alpha_min"], cfg["noise_alpha_max"]))
    if random.random() < cfg["compression_prob"]:
        p = jpeg(p, random.randint(cfg["compression_quality_min"], cfg["compression_quality_max"]))
    return p, (x1, y1, x1 + p.width, y1 + p.height)


# --------------------------------------------------------- placement
class PersonPool:
    """Cycles through the cut-outs in random order without repeats within a cycle."""

    def __init__(self, paths: List[Path]):
        self.paths, self.pool = paths, []

    def next(self) -> Image.Image | None:
        while True:
            if not self.pool:
                self.pool = self.paths[:]
                random.shuffle(self.pool)
            try:
                return downscale_max(Image.open(self.pool.pop()).convert("RGBA"), C.AUGMENT["person_max_side"])
            except Exception:
                continue


class Composer:
    def __init__(self, bg: Image.Image, orig_boxes: List[Box], y_min: int, cfg, pool: PersonPool):
        self.canvas = bg.copy()
        self.w, self.h = bg.size
        self.placed = list(orig_boxes)
        self.new_boxes: List[Box] = []
        self.y_min, self.cfg, self.pool = y_min, cfg, pool

    def paste(self, region: Box | None) -> bool:
        person = self.pool.next()
        hit = try_place(self.w, self.h, person, self.placed, self.cfg, self.y_min, region)
        if hit is None:
            return False
        patch, box = hit
        if self.cfg["augment"]:
            patch, box = augment(patch, box, self.canvas, self.cfg)
        self.canvas.alpha_composite(patch, dest=(box[0], box[1]))
        self.placed.append(box)
        self.new_boxes.append(box)
        return True

    def grid_cells(self, rect: Box, rows: int, cols: int) -> List[Box]:
        x1, y1, x2, y2 = rect
        cw, ch = (x2 - x1) / cols, (y2 - y1) / rows
        cells = [(int(x1 + c * cw), int(y1 + r * ch), int(x1 + (c + 1) * cw), int(y1 + (r + 1) * ch))
                 for r in range(rows) for c in range(cols)]
        return [c for c in cells if c[2] - c[0] >= 4 and c[3] - c[1] >= 4]

    def fill_cells(self, cells: List[Box], budget: int) -> int:
        """1..max_per_cell figures per cell, cells in random order, until budget is spent."""
        random.shuffle(cells)
        n = 0
        for cell in cells:
            for _ in range(random.randint(1, self.cfg["max_per_cell"])):
                if n >= budget:
                    return n
                n += self.paste(cell)
        return n

    def run_fill(self) -> int:
        rows, cols = self.cfg["grid"]
        budget = self.cfg["max_per_bg"]
        n = self.fill_cells(self.grid_cells((0, self.y_min, self.w, self.h), rows, cols), budget)
        tries = 0
        while n < budget and tries < budget * 20:
            tries += 1
            n += self.paste(None)
        return n

    def cluster_rects(self) -> List[Box]:
        cfg = self.cfg
        n = random.randint(*cfg["clusters"])
        allowed_h = self.h - self.y_min
        rects: List[Box] = []
        for _ in range(n * 5):
            if len(rects) >= n:
                break
            cw = int(self.w * random.uniform(cfg["cluster_min_width_frac"], cfg["cluster_max_width_frac"]))
            ch = int(allowed_h * random.uniform(cfg["cluster_min_height_frac"], cfg["cluster_max_height_frac"]))
            if cw < 8 or ch < 8 or self.w - cw <= 0 or self.h - ch < self.y_min:
                continue
            x1, y1 = random.randint(0, self.w - cw), random.randint(self.y_min, self.h - ch)
            rect = (x1, y1, x1 + cw, y1 + ch)
            if all(iou(rect, r) <= cfg["cluster_max_iou"] for r in rects):
                rects.append(rect)
        return rects

    def run_clusters(self) -> int:
        rows, cols = self.cfg["grid"]
        cells = [c for rect in self.cluster_rects() for c in self.grid_cells(rect, rows, cols)]
        return self.fill_cells(cells, self.cfg["max_per_bg"])


# ------------------------------------------------------------- main
def next_index(out_images: Path) -> int:
    idx = [int(p.stem.split("_")[1]) for p in out_images.glob("img_*.jpg")]
    return max(idx) if idx else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=C.DEFAULT_VERSION, choices=sorted(C.VERSIONS))
    ap.add_argument("--variants", type=int, default=None, help="images per background (default from config)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N backgrounds")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--debug-samples", type=int, default=30, help="save debug triples for the first N images")
    ap.add_argument("--clean", action="store_true", help="delete out/images, out/labels, out/debug first")
    args = ap.parse_args()

    cfg = C.combine_config(args.version)
    if args.variants:
        cfg["variants_per_bg"] = args.variants
    random.seed(args.seed)

    bgs = list_images(C.BG_IMAGES)[: args.limit]
    people = list_images(C.PERSONS_VALID)
    if not bgs or not people:
        raise SystemExit("run steps 01-04 first: no backgrounds or no validated persons found")
    cloud = read_cloud_file(C.CLOUD_FILE)
    pool = PersonPool(people)

    if args.clean:
        for d in (C.OUT_IMAGES, C.OUT_LABELS, C.OUT_DEBUG):
            shutil.rmtree(d, ignore_errors=True)
        C.OUT_MANIFEST.unlink(missing_ok=True)
    for d in (C.OUT_IMAGES, C.OUT_LABELS, C.OUT_DEBUG):
        d.mkdir(parents=True, exist_ok=True)

    print(f"version {args.version}: {cfg['note']}")
    print(f"  mode={cfg['mode']} clusters={cfg['clusters']} grid={cfg['grid']} "
          f"max_per_cell={cfg['max_per_cell']} max_per_bg={cfg['max_per_bg']} augment={cfg['augment']}")
    print(f"  {len(bgs)} backgrounds x {cfg['variants_per_bg']} variants, {len(people)} cut-outs, "
          f"sky band for {len(cloud)} images")

    idx = next_index(C.OUT_IMAGES)
    new_manifest = not C.OUT_MANIFEST.exists()
    fields = ["image", "background", "version", "variant", "n_original", "n_pasted", "sky_px"]
    fm = C.OUT_MANIFEST.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fm, fieldnames=fields)
    if new_manifest:
        writer.writeheader()

    debug_left = args.debug_samples
    tot_orig = tot_new = 0
    for bg_path in tqdm(bgs, desc=f"Combining {args.version}"):
        try:
            bg = Image.open(bg_path).convert("RGBA")
        except Exception as e:
            tqdm.write(f"[ERR] {bg_path.name}: {e}")
            continue
        orig_h = bg.height
        bg = downscale_max(bg, C.BG_MAX_SIDE)
        w, h = bg.size
        sky_px = cloud.get(bg_path.name)
        y_min = int(h * C.DEFAULT_TOP_EXCLUSION) if sky_px is None else min(h, round(sky_px * h / orig_h))
        orig_lines, orig_boxes = read_yolo_labels(C.BG_LABELS / f"{bg_path.stem}.txt", w, h)

        n_variants = cfg["variants_per_bg"] + cfg["empty_variants_per_bg"]
        for v in range(n_variants):
            comp = Composer(bg, orig_boxes, y_min, cfg, pool)
            if v < cfg["variants_per_bg"]:
                n_new = comp.run_clusters() if cfg["mode"] == "clusters" else comp.run_fill()
            else:
                n_new = 0  # negative / background-only variant
            lines = list(orig_lines) + [xyxy_to_yolo(0, b, w, h) for b in comp.new_boxes]

            idx += 1
            base = f"img_{idx:06d}"
            rgb = comp.canvas.convert("RGB")
            rgb.save(C.OUT_IMAGES / f"{base}.jpg", quality=95)
            (C.OUT_LABELS / f"{base}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            writer.writerow(dict(image=f"{base}.jpg", background=bg_path.name, version=args.version,
                                 variant=v, n_original=len(orig_boxes), n_pasted=n_new, sky_px=y_min))
            tot_orig += len(orig_boxes)
            tot_new += n_new

            if debug_left > 0:
                debug_left -= 1
                bg.convert("RGB").save(C.OUT_DEBUG / f"{base}_1_background.jpg", quality=90)
                rgb.save(C.OUT_DEBUG / f"{base}_2_composite.jpg", quality=90)
                vis = draw_boxes(rgb, orig_boxes, color=(0, 128, 255), width=2)
                vis = draw_boxes(vis, comp.new_boxes, color=(255, 0, 0), width=2)
                vis.save(C.OUT_DEBUG / f"{base}_3_boxes.jpg", quality=90)
        fm.flush()
    fm.close()

    n_imgs = len(list(C.OUT_IMAGES.glob("*.jpg")))
    print(f"\nGenerated up to img_{idx:06d}: {n_imgs} images in total")
    print(f"  this run: {tot_orig} real boxes reused + {tot_new} pasted instances "
          f"= {tot_orig + tot_new} labels")
    print(f"Images   : {C.OUT_IMAGES}")
    print(f"Labels   : {C.OUT_LABELS}")
    print(f"Manifest : {C.OUT_MANIFEST}")
    print(f"Debug    : {C.OUT_DEBUG}")


if __name__ == "__main__":
    main()

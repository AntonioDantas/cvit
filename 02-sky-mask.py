#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 02 - Sky exclusion mask (paper Sec. 3.1.1).

"To ensure the physical coherence of the collage (preventing people from
 'floating' in the sky), a semantic segmentation process was applied to
 identify and mask the unusable area."

Input : backgrounds/images/*
Output: backgrounds/cloud.txt      one line per image: "<file> <px>"
                                   px = lowest sky pixel measured from the top
                                   (0 = no sky touching the top band)
        backgrounds/preview/sky/*  optional red overlay of the forbidden band

Model : SegFormer-B0 fine-tuned on ADE20K (class "sky"), run through
        HuggingFace transformers. Weights are cached in models/.

The script is resumable: images already listed in cloud.txt are skipped.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

import config as C
from pipeline_utils import list_images, read_cloud_file, torch_device


def load_segformer(device: str):
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    cache = C.MODELS_DIR / "segformer"
    processor = AutoImageProcessor.from_pretrained(C.SEGFORMER_MODEL, cache_dir=cache)
    model = SegformerForSemanticSegmentation.from_pretrained(C.SEGFORMER_MODEL, cache_dir=cache)
    model.to(device).eval()
    sky_id = int(model.config.label2id["sky"])
    return processor, model, sky_id


@torch.no_grad()
def sky_limit_px(img: Image.Image, processor, model, sky_id: int, device: str) -> int:
    """Lowest row (1-based, from the top) classified as sky, or 0."""
    img = img.convert("RGB")
    w, h = img.size
    # the 150-class logit map is upsampled to image size; cap that size so
    # 20-Mpx photos do not need >10 GB (the result is scaled back to pixels)
    scale = min(1.0, C.SKY_MAX_SIDE / max(w, h))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((sw, sh), Image.BILINEAR) if scale < 1 else img
    inputs = {k: v.to(device) for k, v in processor(images=small, return_tensors="pt").items()}
    logits = model(**inputs).logits
    logits = F.interpolate(logits, size=(sh, sw), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    ys = np.where(pred == sky_id)[0]
    if ys.size == 0 or not np.any(ys <= int(sh * C.SKY_TOP_BAND_FRAC)):
        return 0
    return min(h, int(round((int(ys.max()) + 1) / scale)))


def save_preview(img: Image.Image, px: int, dst: Path):
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle((0, 0, rgba.width, px), fill=(255, 0, 0, 150))
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(rgba, overlay).convert("RGB").save(dst, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=int, default=20, help="save N overlay previews (0 = none)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N images")
    args = ap.parse_args()

    device = torch_device()
    print(f"Loading {C.SEGFORMER_MODEL} on {device} ...")
    processor, model, sky_id = load_segformer(device)

    done = read_cloud_file(C.CLOUD_FILE)
    images = list_images(C.BG_IMAGES)[: args.limit]
    todo = [p for p in images if p.name not in done]
    print(f"{len(images)} backgrounds, {len(done)} already in cloud.txt, {len(todo)} to process")

    C.CLOUD_FILE.parent.mkdir(parents=True, exist_ok=True)
    previews = 0
    with C.CLOUD_FILE.open("a", encoding="utf-8") as f:
        for p in tqdm(todo, desc="Sky mask"):
            try:
                img = Image.open(p)
                px = sky_limit_px(img, processor, model, sky_id, device)
            except Exception as e:
                tqdm.write(f"[ERR] {p.name}: {e}")
                continue
            f.write(f"{p.name} {px}px\n")
            f.flush()
            if previews < args.preview:
                save_preview(img, px, C.BG_PREVIEW / "sky" / p.name)
                previews += 1

    cloud = read_cloud_file(C.CLOUD_FILE)
    with_sky = sum(1 for v in cloud.values() if v > 0)
    print(f"\ncloud.txt: {len(cloud)} images, {with_sky} with a sky band, "
          f"mean band {np.mean(list(cloud.values())):.1f}px")
    print(f"Output : {C.CLOUD_FILE}")


if __name__ == "__main__":
    main()

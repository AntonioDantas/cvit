#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 01 - Background curation (paper Sec. 3.1.1, "Acquisition and Processing
of Background Images").

Input : datasets/<SOURCE>/**  (already manually filtered aerial views of
        hydrological disasters: AIDER, ERA, Alley Flood Net, Kaggle, Roboflow)
Output: backgrounds/images/<SOURCE>_<name>.<ext>   one flat folder, safe names
        backgrounds/manifest.csv                    new name -> original path

What it does
  1. walks every source folder recursively and renames files to a flat,
     shell-safe scheme ("<SOURCE>_<stem>") so cloud.txt / labels match by stem;
  2. removes near-duplicates with a perceptual hash (pHash, Hamming
     distance <= DEDUP_PHASH_THRESHOLD), keeping one image per group, exactly
     as the original duplicate.py did.

Usage
  python 01-prepare-backgrounds.py            # default
  python 01-prepare-backgrounds.py --no-dedup # keep every image
  python 01-prepare-backgrounds.py --clean    # rebuild backgrounds/images
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

import config as C
from pipeline_utils import list_images, safe_name


def collect_sources():
    """Yield (source_tag, path) for every image below datasets/."""
    for src_dir in sorted(p for p in C.DATASETS_DIR.iterdir() if p.is_dir()):
        for img in list_images(src_dir, recursive=True):
            yield src_dir.name.upper(), img


def dedup_groups(hashes: dict, threshold: int):
    """Union-find over pairs whose pHash distance <= threshold."""
    names = list(hashes)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in tqdm(range(len(names)), desc="pHash pairs", leave=False):
        hi = hashes[names[i]]
        for j in range(i + 1, len(names)):
            if hi - hashes[names[j]] <= threshold:
                parent[find(names[i])] = find(names[j])

    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dedup", action="store_true", help="skip perceptual-hash de-duplication")
    ap.add_argument("--threshold", type=int, default=C.DEDUP_PHASH_THRESHOLD, help="pHash Hamming distance")
    ap.add_argument("--clean", action="store_true", help="delete backgrounds/images before running")
    args = ap.parse_args()

    if args.clean and C.BG_IMAGES.exists():
        shutil.rmtree(C.BG_IMAGES)
    C.BG_IMAGES.mkdir(parents=True, exist_ok=True)

    # 1) flat, safe naming ------------------------------------------------
    records, used = [], set()
    for tag, src in tqdm(list(collect_sources()), desc="Scanning sources"):
        try:
            with Image.open(src) as im:
                im.verify()
            with Image.open(src) as im:
                w, h = im.size
                mode = im.mode
                phash = str(imagehash.phash(im.convert("RGB")))
        except Exception as e:
            tqdm.write(f"[SKIP] unreadable {src}: {e}")
            continue

        ext = ".png" if src.suffix.lower() == ".png" else ".jpg"
        base = f"{tag}_{safe_name(src.stem)}"
        name, k = base + ext, 1
        while name in used:
            name, k = f"{base}_{k}{ext}", k + 1
        used.add(name)
        records.append(dict(name=name, source=str(src.relative_to(C.ROOT)), dataset=tag,
                            width=w, height=h, mode=mode, phash=phash, kept=1, dup_of=""))

    # 2) perceptual de-duplication ----------------------------------------
    removed = 0
    if not args.no_dedup:
        hashes = {r["name"]: imagehash.hex_to_hash(r["phash"]) for r in records}
        by_name = {r["name"]: r for r in records}
        for group in dedup_groups(hashes, args.threshold):
            if len(group) < 2:
                continue
            keep = sorted(group)[0]
            for n in group:
                if n != keep:
                    by_name[n]["kept"] = 0
                    by_name[n]["dup_of"] = keep
                    removed += 1

    # 3) copy kept files ----------------------------------------------------
    for r in tqdm(records, desc="Copying"):
        if not r["kept"]:
            continue
        dst = C.BG_IMAGES / r["name"]
        if dst.exists():
            continue
        src = C.ROOT / r["source"]
        if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copy2(src, dst)
        else:  # normalise exotic formats to jpg
            Image.open(src).convert("RGB").save(dst, quality=95)

    with C.BG_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    kept = sum(r["kept"] for r in records)
    per_src = defaultdict(int)
    for r in records:
        if r["kept"]:
            per_src[r["dataset"]] += 1
    print(f"\nScanned {len(records)} images, removed {removed} near-duplicates, kept {kept}")
    for k, v in sorted(per_src.items()):
        print(f"  {k:10s} {v}")
    print(f"Backgrounds : {C.BG_IMAGES}")
    print(f"Manifest    : {C.BG_MANIFEST}")


if __name__ == "__main__":
    main()

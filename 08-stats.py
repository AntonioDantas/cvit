#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 08 - Dataset statistics (compare with Table 3 of the paper:
OURs = 1 class, 226,638 instances, 6,162 images).

Reads out/manifest.csv, out/labels and the split lists; prints a markdown
table with images / instances per subset and the share of real vs. pasted
people.
"""

from collections import Counter

import config as C
from pipeline_utils import read_csv


def main():
    rows = read_csv(C.OUT_MANIFEST)
    if not rows:
        raise SystemExit("out/manifest.csv not found - run 05-combine.py first")

    split_of = {}
    for name in ("train", "val", "test"):
        f = C.OUT_DIR / f"{name}.txt"
        if f.exists():
            for line in f.read_text().splitlines():
                split_of[line.strip().split("/")[-1]] = name

    imgs, inst, real, pasted = Counter(), Counter(), Counter(), Counter()
    box_w, box_h, n_boxes = 0.0, 0.0, 0
    for r in rows:
        s = split_of.get(r["image"], "all")
        n_lab = sum(1 for l in (C.OUT_LABELS / r["image"].replace(".jpg", ".txt")).read_text().splitlines() if l.strip())
        for key in {s, "total"}:
            imgs[key] += 1
            inst[key] += n_lab
            real[key] += int(r["n_original"])
            pasted[key] += int(r["n_pasted"])
        for l in (C.OUT_LABELS / r["image"].replace(".jpg", ".txt")).read_text().splitlines():
            p = l.split()
            if len(p) == 5:
                box_w += float(p[3]); box_h += float(p[4]); n_boxes += 1

    versions = Counter(r["version"] for r in rows)
    print(f"versions: {dict(versions)}   backgrounds: {len({r['background'] for r in rows})}\n")
    print("| subset | images | instances | real | pasted | inst/img |")
    print("|---|---:|---:|---:|---:|---:|")
    for k in [k for k in ("train", "val", "test", "all", "total") if imgs[k]]:
        print(f"| {k} | {imgs[k]} | {inst[k]} | {real[k]} | {pasted[k]} | {inst[k] / imgs[k]:.1f} |")
    if n_boxes:
        print(f"\nmean box size (normalised): {box_w / n_boxes:.4f} x {box_h / n_boxes:.4f}")


if __name__ == "__main__":
    main()

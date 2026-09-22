#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for the C2A cluster-based synthetic pipeline.

Every step (01-...07-) imports its paths, model names and generation presets
from here, so the whole pipeline can be re-targeted by editing one file.

Reference:
  Fh, A. J. D. et al. "Long-range human detection in SAR: The C2A
  cluster-based synthetic approach", CVIT 2026 (SPIE vol. 14321, pp. 21-28).
"""

from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent

# INPUT (provided): raw background images, one sub-folder per public source
DATASETS_DIR = ROOT / "datasets"
# INPUT (provided): MPHB raw images (26,675 jpg)
PERSONS_RAW = ROOT / "persons" / "raw"

# Pre-trained weights / model caches downloaded by the scripts
MODELS_DIR = ROOT / "models"

# Step 01-03 outputs: curated backgrounds + sky mask + original person labels
BG_DIR = ROOT / "backgrounds"
BG_IMAGES = BG_DIR / "images"
BG_LABELS = BG_DIR / "labels"          # YOLO txt of people ALREADY in the background
BG_MANIFEST = BG_DIR / "manifest.csv"  # new_name -> source path, size, phash
CLOUD_FILE = BG_DIR / "cloud.txt"      # "<file> <px>" forbidden top band per image
BG_PREVIEW = BG_DIR / "preview"        # optional visual checks

# Step 04 outputs: validated transparent person cut-outs
PERSONS_VALID = PERSONS_RAW.parent / "valid"
PERSONS_REPORT = PERSONS_RAW.parent / "report.csv"

# Step 05-07 outputs: YOLO dataset, split and training runs
OUT_DIR = ROOT / "out"
OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels"
OUT_DEBUG = OUT_DIR / "debug"
OUT_MANIFEST = OUT_DIR / "manifest.csv"
OUT_YAML = OUT_DIR / "dataset.yaml"
OUT_RUNS = OUT_DIR / "runs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42
CLASS_NAMES = ["person"]

# ----------------------------------------------------------------- models
# Sky segmentation (Sec. 3.1.1): SegFormer-B0 fine-tuned on ADE20K
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SKY_MAX_SIDE = 1024          # segmentation resolution cap (memory)
SKY_TOP_BAND_FRAC = 0.20       # sky must touch the top 20 % to count as sky
DEFAULT_TOP_EXCLUSION = 0.30   # fallback when an image is missing in cloud.txt

# Background removal (Sec. 3.1.2): rembg U2-Net human segmentation
REMBG_MODEL = "u2net_human_seg"
REMBG_PROVIDERS = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
PERSON_MAX_SIDE_IN = 800       # MPHB frames are shrunk to this before rembg (U2-Net
                               # works at 320 px; cut-outs end up <= 81 px anyway)
PERSON_BATCH = 16              # images per detector call in step 04

# Person detector (COCO, class 0 = person) used for
#   - validating that a cut-out still looks like a person (Sec. 3.1.2)
#   - pseudo-labelling people already present in the backgrounds (Sec. 3.1.3)
YOLO_COCO_WEIGHTS = "yolov8x.pt"
PERSON_MIN_CONF = 0.50         # "minimum confidence of 50 %" (paper)
PERSON_MIN_VISIBLE_PIXELS = 100
PERSON_MIN_TRANSPARENT_FRAC = 0.02
BG_PERSON_MIN_CONF = 0.30
BG_PERSON_IMGSZ = 1280         # larger input helps the tiny people of aerial views

# Detector trained in the paper (Sec. 4.1)
TRAIN_WEIGHTS = "yolo11n.pt"
TRAIN = dict(epochs=100, imgsz=640, optimizer="Adam", batch=16)

# ----------------------------------------------------- dataset de-dup
DEDUP_PHASH_THRESHOLD = 7      # same value used in the original duplicate.py

# ------------------------------------------------ combination (Sec. 3.1.3)
# Instance augmentation ranges (Sec. 4.1, after Fang et al. InstaBoost)
AUGMENT = {
    "min_scale": 0.01,            # fraction of the shorter background side
    "max_scale": 0.04,
    "max_rotation": 90.0,         # degrees, uniform in [-90, +90]
    "person_max_side": 81,        # cut-outs are first reduced to <=81 px

    "random_crop_prob": 0.30,     # occlusion: cut one side of the patch
    "random_crop_max_fraction": 0.10,

    "color_match_prob": 0.30,     # brightness adaptation to local background
    "color_match_strength": 0.8,
    "color_match_min_factor": 0.5,
    "color_match_max_factor": 1.5,

    "blur_prob": 0.30,
    "blur_radius_min": 0.3,
    "blur_radius_max": 1.2,

    "noise_prob": 0.30,
    "noise_variance_min": 5.0,
    "noise_variance_max": 30.0,
    "noise_alpha_min": 0.10,
    "noise_alpha_max": 0.30,

    "compression_prob": 0.30,     # JPEG quality 10-80 on the patch only
    "compression_quality_min": 10,
    "compression_quality_max": 80,
}

# Geometry shared by every version
BG_MAX_SIDE = 1280             # synthetic images are capped at this size (training is 640)

COMBINE_BASE = {
    "variants_per_bg": 3,         # ~2,000 backgrounds x 3 = 6,162 images (Tab. 3)
    "margin": 2,                  # px safety margin between boxes
    "cluster_min_width_frac": 0.10,
    "cluster_max_width_frac": 0.50,
    "cluster_min_height_frac": 0.10,
    "cluster_max_height_frac": 0.50,
    "cluster_max_iou": 0.30,      # clusters may overlap at most this much
    "empty_variants_per_bg": 0,   # extra negatives (background only)
}

# Dataset versions evaluated in Sec. 4.1 (Table 1). v4 is the published model.
#   mode "fill"     : grid over the whole allowed area + random fill to max_per_bg
#   mode "clusters" : n clusters, each with an internal rows x cols grid and
#                     1..max_per_cell instances per cell, capped at max_per_bg
VERSIONS = {
    "v1": dict(mode="fill", clusters=(1, 1), grid=(3, 4), max_per_cell=2,
               max_per_bg=100, augment=False,
               note="baseline after Nihal et al.: full fill, patches not altered"),
    "v2": dict(mode="fill", clusters=(1, 1), grid=(3, 4), max_per_cell=2,
               max_per_bg=100, augment=True,
               note="full fill with instance augmentation"),
    "v3": dict(mode="clusters", clusters=(1, 3), grid=(5, 5), max_per_cell=3,
               max_per_bg=30, augment=True,
               note="1-3 clusters, 5x5 grid, 3 per cell"),
    "v4": dict(mode="clusters", clusters=(1, 5), grid=(4, 4), max_per_cell=2,
               max_per_bg=30, augment=True,
               note="1-5 clusters, 4x4 grid, 2 per cell (best, published)"),
    "v5": dict(mode="clusters", clusters=(1, 7), grid=(3, 3), max_per_cell=2,
               max_per_bg=30, augment=True,
               note="1-7 clusters, 3x3 grid, 2 per cell"),
}
DEFAULT_VERSION = "v4"


def combine_config(version: str = DEFAULT_VERSION) -> dict:
    """Merge base geometry + augmentation ranges + one version preset."""
    if version not in VERSIONS:
        raise KeyError(f"unknown version {version!r}; choose from {sorted(VERSIONS)}")
    cfg = dict(COMBINE_BASE)
    cfg.update(AUGMENT)
    cfg.update(VERSIONS[version])
    cfg["version"] = version
    if not cfg["augment"]:
        # v1 uses "unaltered markings": only the scale needed to fit the scene
        cfg["max_rotation"] = 0.0
    return cfg

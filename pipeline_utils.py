#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small helpers shared by the numbered pipeline steps."""

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw

import config as C


# ------------------------------------------------------------- files
def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS
             and not p.name.startswith(".")]
    return sorted(files)


def safe_name(stem: str) -> str:
    """Keep only [A-Za-z0-9._-] so every downstream tool can use the name."""
    stem = re.sub(r"\s+", "_", stem.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: Dict, fieldnames: Iterable[str]):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if new:
            w.writeheader()
        w.writerow(row)


def torch_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def yolo_weights(name: str = C.YOLO_COCO_WEIGHTS):
    """Load an Ultralytics model, downloading the .pt into models/ if needed."""
    from ultralytics import YOLO
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = C.MODELS_DIR / name
    if not dst.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(dst))
    return YOLO(str(dst))


# ------------------------------------------------------------ YOLO txt
def yolo_to_xyxy(line: str, w: int, h: int) -> Tuple[int, int, int, int] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    x1 = max(0, min(w - 1, int(round((cx - bw / 2) * w))))
    y1 = max(0, min(h - 1, int(round((cy - bh / 2) * h))))
    x2 = max(0, min(w, int(round((cx + bw / 2) * w))))
    y2 = max(0, min(h, int(round((cy + bh / 2) * h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(cls_id: int, box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return f"{cls_id} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def read_yolo_labels(path: Path, w: int, h: int) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """Return (valid lines, pixel boxes) of a YOLO label file, or ([], [])."""
    if not path.exists():
        return [], []
    lines, boxes = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        box = yolo_to_xyxy(line, w, h)
        if box is not None:
            lines.append(line.strip())
            boxes.append(box)
    return lines, boxes


def draw_boxes(img: Image.Image, boxes, color=(255, 0, 0), width: int = 2) -> Image.Image:
    canvas = img.convert("RGB")
    d = ImageDraw.Draw(canvas)
    for b in boxes:
        d.rectangle(list(b), outline=color, width=width)
    return canvas


# ------------------------------------------------------------ cloud.txt
def read_cloud_file(path: Path) -> Dict[str, int]:
    """Parse lines like 'file name.jpg 96px' into {name: 96}."""
    result: Dict[str, int] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        m = re.match(r"(\d+)", parts[-1])
        if m:
            result[" ".join(parts[:-1])] = int(m.group(1))
    return result

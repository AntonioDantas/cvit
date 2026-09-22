#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 07 - Train the detector (paper Sec. 4.1).

"All models were trained using the YOLO11n architecture (2.5M parameters,
 6.3 GFLOPs) under the same conditions: 100 epochs, resolution of 640x640, Adam optimizer, and batch size of 16" (Ultralytics 8.3.205, PyTorch 2.8, RTX 5090).

Input : out/dataset.yaml (step 06)
Output: out/runs/<name>/weights/best.pt, results.csv, curves ...

Usage
  python 07-train.py                    # full recipe (needs a GPU; hours)
  python 07-train.py --epochs 1 --fraction 0.02 --name smoke   # smoke test
  python 07-train.py --evaluate PATH    # only evaluate weights on the test split
  python 07-train.py --evaluate PATH --data /path/SARD/data.yaml --split val   # cross-dataset evaluation (paper Tables 4-5)
"""

import argparse

import config as C
from pipeline_utils import torch_device, yolo_weights


def print_confusion(metrics) -> None:
    """Markdown confusion matrix (Ultralytics counts at conf 0.25 / IoU 0.45).

    Columns are the ground truth, rows the prediction; the extra "background" row/column holds the false negatives (a labelled person with no matching box) and the false positives (a box on an empty patch of scene). The true-background / predicted-background cell is always 0: an image with no person and no detection produces no entry at all, so a background-only image shows up here only when it costs a false positive.
    """
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is None:
        return
    labels = list(C.CLASS_NAMES) + ["background"]
    mat = cm.matrix.astype(int)
    print("\n| pred \\ true | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for i, row in enumerate(mat):
        print(f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |")
    tp, fp, fn = mat[0][0], mat[0][-1], mat[-1][0]
    if tp + fp and tp + fn:
        print(f"\nTP={tp}  FP={fp}  FN={fn}  "
              f"precision={tp / (tp + fp):.4f}  recall={tp / (tp + fn):.4f}")


def image_level_confusion(weights: str, split: str, imgsz: int, device: str, conf: float = 0.25):
    """Per-IMAGE confusion matrix over a split of the generated dataset.

    The Ultralytics matrix counts boxes, so it can never fill the background/background cell: an image with no person and no detection produces no entry at all. This one asks a different, image-level question - "does the model see anybody in this frame?" - which is what tells us how many of the background-only images were actually READ as background:

        positive image = its label file has at least one box
        detected      = the model returns at least one box at `conf`

    so background/background here is a real true negative (a person-free scene left completely clean). It is not comparable with the box-level matrix nor with the tables of the paper; it only measures the negatives.
    """
    from ultralytics import YOLO

    listing = C.OUT_DIR / f"{split}.txt"
    if not listing.exists():
        print(f"[skip] {listing} not found - image-level matrix only works on the generated dataset")
        return
    names = [l.strip().split("/")[-1] for l in listing.read_text(encoding="utf-8").splitlines() if l.strip()]
    positive = {n: bool((C.OUT_LABELS / n.replace(".jpg", ".txt")).read_text(encoding="utf-8").strip())
                for n in names}

    model = YOLO(weights)
    detected, boxes_on_negatives = {}, 0
    for i in range(0, len(names), 16):
        chunk = names[i:i + 16]
        for name, r in zip(chunk, model.predict([str(C.OUT_IMAGES / n) for n in chunk],
                                                imgsz=imgsz, conf=conf, device=device, verbose=False)):
            detected[name] = len(r.boxes) > 0
            if not positive[name]:
                boxes_on_negatives += len(r.boxes)

    tp = sum(positive[n] and detected[n] for n in names)
    fn = sum(positive[n] and not detected[n] for n in names)
    fp = sum(not positive[n] and detected[n] for n in names)
    tn = sum(not positive[n] and not detected[n] for n in names)
    n_neg = fp + tn

    print(f"\nImage-level ({split}, conf {conf}): {len(names)} images, {n_neg} of them person-free")
    print("\n| pred \\ true | person | background |")
    print("|---|---:|---:|")
    print(f"| person | {tp} | {fp} |")
    print(f"| background | {fn} | {tn} |")
    if n_neg:
        print(f"\n{tn}/{n_neg} ({tn / n_neg:.1%}) of the background-only images were read as background "
              f"(no detection); the other {fp} produced {boxes_on_negatives} boxes in total "
              f"({boxes_on_negatives / n_neg:.2f} false positives per person-free image).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=C.TRAIN["epochs"])
    ap.add_argument("--imgsz", type=int, default=C.TRAIN["imgsz"])
    ap.add_argument("--batch", type=int, default=C.TRAIN["batch"])
    ap.add_argument("--optimizer", default=C.TRAIN["optimizer"])
    ap.add_argument("--weights", default=C.TRAIN_WEIGHTS)
    ap.add_argument("--name", default=None, help="run name (default: <weights>_<version>)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set (smoke tests)")
    ap.add_argument("--evaluate", default=None, help="skip training; evaluate these weights")
    ap.add_argument("--data", default=None, help="dataset yaml for --evaluate (default: out/dataset.yaml)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="subset for --evaluate")
    ap.add_argument("--image-level", action="store_true",
                    help="with --evaluate: also print the per-image confusion matrix "
                         "(the only one whose background/background cell is a real true negative)")
    args = ap.parse_args()

    device = args.device or torch_device()

    if args.evaluate:
        from ultralytics import YOLO
        data = args.data or str(C.OUT_YAML)
        m = YOLO(args.evaluate).val(data=data, split=args.split, imgsz=args.imgsz,
                                   device=device, project=str(C.OUT_RUNS), name="eval")
        print(f"{data} [{args.split}]  P={m.box.mp:.4f} R={m.box.mr:.4f} "
              f"mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")
        print_confusion(m)
        if args.image_level:
            image_level_confusion(args.evaluate, args.split, args.imgsz, device)
        return

    if not C.OUT_YAML.exists():
        raise SystemExit("out/dataset.yaml not found - run 06-split.py first")
    model = yolo_weights(args.weights)
    version = C.DEFAULT_VERSION
    if C.OUT_MANIFEST.exists():
        with C.OUT_MANIFEST.open(encoding="utf-8") as f:
            f.readline()
            version = (f.readline().split(",") + [version])[2].strip() or version
    name = args.name or f"{args.weights.replace('.pt', '')}_{version}"
    model.train(data=str(C.OUT_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                optimizer=args.optimizer, seed=C.SEED, device=device, fraction=args.fraction,
                project=str(C.OUT_RUNS), name=name, exist_ok=True)
    print(f"Weights: {C.OUT_RUNS / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

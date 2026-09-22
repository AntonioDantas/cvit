#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 07 - Train the detector (paper Sec. 4.1).

"All models were trained using the YOLO11n architecture (2.5M parameters,
 6.3 GFLOPs) under the same conditions: 100 epochs, resolution of 640x640,
 Adam optimizer, and batch size of 16" (Ultralytics 8.3.205, PyTorch 2.8,
 RTX 5090).

Input : out/dataset.yaml (step 06)
Output: out/runs/<name>/weights/best.pt, results.csv, curves ...

Usage
  python 07-train.py                    # full recipe (needs a GPU; hours)
  python 07-train.py --epochs 1 --fraction 0.02 --name smoke   # smoke test
  python 07-train.py --evaluate PATH    # only evaluate weights on the test split
  python 07-train.py --evaluate PATH --data /path/SARD/data.yaml --split val
                                        # cross-dataset evaluation (paper Tables 4-5)
"""

import argparse

import config as C
from pipeline_utils import torch_device, yolo_weights


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
    args = ap.parse_args()

    device = args.device or torch_device()

    if args.evaluate:
        from ultralytics import YOLO
        data = args.data or str(C.OUT_YAML)
        m = YOLO(args.evaluate).val(data=data, split=args.split, imgsz=args.imgsz,
                                   device=device, project=str(C.OUT_RUNS), name="eval")
        print(f"{data} [{args.split}]  P={m.box.mp:.4f} R={m.box.mr:.4f} "
              f"mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")
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

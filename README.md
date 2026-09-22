# Long-Range Human Detection in SAR: The C2A Cluster-Based Synthetic Approach

![Original / Combine / Annotated examples on ERA flood, landslide and mudslide backgrounds](docs/diagram.jpg)

*Synthetic image produced and the detections of the released weights*

Reproducible pipeline for the synthetic dataset and the detector presented in:

> A. J. Dantas Fh, E. H. Teixeira, L. D. Jesus, W. M. Martins, A. A. Chaves,
> E. H. Shiguemori and T. C. Pimenta, **"Long-range human detection in SAR:
> The C2A cluster-based synthetic approach"**, *Seventh International
> Conference on Computer Vision and Information Technology (CVIT 2026)*,
> SPIE vol. 14321, pp. 21-28, 2026.

**The synthetic dataset (v4) is included in this repository**: 6,135 images
and 174,295 *person* boxes in YOLO format under `out/`, ready for
`yolo train data=out/dataset.yaml`. Sections 3 and 4 explain how it was built
and how to regenerate it or the other versions.

Search-and-rescue (SAR) datasets rarely show people partially submerged in
muddy water or surrounded by debris. The paper builds such a dataset
synthetically with a **Combination-to-Application (C2A)** pipeline: validated
human cut-outs (MPHB) are pasted onto real aerial shots of hydrological
disasters, **in clusters rather than uniformly**, below an automatically
detected sky line and without covering the people that already appear in the
shot. A YOLO11n trained on the best version (v4) reached P = 0.907 and
mAP50 = 0.844 on its own test split and generalised competitively to SARD,
HERIDAL, LADD, AFO, SeaDronesSee and C2A.

```
datasets/ (aerial flood / landslide shots)        persons/raw (MPHB, 26,675 images)
     |                                                   |
  01 curate + pHash de-dup                        04 rembg (U2-Net human)
     |                                               + YOLOv8x person check >= 0.50
  03 YOLOv8x: people already in the shot --+         + crop to box
  02 SegFormer: sky line (cloud.txt) ------+              |
                                           |              |
                                           v              v
                     05 cluster-based C2A composition + instance augmentation
                                           |
                     06 split 80/10/10 (by background) -> out/dataset.yaml
                                           |
                     07 YOLO11n, 100 epochs, 640 px, Adam, batch 16
```

## 1. Layout

```
public/
|-- config.py                  paths, model names, augmentation ranges, presets v1-v5
|-- pipeline_utils.py          helpers shared by the steps
|-- 01-prepare-backgrounds.py  datasets/** -> backgrounds/images + manifest (pHash de-dup)
|-- 02-sky-mask.py             SegFormer-B0 (ADE20K "sky") -> backgrounds/cloud.txt
|-- 03-annotate-backgrounds.py YOLOv8x person boxes already in the shots -> backgrounds/labels
|-- 04-prepare-persons.py      MPHB -> rembg -> YOLOv8x validation (>= 0.50) -> persons/valid
|-- 05-combine.py              cluster-based C2A synthesis -> out/images, out/labels
|-- 06-split.py                80/10/10 split -> out/train.txt, val.txt, test.txt, dataset.yaml
|-- 07-train.py                YOLO11n training / evaluation -> out/runs
|-- 08-stats.py                images / instances per subset (compare with Table 3)
|-- 09-example-figure.py       Fig. 1-style grid (Original / Combine / Annotated) -> out/diagram.png
|-- run_all.sh                 the whole thing, in order
|-- requirements.txt
|-- docs/                      figures used in this README
|-- datasets/                  INPUT  - one folder per public source (see section 2)
|-- persons/raw/               INPUT  - MPHB images
|-- models/                    downloaded weights (auto)
|-- backgrounds/               step 01-03 outputs
|-- persons/valid/             step 04 output (transparent PNG cut-outs)
|-- out/                       step 05-07 outputs (YOLO dataset + runs)
`-- out/weight/best.pt         released detector weights (download, see section 5)
```

The repository tracks the code, this README, `docs/` and the **produced
dataset** (`out/images`, `out/labels`, split lists, `dataset.yaml`,
`manifest.csv`). Inputs (`datasets/`, `persons/`), downloaded weights
(`models/`), intermediates (`backgrounds/`, `persons/valid/`) and the
released detector (`out/weight/`) are git-ignored, see `.gitignore`.

## 2. Inputs

**Backgrounds** (`datasets/<SOURCE>/`): aerial views (about 30 to 150 m) of
floods, landslides and mudslides, manually filtered from:

| folder | source |
|---|---|
| `AIDER/` | Aerial Image Database for Emergency Response (Kyrkou and Theocharides, 2019), *flood* class |
| `ERA/` | Event Recognition in Aerial videos (Mou et al., 2020), *Flood*, *Landslide*, *Mudslide* |
| `ALLEY/` | Alley Flood Net (Electronics 14(10):2082, 2025) |
| `KAGGLE/` | Kaggle flood-image collections (Karanjit, Pally and Samadi 2022; Karim, Sharma and Barman 2022) |
| `ROBOFLOW/` | Roboflow Universe projects *landslide*, *flood-area-segmentation*, *flood-detection* |

Place the selected images in these folders (they are not redistributed here
because of the sources' licences). The run documented below started from
2,349 images; step 01 removed 304 near-duplicates and kept **2,045** (the
paper reports about 2,000).

**Human figures** (`persons/raw/`): the *Multiple Poses Human Body* dataset
(MPHB, Cai et al., 2016), 26,675 images of people in bent, kneeling,
sitting, standing and lying poses. Download it from the authors and copy the
JPGs into `persons/raw/`.

## 3. Setup

```bash
cd public
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # rembg[gpu] on CUDA machines
```

Weights (`yolov8x.pt`, `yolo11n.pt`, `u2net_human_seg.onnx`, SegFormer-B0)
are downloaded automatically into `models/` on first use.

## 4. Step by step

Every step is **resumable** (already-processed files are skipped) and has
`--help`, `--limit N` for a quick check and a fixed seed (`config.SEED = 42`).

### 01 - Background curation (paper Sec. 3.1.1)
```bash
python 01-prepare-backgrounds.py
```
Walks `datasets/**`, renames files to a flat, shell-safe scheme
(`<SOURCE>_<name>`), computes a perceptual hash and drops near-duplicates
(Hamming distance <= 7, the value used by the original `duplicate.py`).
Writes `backgrounds/images/` and `backgrounds/manifest.csv`.

### 02 - Sky exclusion mask (paper Sec. 3.1.1)
```bash
python 02-sky-mask.py --preview 20
```
Semantic segmentation with **SegFormer-B0 (ADE20K)**. For every image the
lowest pixel row classified as *sky* (when sky touches the top 20 % of the
frame) is written to `backgrounds/cloud.txt` as `"<file> <px>"`. Step 05
never pastes a figure above that line, so nobody "floats in the sky".
Images without an entry fall back to a 30 % top band. Segmentation runs on
a copy of at most 1024 px (`SKY_MAX_SIDE`) and the row is scaled back. In
this run 923 of the 2,045 backgrounds received a band (mean 53 px).

### 03 - People already present in the shots (paper Sec. 3.1.3)
```bash
python 03-annotate-backgrounds.py --preview 30
```
Some disaster shots already contain rescuers or victims. **YOLOv8x (COCO,
class *person*, conf >= 0.30, 1280 px)** annotates them into
`backgrounds/labels/<stem>.txt`. These real boxes are (i) copied into the
label of every synthetic image generated from that background and (ii)
treated as occupied space so pasted figures never overlap them. The boxes
can be checked in `backgrounds/preview/labels/`. In this run 180 backgrounds
contain people, giving 452 real boxes (1,356 after the three variants).

### 04 - Human figures (paper Sec. 3.1.2)
```bash
python 04-prepare-persons.py
```
For every MPHB image:
1. **Background removal**: `rembg` with the U2-Net `u2net_human_seg` model;
   the alpha channel is trimmed to its bounding box.
2. **Identification and clipping**: the cut-out is placed on white and
   checked with YOLOv8x; it is kept only if a *person* is found with
   **confidence >= 0.50**, no other COCO class is detected, at least 100 px
   are opaque and at least 2 % of the patch is transparent. The patch is
   re-cropped to the detected box and saved to `persons/valid/<id>.png`.

`persons/report.csv` records the decision for every image. The paper ends
this stage with about 18,000 valid figures; this run keeps **17,111**
(5,876 *no person*, 3,577 *other objects*, 111 empty or too small).

### 05 - Cluster-based C2A composition (paper Sec. 3.1.3 and 4.1)
```bash
python 05-combine.py                 # v4, the published configuration
python 05-combine.py --version v2    # any preset from config.VERSIONS
```
For each background three images are generated (backgrounds larger than
1280 px are shrunk first, `BG_MAX_SIDE`; training uses 640 px). Each starts
from the real person boxes of step 03 and then inserts new instances:

| version | mode | clusters | grid / cluster | per cell | cap / image | augment | note |
|---|---|---|---|---|---|---|---|
| v1 | full fill | - | 3x4 + random fill | 2 | 100 | no | baseline after Nihal et al. |
| v2 | full fill | - | 3x4 + random fill | 2 | 100 | yes | |
| v3 | clusters | 1-3 | 5x5 | 3 | 30 | yes | |
| **v4** | clusters | 1-5 | 4x4 | 2 | 30 | yes | **best, published model** |
| v5 | clusters | 1-7 | 3x3 | 2 | 30 | yes | |

Clusters are rectangles covering 10 to 50 % of the frame, drawn under the
sky line (IoU between clusters <= 0.3). Each cluster has an internal grid
and every cell receives 1 to *per cell* figures until the per-image cap is
reached, which produces the groupings of survivors the paper argues for.

Every figure is reduced to at most 81 px and scaled to **1 to 4 %** of the
shorter side of the background. With `augment` on (v2 to v5) it is also
rotated in **[-90, +90] degrees** and may get a one-side crop (occlusion,
up to 10 %), local brightness matching (factor 0.5 to 1.5), Gaussian blur
(radius 0.3 to 1.2), noise (blend 0.1 to 0.3) and JPEG re-compression
(quality 10 to 80), each with probability 0.3 (Sec. 4.1, after InstaBoost).
v1 pastes the figures as they are. Boxes never overlap (2 px margin).

Outputs: `out/images/img_XXXXXX.jpg`, `out/labels/img_XXXXXX.txt`
(class 0 = person, real + pasted), `out/manifest.csv` (background, variant,
number of real and pasted boxes, sky line) and `out/debug/` with triples
*background / composite / boxes* for the first 30 images (blue = real
people, red = pasted).

### 06 - Split (paper Sec. 4.1)
```bash
python 06-split.py
```
80 % / 10 % / 10 %. The split is made **per background**, so the three
variants of one shot always fall in the same subset. Produces
`out/train.txt`, `out/val.txt`, `out/test.txt` and `out/dataset.yaml`
(paths relative to `out/`, so the folder can be moved).

### 07 - Training and evaluation (paper Sec. 4.1 and 4.2)
```bash
python 07-train.py                       # YOLO11n, 100 epochs, 640 px, Adam, batch 16
python 07-train.py --evaluate out/runs/yolo11n_v4/weights/best.pt          # test split
python 07-train.py --evaluate best.pt --data /path/SARD/data.yaml --split val
                                         # cross-dataset evaluation, as in Tables 4-5
```
Same recipe as the paper (Ultralytics 8.3.205 / PyTorch 2.8 / RTX 5090 in
the original run). Use `--epochs 1 --fraction 0.02` for a smoke test; a full
run on CPU or Apple MPS takes many hours.

### 08 - Statistics
```bash
python 08-stats.py
```
Prints images / instances per subset to compare with Table 3 of the paper
(OURs: 6,162 images, 226,638 instances). Result of this run (v4):

| subset | images | instances | real | pasted | inst/img |
|---|---:|---:|---:|---:|---:|
| train | 4,908 | 139,551 | 1,224 | 138,327 | 28.4 |
| val | 612 | 17,235 | 36 | 17,199 | 28.2 |
| test | 615 | 17,509 | 96 | 17,413 | 28.5 |
| **total** | **6,135** | **174,295** | 1,356 | 172,939 | 28.4 |

Mean normalised box size 0.021 x 0.023 (about 13 x 15 px at 640 px), the
long-range, small-object regime targeted by the paper.

### 09 - Example figure (optional)
```bash
python 09-example-figure.py                                # label boxes
python 09-example-figure.py --weights out/weight/best.pt   # detector boxes + confidence
```
Builds `out/diagram.png`: Flood / Landslide / Mudslide examples from the ERA
backgrounds, rows *Original*, *Combine* and *Annotated* (as Fig. 1).

## 5. Pre-trained weights

The detector trained on the v4 dataset is released as **`best.pt`**
(Ultralytics YOLO11 checkpoint, single class *person*, 100 epochs, 640 px,
batch 16, Ultralytics 8.3.205; YOLO11x backbone, 56.9 M parameters,
114 MB). It is larger than GitHub's 100 MB file limit, so it is distributed
as a release asset rather than inside the repository:

```bash
mkdir -p out/weight
curl -L -o out/weight/best.pt \
  https://github.com/<user>/<repo>/releases/download/v1.0/best.pt
```

Then:

```bash
python 07-train.py --evaluate out/weight/best.pt          # metrics on the test split
python 09-example-figure.py --weights out/weight/best.pt  # figure above
yolo predict model=out/weight/best.pt source=my_flight.mp4 imgsz=640 conf=0.25
```

## 6. Results reported in the paper

Dataset versions, YOLO11n on the respective test split (Table 1):

| version | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v1 | 0.8621 | 0.6758 | 0.7638 | 0.3977 |
| v2 | 0.8819 | 0.7437 | 0.8351 | 0.4897 |
| v3 | 0.8961 | 0.7488 | 0.8327 | 0.4855 |
| **v4** | **0.9074** | **0.7598** | **0.8441** | **0.4967** |
| v5 | 0.8839 | 0.7390 | 0.8122 | 0.4140 |

Cross-domain evaluation of the v4 model ("OURs") on the validation sets of
public SAR datasets (Tables 4 and 5), considering only models tested outside
their own domain: in precision OURs is the best on AFO (0.63) and C2A (0.51)
and within 0.05 of the best on SeaDronesSee; its lowest relative scores are
on the terrestrial sets SARD, HERIDAL and LADD. In recall it is the best on
SARD, AFO, SeaDronesSee and C2A and is surpassed only on HERIDAL and LADD.

## 7. Notes on this re-implementation

* The numbered scripts consolidate the original research scripts
  (`cloud.py`, `identify.py`, `remover_fundo.py`, `validacao_claude.py`,
  `corte.py`, `combine.py`, `duplicate.py`, ...) into one linear, resumable
  pipeline with a single `config.py`.
* Version presets follow the parameters printed in the paper. The per-image
  cap of 30 for the cluster versions and 100 for the fill versions reproduce
  the "about 30 / about 100 instances per image" figures.
* In the paper the 50 % confidence check of Sec. 3.1.2 is attributed to the
  U-Net stage. In the original validation script, and here, U2-Net only
  removes the background; the confidence threshold comes from a COCO person
  detector (YOLOv8x) run on the cut-out.
* The paper lists a crop factor of 0.01; the original code (and this one)
  uses 0.10 (up to 10 % of one side). With 0.01 the crop would remove less
  than one pixel from patches of 81 px or less.
* v1 ("unaltered markings") keeps only the scale needed to fit the figure
  in the scene; rotation and every other change are off.
* The per-image cap of 30 gives 28.4 instances per image here versus 36.8 in
  Table 3 of the paper (the original run also counted the real boxes and
  used a slightly different fill); adjust `max_per_bg` in `config.VERSIONS`
  to change the density.
* Real people in the backgrounds are pseudo-labelled by YOLOv8x at
  conf >= 0.30; small figures below that confidence remain unlabeled
  background.
* Steps 02, 03 and 04 each load a large model; on a 16 GB machine run them
  one at a time (as `run_all.sh` does). Timings on an Apple M4: 01 about
  1 min, 02 about 25 min, 03 about 20 min, 04 about 2 h, 05 about 5 min.

## 8. Citation

```bibtex
@inproceedings{fh2026long,
  title     = {Long-range human detection in SAR: The C2A cluster-based synthetic approach},
  author    = {Fh, Antonio J Dantas and Teixeira, Eduardo H and Jesus, Leandro D and
               Martins, Wander M and Chaves, Antonio A and Shiguemori, Elcio H and
               Pimenta, Tales C},
  booktitle = {Seventh International Conference on Computer Vision and Information
               Technology (CVIT 2026)},
  volume    = {14321},
  pages     = {21--28},
  year      = {2026},
  organization = {SPIE}
}
```

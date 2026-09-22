#!/bin/zsh
# End-to-end reproduction of the C2A cluster-based synthetic dataset (v4) followed by YOLO11n training. Every step is resumable, so re-running the script continues where it stopped.
#
#   ./run_all.sh              # full pipeline
#   VERSION=v1 ./run_all.sh   # another preset from config.VERSIONS
#   EPOCHS=2 ./run_all.sh     # smoke-test the training step
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { python3.12 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
source .venv/bin/activate
VERSION=${VERSION:-v4}
EPOCHS=${EPOCHS:-100}

python 01-prepare-backgrounds.py
python 03-annotate-backgrounds.py    # memory-heavy steps run one at a time
python 02-sky-mask.py
python 04-prepare-persons.py
python 05-combine.py --version "$VERSION"
python 06-split.py
python 08-stats.py
python 07-train.py --epochs "$EPOCHS"

#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -U pip setuptools wheel

# Install everything
python3 -m pip install  -r requirements.txt
python3 -m pip install --no-build-isolation "git+https://github.com/bm424/scikit-cmeans.git@master"


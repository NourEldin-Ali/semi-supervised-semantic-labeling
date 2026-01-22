@echo off
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation "git+https://github.com/bm424/scikit-cmeans.git@master"

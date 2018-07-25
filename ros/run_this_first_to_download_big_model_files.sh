#!/bin/sh

cd src/tl_detector/light_classification/
mkdir -p runs/14_both_full_frames_model_saved
python setup_download_big_model_files.py
cd -

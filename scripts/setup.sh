#!/bin/bash
apt-get install ffmpeg

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U fvcore

poetry install
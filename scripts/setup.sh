#!/bin/bash
apt-get install ffmpeg

pip install poetry
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U fvcore

poetry install
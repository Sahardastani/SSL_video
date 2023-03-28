# SSL_video

## Instructions to setup the project

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

`pip install -e .`

## SimCLR feature extractor

The purpose here is to check if current image-based SSL methods have temporal consistency or not. In order to do that, we applied SimCLR resnet (pre-trained model) on each two consecutive frames of a video from `UCF 101` dataset.

### Instructions to setup the project

- First download the pretrained model (using this repository: `https://github.com/sthalles/SimCLR`) and change the directory of checkpoint in `predict.py` file.
- Then download the UCF101 dataset and unzip it (more details in `ucf101.py`).
- run `predict.py`.
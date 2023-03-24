# SSL_video

## Instructions to setup the project

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

`pip install -e .`

## SimCLR feature extractor

The purpose here is to check if current image-based SSL methods have temporal consistency or not. In order to do that, we applied SimCLR resnet (pre-trained model) on each two consecutive frames of a video from `UCF 101` dataset.

### Instructions to setup the project

- First download the pretrained model (using this repository: `https://github.com/sthalles/SimCLR`) and change the directory of checkpoint in `predict*.py` file.
- Then download the UCF101 dataset and extract each video frame using `feature_extractor.py`.
- Now you have two options: (1) Compute MSE loss between each two consecutive frame in one video --> `predict_two_frame_of_one_video.py`(2) Compute std of each video MSE losses --> `predict_all_frame_of_all_videos.py`
- You can find the results in `visualization` folder.
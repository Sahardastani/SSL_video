# SimCLR feature extractor

The purpose here is to check if current image-based SSL methods have temporal consistency or not. In order to do that, we applied SimCLR resnet (pre-trained model) on two frames of a video from `UCF 101` dataset.

## Instructions to setup the project

- First download the pretrained model (using this repository: `https://github.com/sthalles/SimCLR`) and change the directory of checkpoint in `fextractor.py` file.
- Then change two frame's direcotry in the code.
- Finally run the code using `python fextractor.py > out.txt` and analyse the result in `out.txt` file.
#!/bin/bash

# training
python src/pretrain.py > results/svt/train.txt
echo training finished =================

# testing
PROJECT_PATH="$HOME/projects/SSL_video"
CHECKPOINT="$PROJECT_PATH/checkpoints_pretraining/vicregl/video_vicregl.pth"
DATASET="ucf101"
DATA_PATH="${HOME}/projects/data/${DATASET}"

cd "$PROJECT_PATH/svt" || exit

export CUDA_VISIBLE_DEVICES=0
torchrun eval_knn.py \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --batch_size_per_gpu 128 \
  --nb_knn 5 \
  --temperature 0.07 \
  --num_workers 4 \
  --dataset "$DATASET" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/knn_splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" > ../results/svt/test.txt

echo testing finished =================
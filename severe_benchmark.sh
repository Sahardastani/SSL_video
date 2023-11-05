#!/bin/bash

# training
python src/pretrain.py > results/severe_benchmark/train.txt
echo training finished =================


cd SEVERE-BENCHMARK/action_recognition
file_path="configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml"

# finetuning
echo Sample_size , UCF1000 , finetune
sed -i 's/test_only: true/test_only: false/' "$file_path"
python finetune.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  vicregl  --pretext-model-path ../../checkpoints_pretraining/vicregl/video_vicregl.pth --finetune-ckpt-path ./checkpoints/vicregl/ --seed 100 > ../../results/severe_benchmark/finetune.txt
echo finetuning finished =================


# testing
echo Sample_size , UCF1000 , test
sed -i 's/test_only: false/test_only: true/' "$file_path"
python test.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  vicregl  --pretext-model-path ../../checkpoints_pretraining/vicregl/video_vicregl.pth --finetune-ckpt-path ./checkpoints/vicregl/ > ../../results/severe_benchmark/test.txt
echo testing finished =================
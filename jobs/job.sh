#!/bin/bash
#SBATCH --mail-user=sahar.dastani4776@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=brats
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00
#SBATCH --account=rrg-ebrahimi

source ~/env3.8/bin/activate

cd /home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/severe_benchmark/action_recognition

echo "-----<linear evaluation - ucf - tclr>-----"

python linear_eval.py configs/benchmark/ucf/112x112x32-fold1-linear_tclr.yaml   --pretext-model-name  tclr --pretext-model-path ../checkpoints_pretraining/tclr/rpd18kin400.pth --finetune-ckpt-path ./checkpoints/tclr/ 

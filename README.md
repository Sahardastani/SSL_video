# SSL_video

<!-- ## Instructions to setup the project

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

`pip install -e .` -->

# SEVERE Benchmark

Download Kinetics-400 pretrained R(2+1D)-18 weights for each method from [here](https://surfdrive.surf.nl/files/index.php/s/Zw9tbuOYAInzVQC). Unzip the downloaded file and it shall create a folder `checkpoints_pretraining/` with all the pretraining model weights.

## Dataset Preparation

The datasets can be downloaded from the following links (The annotations is provided for each dataset in the ./data/ directory):

* [UCF101 ](http://crcv.ucf.edu/data/UCF101.php)
Download the dataset and unrar it using `7z x UC101.zip`.
* [Something_something_v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
* [Fine-Gym v_1.0](https://sdolivia.github.io/FineGym/)

* [AVA preprocessing](https://research.google.com/ava/download.html): First, you need to create a symlink to the root dataset folder into the repo. For e.g., if you store all your datasets at `/path/to/datasets/`, then,
    ```sh
    # make sure you are inside the `SlowFast-ssl-vssl/` folder in the repo
    ln -s /path/to/datasets/ data
    ```
    These steps are based on the ones in original repo.

    1. Download: This step takes about 3.5 hours.
    ```sh
    cd scripts/prepare-ava/
    bash download_data.sh
    ```

    2. Cut each video from its 15th to 30th minute: This step takes about 14 hours.
    ```sh
    bash cut_videos.sh
    ```

    3. Extract frames: This step takes about 1 hour.
    ```sh
    bash extract_frames.sh
    ```

    4. Download annotations: This step takes about 30 minutes.
    ```sh
    bash download_annotations.sh
    ```

    5. Setup exception videos that may have failed the first time. For me, there was this video `I8j6Xq2B5ys.mp4` that failed the first time. See `scripts/prepare-ava/exception.sh` to re-run the steps for such videos.

* [Charades](https://prior.allenai.org/projects/charades) dataset.

    :hourglass: This, overall, takes about 2 hours.

    1. Download and unzip RGB frames
    ```sh
    cd scripts/prepare-charades/
    bash download_data.sh
    ```

    2. Download the split files
    ```sh
    bash download_annotations.sh
    ```

## Experiments

### Linear Evaluation 

* For evaluating pretrained models using linear evaluation on UCF-101 or Kinetics-400  use training scripts in  [./scripts_linear_evaluation](./scripts_linear_evaluation).

```bash
# Example linear evaluation on UCF-101

# Training
python linear_eval.py configs/benchmark/ucf/112x112x32-fold1-linear.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ 

# Testing
#set test_only flag to true in the  config file and run
python test.py configs/benchmark/ucf/112x112x32-fold1-linear.yaml   --pretext-model-name  rspnet --pretext-model-path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar --finetune-ckpt-path ./checkpoints/rspnet/ 
```

### I. Downstream domain-shift

* For finetuning pretrained models on domain shift datasets (e.g `something_something_v2`, `gym_99`, etc) use training scripts in  [./scripts_domain_shift/](./scripts_domain_shift/).
```bash
# Example finetuning pretrained  gdt model on something-something-v2 

## Training 
python finetune.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/ --seed 100
## Testing
# After finetuning, set test_only flag to true in the  config file (e.g configs/benchmark/something/112x112x32.yaml)  and run
python test.py  configs/benchmark/something/112x112x32.yaml  --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --finetune-ckpt-path ./checkpoints/gdt/
```

### II. Downstream sample-sizes

* For finetuning pretrained models with different sample sizes use training scripts in  [./scripts_sample_sizes](./scripts_sample_sizes).

```bash
# Example finetuning pretrained  video_moco model with 1000 ucf101 examples  

# Training
python finetune.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ --seed 100
# Note, set flag 'num_of_examples: to N'in the corresponding config file (e.g configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml) if you want to change the number of training samples to N.

# Testing
#set test_only flag to true in the  config file and run
python test.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  video_moco --pretext-model-path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar --finetune-ckpt-path ./checkpoints/video_moco/ 
```

### III. Downstream action-granularities

For this, we use the FineGym dataset that comes with a hierarchicy of actions.

* For finetuning pretrained models with different Fine-gym granularities (e.g `gym_event_vault`, `gym_set_FX_S1`, `gym288`, etc) use training scripts in  [./scripts_finegym_actions](./scripts_finegym_actions).

```bash
# Example finetuning pretrained  fully_supervised_kinetics model on set FX_S1  granularity

# Training
python finetune.py configs/benchmark/ucf/112x112x32-fold1_1000_examples.yaml   --pretext-model-name  gdt --pretext-model-path ../checkpoints_pretraining/gdt/gdt_K400.pth --seed 100

# Testing
#set test_only flag to true in the  config file and run
python test.py configs/benchmark/gym_set_FX_S1/112x112x32.yaml   --pretext-model-name  supervised --pretext-model-path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth 
```
### IV. Downstream Task-shift

#### In-domain
* **Training:**\
    For finetuning pretrained models for the task of Repetition-Counting run as following:

    ```bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_from_scratch  --pretext_model_name scratch

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_selavi  --pretext_model_name selavi  --pretext_model_path ../checkpoints_pretraining/selavi/selavi_kinetics.pth 

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_pretext_contrast  --pretext_model_name pretext_contrast --pretext_model_path ../checkpoints_pretraining/pretext_contrast/pcl_r2p1d_res_ssl.pt

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_avid_cma  --pretext_model_name avid_cma  --pretext_model_path ../checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_moco  --pretext_model_name moco --pretext_model_path ../checkpoints_pretraining/moco/checkpoint_0199.pth.tar

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_video_moco  --pretext_model_name video_moco --pretext_model_path ../checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_ctp  --pretext_model_name ctp --pretext_model_path ../checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_rspnet_snellius  --pretext_model_name rspnet --pretext_model_path ../checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_tclr  --pretext_model_name tclr --pretext_model_path ../checkpoints_pretraining/tclr/rpd18kin400.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_gdt  --pretext_model_name gdt --pretext_model_path ../checkpoints_pretraining/gdt/gdt_K400.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py   --batch_size 32  --result_path results/r2+1d_18_kinetics_pretrained_full_supervision  --pretext_model_name supervised --pretext_model_path ../checkpoints_pretraining/fully_supervised_kinetics/r2plus1d_18-91a641e6.pth 

    ```

* Testing:
    python main.py --no_train --resume_path = path to the finetuned_checkpoint with best validation accuracy (check validations logs)

#### Out-domain

* Install dependencies in [severe_benchmark/action_detection_multi_label_classification/setup/create_env.sh](severe_benchmark/action_detection_multi_label_classification/setup/create_env.sh)

* Task: Action Detection
    Symlink the pre-trained models for initialization. Suppose all your VSSL pre-trained checkpoints are at `../checkpoints_pretraining`
    ```sh
    ls -s ../checkpoints_pretraining/ checkpoints_pretraining
    ```
    Configure an output folder where all logs/checkpoints should be stored. For e.g., if you want to store all outputs at `/path/to/outputs/`, then symlink it:
    ```sh
    ln -s /path/to/outputs/ outputs
    ```

    We run all our experiments on AVA 2.2. To run fine-tuning on AVA, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
    ```sh
    cfg=configs/AVA/VSSL/32x2_112x112_R18_v2.2_supervised.yaml
    bash scripts/jobs/train_on_ava.sh -c $cfg
    ```

    (Optional) **W&B logging**: If you want to enable logging training curves on [Weights and Biases](wandb.ai), use the following command:
    ```sh
    bash scripts/jobs/train_on_ava.sh -c $cfg -w True -e <wandb_entity>
    ```
    where replace `<wandb_entity>` by your W&B username. Note that you first need to create an account on [Weights and Biases](wandb.ai) and then login in your terminal via `wandb login`.

    You can check out other configs for fine-tuning with other video self-supervised methods. The configs for all pre-training methods is provided below:

    | **Model**        | **Config**                                    |
    |------------------|-----------------------------------------------|
    | No pre-training  | `32x2_112x112_R18_v2.2_scratch.yaml`          |
    | SeLaVi           | `32x2_112x112_R18_v2.2_selavi.yaml`           |
    | MoCo             | `32x2_112x112_R18_v2.2_moco.yaml`             |
    | VideoMoCo        | `32x2_112x112_R18_v2.2_video_moco.yaml`       |
    | Pretext-Contrast | `32x2_112x112_R18_v2.2_pretext_contrast.yaml` |
    | RSPNet           | `32x2_112x112_R18_v2.2_rspnet.yaml`           |
    | AVID-CMA         | `32x2_112x112_R18_v2.2_avid_cma.yaml`         |
    | CtP              | `32x2_112x112_R18_v2.2_ctp.yaml`              |
    | TCLR             | `32x2_112x112_R18_v2.2_tclr.yaml`             |
    | GDT              | `32x2_112x112_R18_v2.2_gdt.yaml`              |
    | Supervised       | `32x2_112x112_R18_v2.2_supervised.yaml`       |

    The training is followed by an evaluation on the test set. Thus, the numbers will be displayed in logs at the end of the run.

* Task: Multi-Label Classification
    Configure an output folder where all logs/checkpoints should be stored. For e.g., if you want to store all outputs at `/path/to/outputs/`, then symlink it:
    ```sh
    ln -s /path/to/outputs/ outputs
    ```

    To run fine-tuning on Charades, using `r2plus1d_18` backbone initialized from Kinetics-400 supervised pretraining, we use the following command(s):
    ```sh
    # activate the environment
    conda activate slowfast

    # make sure you are inside the `SlowFast-ssl-vssl/` folder in the repo
    export PYTHONPATH=$PWD

    cfg=configs/Charades/VSSL/32x8_112x112_R18_supervised.yaml
    bash scripts/jobs/train_on_charades.sh -c $cfg -n 1 -b 16
    ```
    This assumes that you have setup data folders symlinked into the repo. This shall save outputs in `./outputs/` folder. You can check `./outputs/<expt-folder-name>/logs/train_logs.txt` to see the training progress.

    For other VSSL models, please check other configs in `configs/Charades/VSSL/`.

### Citation

If you use our work or code, kindly consider citing our paper:
```
@inproceedings{thoker2022severe,
  author    = {Thoker, Fida Mohammad and Doughty, Hazel and Bagad, Piyush and Snoek, Cees},
  title     = {How Severe is Benchmark-Sensitivity in Video Self-Supervised Learning?},
  journal   = {ECCV},
  year      = {2022},
}
```


### Acknowledgements

### Maintainers

* [Fida Thoker](https://fmthoker.github.io/)
* [Piyush Bagad](https://bpiyush.github.io/)

:bell: If you face an issue or have suggestions, please create a Github issue and we will try our best to address soon.

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

* Install dependencies in [severe_benchmark/action_detection_multi_label_classification/setup/create_env.sh](severe_benchmark/action_detection_multi_label_classification/setup/create_env.sh)

* Symlink the pre-trained models for initialization. Suppose all your VSSL pre-trained checkpoints are at `../checkpoints_pretraining`
    ```sh
    ls -s ../checkpoints_pretraining/ checkpoints_pretraining
    ```



#### Out-domain

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

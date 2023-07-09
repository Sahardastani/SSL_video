import os
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class DataConfig:
    PATH_PREFIX: str
    PATH_TO_DATA_DIR: str | Any = None
    NUM_FRAMES: int = 8
    SAMPLING_RATE: int = 32
    TRAIN_JITTER_SCALES: list = (256, 320)
    TRAIN_CROP_SIZE: int = 224
    TEST_CROP_SIZE: int = 224
    INPUT_CHANNEL_NUM: list = (3,)
    PIN_MEMORY: bool = True
    NUM_WORKERS: int = 12
    PATH_LABEL_SEPARATOR: str = " "
    DECODING_BACKEND: str = "pyav"
    MEAN: list = (0.45, 0.45, 0.45)
    STD: list = (0.225, 0.225, 0.225)
    CROP_SIZE: int = 224
    TARGET_FPS: int = 30
    INV_UNIFORM_SAMPLE: bool = False
    RANDOM_FLIP: bool = True
    MULTI_LABEL: bool = False
    USE_FLOW: bool = False
    NO_FLOW_AUG: bool = False
    NO_RGB_AUG: bool = False
    RAND_CONV: bool = False
    NO_SPATIAL: bool = False
    RAND_FR: bool = False
    TEMPORAL_EXTENT: int = 8
    DEIT_TRANSFORMS: bool = False
    COLOR_JITTER: float = 0.
    AUTO_AUGMENT: str = ''
    RE_PROB: float = 0.0


@dataclass
class DataLoader:
    PIN_MEMORY: bool = True
    NUM_WORKERS: int = 12
    ENABLE_MULTI_THREAD_DECODE: bool = True


@dataclass
class MultiGrid:
    # Multigrid training allows us to train for more epochs with fewer iterations.
    # This hyperparameter specifies how many times more epochs to train.
    # The default setting in paper trains for 1.5x more epochs than baseline.
    EPOCH_FACTOR: float = 1.5

    # Enable short cycles.
    SHORT_CYCLE: bool = False
    # Short cycle additional spatial dimensions relative to the default crop size.
    SHORT_CYCLE_FACTORS: list = (0.5, 0.5 ** 0.5)

    LONG_CYCLE: bool = False
    # (Temporal, Spatial) dimensions relative to the default shape.
    LONG_CYCLE_FACTORS: list = (
        (0.25, 0.5 ** 0.5),
        (0.5, 0.5 ** 0.5),
        (0.5, 1),
        (1, 1),)

    # While a standard BN computes stats across all examples in a GPU,
    # for multigrid training we fix the number of clips to compute BN stats on.
    # See https://arxiv.org/abs/1912.00998 for details.
    BN_BASE_SIZE: int = 8

    # Multigrid training epochs are not proportional to actual training time or
    # computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
    # evaluation. We use a multigrid-specific rule to determine when to evaluate:
    # This hyperparameter defines how many times to evaluate a model per long
    # cycle shape.
    EVAL_FREQ: int = 3

    # No need to specify; Set automatically and used as global variables.
    LONG_CYCLE_SAMPLING_RATE: int = 0
    DEFAULT_B: int = 0
    DEFAULT_T: int = 0
    DEFAULT_S: int = 0


@dataclass
class TestConfig:
    # If True test the model, else skip the testing.
    ENABLE: bool = True

    # Dataset for testing.
    DATASET: str = "kinetics"

    # Total mini-batch size
    BATCH_SIZE: int = 8

    # Path to the checkpoint to load the initial weight.
    CHECKPOINT_FILE_PATH: str = ""

    # Number of clips to sample from a video uniformly for aggregating the
    # prediction results.
    NUM_ENSEMBLE_VIEWS: int = 10

    # Number of crops to sample from a frame spatially for aggregating the
    # prediction results.
    NUM_SPATIAL_CROPS: int = 3

    # Checkpoint types include `caffe2` or `pytorch`.
    CHECKPOINT_TYPE: str = "pytorch"

    # Path to saving prediction results file.
    SAVE_RESULTS_PATH: str = ""


@dataclass
class ModelConfig:
    # Model architecture.
    ARCH: str = "slowfast"

    # Model name
    MODEL_NAME: str = "SlowFast"

    # The number of classes to predict for the model.
    NUM_CLASSES: int = 400

    # Loss function.

    LOSS_FUNC: str = "cross_entropy"

    # Model architectures that has one single pathway.
    SINGLE_PATHWAY_ARCH: list = ("c2d", "i3d", "slow", "x3d")

    # Model architectures that has multiple pathways.
    MULTI_PATHWAY_ARCH: list = ("slowfast")

    # Dropout rate before final projection in the backbone.
    DROPOUT_RATE: float = 0.5

    # Randomly drop rate for Res-blocks, linearly increase from res2 to res5
    DROPCONNECT_RATE: float = 0.0

    # The std to initialize the fc layer(s).
    FC_INIT_STD: float = 0.01

    # Activation layer for the output head.
    HEAD_ACT: str = "softmax"


@dataclass
class Config:
    DATA: DataConfig
    MULTIGRID: MultiGrid
    DATA_LOADER: DataLoader
    TEST: TestConfig
    MODEL: ModelConfig


def build_config(cfg: DictConfig) -> None:
    PATH_PREFIX = os.path.expanduser(cfg['dataset']['PATH_PREFIX'])
    # This is to account for the downsample_videos.py path
    if 'trainvaltest' in cfg.keys():
        PATH_TO_DATA_DIR = os.path.join(PATH_PREFIX, cfg['split'])
    else:
        PATH_TO_DATA_DIR = PATH_PREFIX

    DATA = DataConfig(PATH_PREFIX=PATH_PREFIX,
                      PATH_TO_DATA_DIR=PATH_TO_DATA_DIR,
                      NUM_FRAMES=cfg['dataset']['NUM_FRAMES'],
                      SAMPLING_RATE=cfg['dataset']['SAMPLING_RATE'],
                      TRAIN_JITTER_SCALES=cfg['dataset']['TRAIN_JITTER_SCALES'],
                      TRAIN_CROP_SIZE=cfg['dataset']['TRAIN_CROP_SIZE'],
                      TEST_CROP_SIZE=cfg['dataset']['TEST_CROP_SIZE'],
                      INPUT_CHANNEL_NUM=cfg['dataset']['INPUT_CHANNEL_NUM'],
                      PATH_LABEL_SEPARATOR=cfg['dataset']['PATH_LABEL_SEPARATOR'])
    DATA_LOADER = DataLoader(PIN_MEMORY=cfg['dataset']['PIN_MEMORY'],
                             NUM_WORKERS=cfg['dataset']['NUM_WORKERS'])
    config = Config(DATA=DATA,
                    MULTIGRID=MultiGrid(),
                    DATA_LOADER=DATA_LOADER,
                    TEST=TestConfig(),
                    MODEL=ModelConfig())
    return config

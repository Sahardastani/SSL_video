import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig


@dataclass
class DataConfig:
    PATH_PREFIX: str
    PATH_TO_DATA_DIR: str | Any = None
    NUM_CLASSES: int = 400
    # NUM_FRAMES: int = 8
    SAMPLING_RATE: int = 32
    TRAIN_JITTER_SCALES: list = (256, 320)
    TRAIN_CROP_SIZE: int = 224
    TEST_CROP_SIZE: int = 224
    INPUT_CHANNEL_NUM: list = (3,)
    PIN_MEMORY: bool = True
    NUM_WORKERS: int = 12
    PATH_LABEL_SEPARATOR: str = " "
    DECODING_BACKEND: str = "pyav"
    MEAN: list = field(default_factory=lambda: [0.45, 0.45, 0.45]) #[0.45, 0.45, 0.45]
    STD: list = field(default_factory=lambda: [0.225, 0.225, 0.225]) #(0.225, 0.225, 0.225)
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
    REVERSE_INPUT_CHANNEL: bool = False


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

    # Learning rate
    BASE_LR: float = 0.0005

    END_LR_RATIO: float = 0.001

@dataclass
class ModelConfig:

    # Model architecture that has one single pathway
    SINGLE_PATHWAY_ARCH: list = field(default_factory=lambda: ["c2d", "i3d", "slow", "x3d", "resnet2d1"])

    # Model architecture that has multiple pathways
    MULTI_PATHWAY_ARCH: list = field(default_factory=lambda: ["slowfast"])

    # Model architecture.
    ARCH: str = "resnet2d1"

    # Model name
    MODEL_NAME: str = "ResNetVicRegL"

    # Warmup epoch
    WARMUP_EPOCHS: int = 4

    # Optimizer
    OPTIMIZER: str = "lars"

    # Weight decay
    WEIGHT_DECAY: float = 0.05

    # The embedding dimension of projector
    MLP: str = "8192-8192-8192"

    # The embedding dimension of maps_projector
    MAPS_MLP: str = "512-512-512"

    # The size of each layer
    LAYER_SIZES: list = (1, 1, 1, 1)

    # 
    ALPHA: int = 0.75

    # The invariance coefficient
    INV_COEFF: float = 25.0

    # The variance coefficient
    VAR_COEFF: float = 25.0

    # The covariance coefficient
    COV_COEFF: float = 1.0

    # Control the use of L2 regularization
    L2_ALL_MATCHES: int = 1

    # Control the use of fast vector quantization (VC) regularization
    FAST_VC_REG: int = 0

    # The number of spatial matches in a feature map
    NUM_MATCHES: int = 20 #4

    # Checkpoint frequency
    CHECKPOINT_FREQ: int = 1

@dataclass
class DistributedConfig:
    
    # The number of distributed processes
    WORLD_SIZE: int = 1


    LOCAL_RANK: int = -1

    # url used to set up distributed training
    DIST_URL: str = "env://"

@dataclass
class Config:
    DATA: DataConfig
    MULTIGRID: MultiGrid
    DATA_LOADER: DataLoader
    TEST: TestConfig
    MODEL: ModelConfig
    DISTRIBUTE: DistributedConfig


def build_config(cfg: DictConfig) -> None:
    PATH_PREFIX = os.path.expanduser(cfg['dataset']['PATH_PREFIX'])
    # This is to account for the downsample_videos.py path
    if 'trainvaltest' in cfg.keys():
        PATH_TO_DATA_DIR = os.path.join(PATH_PREFIX, cfg['split'])
    else:
        PATH_TO_DATA_DIR = PATH_PREFIX

    DATA = DataConfig(PATH_PREFIX=PATH_PREFIX,
                      PATH_TO_DATA_DIR=PATH_TO_DATA_DIR,
                      NUM_CLASSES=cfg['dataset']['NUM_CLASSES'],
                      NUM_FRAMES=cfg['dataset']['NUM_FRAMES'],
                      SAMPLING_RATE=cfg['dataset']['SAMPLING_RATE'],
                      TRAIN_JITTER_SCALES=cfg['dataset']['TRAIN_JITTER_SCALES'],
                      TRAIN_CROP_SIZE=cfg['dataset']['TRAIN_CROP_SIZE'],
                      TEST_CROP_SIZE=cfg['dataset']['TEST_CROP_SIZE'],
                      INPUT_CHANNEL_NUM=cfg['dataset']['INPUT_CHANNEL_NUM'],
                      PATH_LABEL_SEPARATOR=cfg['dataset']['PATH_LABEL_SEPARATOR'])
    DATA_LOADER = DataLoader(PIN_MEMORY=cfg['dataset']['PIN_MEMORY'],
                             NUM_WORKERS=cfg['dataset']['NUM_WORKERS'])
    # TEST = TestConfig(BATCH_SIZE=cfg.common.batch_size)
    MODEL = ModelConfig(ARCH=cfg['feature_extractor']['ARCH'],
                        OPTIMIZER=cfg['feature_extractor']['OPTIMIZER'],
                        MLP=cfg['feature_extractor']['MLP'],
                        MAPS_MLP=cfg['feature_extractor']['MAPS_MLP'],
                        LAYER_SIZES=cfg['feature_extractor']['LAYER_SIZES'],
                        INV_COEFF=cfg['model']['INV_COEFF'],
                        VAR_COEFF=cfg['model']['VAR_COEFF'],
                        COV_COEFF=cfg['model']['COV_COEFF'])
    DISTRIBUTE = DistributedConfig(WORLD_SIZE=cfg['model']['WORLD_SIZE'],
                                   LOCAL_RANK=cfg['model']['LOCAL_RANK'],
                                   DIST_URL=cfg['model']['DIST_URL'])

    config = Config(DATA=DATA,
                    MULTIGRID=MultiGrid(),
                    DATA_LOADER=DATA_LOADER,
                    TEST=TestConfig(),
                    MODEL=MODEL,
                    DISTRIBUTE=DISTRIBUTE)
    return config
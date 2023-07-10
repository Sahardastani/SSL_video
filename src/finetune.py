from torchvision.datasets import Kinetics
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src import configs_dir
import os
@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def my_app(cfg : DictConfig) -> None:
    # Set the output directory to the project root
    a=cfg

if __name__ == "__main__":
    my_app()

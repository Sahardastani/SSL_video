import sys 
sys.path.append("/home/sdastani/scratch/SSL_video")




import wandb
import os

import hydra
import time 
import json 

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from omegaconf import DictConfig, OmegaConf
from src import configs_dir

from src.utils.defaults import build_config
from src.datasets.kinetics import Kinetics
from src.models.vicregl import VICRegL
from src.utils import utils 

@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_pretraining(cfg: DictConfig) -> None:

    config = build_config(cfg)

    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        resume=True,
        **cfg.wandb
    )

    utils.set_seed(cfg.common.seed)

    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=False)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.common.batch_size)
    # val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.common.batch_size)

    model = VICRegL(cfg=config)

    trainer = pl.Trainer(devices=torch.cuda.device_count(), 
                         strategy='ddp_find_unused_parameters_true',
                         max_epochs=cfg.common.epochs)

    trainer.fit(model, train_loader)

    wandb_logger = WandbLogger(project=cfg.wandb.name)
    wandb_logger.watch(model, log="all")
    wandb.finish()

if __name__ == "__main__":
    run_pretraining()

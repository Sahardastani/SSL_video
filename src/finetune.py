import sys
from typing import Any, Optional

from pytorch_lightning.utilities.types import STEP_OUTPUT
sys.path.append('/home/as89480@ens.ad.etsmtl.ca/projects/SSL_video')

import hydra
import os
import wandb
import numpy as np
from tqdm import tqdm
from src import configs_dir
from omegaconf import DictConfig, OmegaConf
from src.utils.defaults import build_config

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger

from src.models.feature_extractors.r2p1d import OurVideoResNet
from src.datasets.kinetics import Kinetics
from src.datasets.ucftransform import UCFtransform
from src.datasets.ucf101 import UCF101
from src.utils import model_utils

from src.models.new_vicregl import UCFReturnIndexDataset, Network

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_pretraining(cfg: DictConfig) -> None:

    config = build_config(cfg)
    # utils_main.set_seed(cfg.common.seed)

    # ucf = UCFtransform(cfg=config, mode="train", num_retries=10, get_flow=False)
    # ucfval = UCFtransform(cfg=config, mode="val", num_retries=10, get_flow=False)

    kin = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=False)
    kinval = Kinetics(cfg=config, mode="val", num_retries=10, get_flow=False)
    ucf_train = UCFReturnIndexDataset(cfg=config, mode="train", num_retries=10)
    ucf_val = UCFReturnIndexDataset(cfg=config, mode="val", num_retries=10)

    # ucf_loader = torch.utils.data.DataLoader(dataset=ucf, 
    #                                             batch_size=cfg.common.batch_size, 
    #                                             drop_last=True, 
    #                                             num_workers=cfg.common.num_workers,
    #                                             pin_memory=True)
    # ucfval_loader = torch.utils.data.DataLoader(dataset=ucfval, 
    #                                             batch_size=cfg.common.batch_size, 
    #                                             drop_last=True, 
    #                                             num_workers=cfg.common.num_workers,
    #                                             pin_memory=True)

    kin_loader = torch.utils.data.DataLoader(dataset=kin, 
                                                batch_size=cfg.common.batch_size, 
                                                drop_last=True, 
                                                num_workers=cfg.common.num_workers,
                                                pin_memory=True)
    kinval_loader = torch.utils.data.DataLoader(dataset=kinval, 
                                                batch_size=cfg.common.batch_size, 
                                                drop_last=True, 
                                                num_workers=cfg.common.num_workers,
                                                pin_memory=True)
    ucf_train_loader = torch.utils.data.DataLoader(ucf_train,
                                                batch_size=config.TESTsvt.batch_size_per_gpu,
                                                num_workers=config.TESTsvt.num_workers,
                                                pin_memory=True,
                                                drop_last=False,)
    ucf_val_loader = torch.utils.data.DataLoader(ucf_val,
                                                batch_size=config.TESTsvt.batch_size_per_gpu,
                                                num_workers=config.TESTsvt.num_workers,
                                                pin_memory=True,
                                                drop_last=False,)

    model = Network(config)
    model.apply(init_weights)

    wandb_logger = WandbLogger(name = 'knn', config=OmegaConf.to_container(cfg, resolve=True), project=cfg.wandb.project, log_model="all")
    trainer = pl.Trainer(devices= 1, strategy='ddp_find_unused_parameters_true', max_epochs=cfg.common.epochs, logger=wandb_logger, log_every_n_steps=1)
    wandb_logger.watch(model, log="all")
    trainer.fit(model, train_dataloaders = kin_loader, val_dataloaders = [kinval_loader, ucf_train_loader, ucf_val_loader])

    torch.save(model.backbone.state_dict(), os.path.join(cfg['dirs']['model_path'],'valid.pth'))

    wandb.finish()

if __name__ == "__main__":
    run_pretraining()
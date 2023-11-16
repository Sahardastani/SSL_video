import sys
sys.path.append('/home/as89480@ens.ad.etsmtl.ca/projects/SSL_video')

import wandb
import os

import hydra
import time 
import json 

import torch
from torch import nn
import torch.nn.init as init
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from pytorch_lightning.loggers import WandbLogger

from omegaconf import DictConfig, OmegaConf
from src import configs_dir

from src.utils.defaults import build_config
from src.datasets.kinetics import Kinetics
from src.models.vicregl import VICRegL, UCFReturnIndexDataset
from src.utils import utils_main
from src.utils.svt import utils 

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_pretraining(cfg: DictConfig) -> None:

    config = build_config(cfg)

    utils_main.set_seed(cfg.common.seed)
    
    kinetics_train = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=False)
    ucf_train = UCFReturnIndexDataset(cfg=config, mode="train", num_retries=10)
    ucf_val = UCFReturnIndexDataset(cfg=config, mode="val", num_retries=10)

    kinetics_train_loader = torch.utils.data.DataLoader(dataset=kinetics_train, 
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

    model = VICRegL(cfg=config)
    model.apply(initialize_weights)
    wandb_logger = WandbLogger(name = 'validation', config=OmegaConf.to_container(cfg, resolve=True), 
                                project=cfg.wandb.project, log_model="all")
    
    trainer = pl.Trainer(devices= 2, #torch.cuda.device_count(), 
                         strategy='ddp_find_unused_parameters_true',
                         max_epochs=cfg.common.epochs,
                         logger=wandb_logger,
                         log_every_n_steps=100)

    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_dataloaders = kinetics_train_loader, val_dataloaders = [ucf_train_loader, ucf_val_loader])
    
    if not os.path.exists(cfg['dirs']['model_path']):
        os.makedirs(cfg['dirs']['model_path'])
   
    torch.save(model.backbone.state_dict(), os.path.join(cfg['dirs']['model_path'],'valid.pth'))

    wandb.finish()


if __name__ == "__main__":
    run_pretraining()
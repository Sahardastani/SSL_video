import sys
sys.path.append('/home/as89480@ens.ad.etsmtl.ca/SSL_video')

import wandb
import os
os.environ["WANDB_MODE"]="offline"

import hydra
import time 
import json 

import torch
from torch import nn
import torch.nn.init as init
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from omegaconf import DictConfig, OmegaConf
from src import configs_dir

from src.utils.defaults import build_config
from src.datasets.kinetics import Kinetics
from src.models.vicregl import VICRegL
from src.utils import utils 

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_pretraining(cfg: DictConfig) -> None:

    config = build_config(cfg)

    # wandb.init(
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     reinit=True,
    #     resume=True,
    #     **cfg.wandb,
    # )

    utils.set_seed(cfg.common.seed)
    
    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=False)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=cfg.common.batch_size, 
                                                drop_last=True, 
                                                num_workers=cfg.common.num_workers,
                                                pin_memory=True)
    # import tqdm
    # for _ in tqdm.tqdm(train_loader):
    #     ...
    # return

    model = VICRegL(cfg=config)
    model.apply(initialize_weights)
    wandb_logger = WandbLogger(config=OmegaConf.to_container(cfg, resolve=True), 
                                project=cfg.wandb.name, 
                                offline = True)
    
    trainer = pl.Trainer(devices= 2, #torch.cuda.device_count(), 
                         strategy='ddp_find_unused_parameters_true',
                         max_epochs=cfg.common.epochs,
                         logger=wandb_logger,
                         log_every_n_steps=1)

    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_loader)
    print(model)
    torch.save(model.backbone.state_dict(), os.path.join(cfg['dirs']['model_path'],'final_global.pth'))

    wandb.finish()


if __name__ == "__main__":
    run_pretraining()
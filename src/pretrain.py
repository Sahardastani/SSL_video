import os
import hydra
import time 
import json 

import torch
from torch import nn

from omegaconf import DictConfig, OmegaConf
from src import configs_dir

from src.utils.defaults import build_config
from src.datasets.kinetics import Kinetics, make_inputs
from src.models.vicregl import VICRegL
from src.optimizers.optimizers import build_optimizer
from src.utils.distributed import init_distributed_mode
from src.utils import utils 

@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_pretraining(cfg: DictConfig) -> None:
    
    config = build_config(cfg)
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(config)
    gpu = torch.device(config.MODEL.DEVICE)

    # Downloads the kinetics dataset to the root folder specified in configs/datasets/kinetics.
    # Shows 10 clips from the same video.
    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=False)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.TEST.BATCH_SIZE)

    model = VICRegL(cfg=config).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    optimizer = build_optimizer(config, model)

    if os.path.isfile((os.path.join(config.MODEL.EXP_DIR, "model.pth"))):
        if config.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(os.path.join(config.MODEL.EXP_DIR, "model.pth"), map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    # start_epoch = 0

    for epoch in range(start_epoch, config.MODEL.EPOCHS):
        for step, inputs in enumerate(train_loader, start=epoch * len(train_loader)):
            lr = utils.learning_schedule(
                global_step=step,
                batch_size=config.TEST.BATCH_SIZE,
                base_lr=config.TEST.BASE_LR,
                end_lr_ratio=config.TEST.END_LR_RATIO,
                total_steps=config.MODEL.EPOCHS * len(train_loader.dataset) // config.TEST.BATCH_SIZE,
                warmup_steps=config.MODEL.WARMUP_EPOCHS
                * len(train_loader.dataset)
                // config.TEST.BATCH_SIZE,
            )
            for g in optimizer.param_groups:
                if "__MAPS_TOKEN__" in g.keys():
                    g["lr"] = lr * config.maps_lr_ratio
                else:
                    g["lr"] = lr

            optimizer.zero_grad()
            loss, logs = model.forward(make_inputs(inputs[0], gpu))
            loss.backward()
            optimizer.step()
        utils.checkpoint(config, epoch + 1, step, model, optimizer)

if __name__ == "__main__":
    run_pretraining()

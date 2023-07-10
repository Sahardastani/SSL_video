import os
import random

import wandb
import yaml

from src import configs_dir

name_of_our_run = "the_name_of_our_run"  # This should be unique. Can be from config file, can be the config file name, etc...

config_file = os.path.join(configs_dir(), "config.yaml")
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
config.update({"lr": 0.2, "batch_size":32})
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="training-runs",
    entity="video_ssl_project",
    name="the_name_of_our_run",
    # track hyperparameters and update meta-data. This is really important to get right.
    config=config
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

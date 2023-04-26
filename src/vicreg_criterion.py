# Source: https://docs.lightly.ai/self-supervised-learning/examples/vicreg.html
# Put your images or videos in the data folder:

import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate

## The projection head is the same as the Barlow Twins one
from lightly.loss import VICRegLoss

## The projection head is the same as the Barlow Twins one
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform

from models.vicreg import VICReg
from __init__ import top_dir, data_dir, configs_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = VICReg(backbone)
model.to(device)

transform = VICRegTransform(input_size=32)
# create a dataset from a folder containing images or videos:
dataset = LightlyDataset(data_dir(), transform=transform)

collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)

criterion = VICRegLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x0, x1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
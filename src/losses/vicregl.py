# Source: https://docs.lightly.ai/self-supervised-learning/examples/vicregl.html
# Put your images or videos in the data folder:

import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import VICRegLLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.transforms.vicregl_transform import VICRegLTransform

from models.vicreg import VICRegL
from __init__ import top_dir, data_dir, configs_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-2])
model = VICRegL(backbone)
model.to(device)

transform = VICRegLTransform(n_local_views=0)
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

criterion = VICRegLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for views_and_grids, _, _ in dataloader:
        views_and_grids = [x.to(device) for x in views_and_grids]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2 :]
        features = [model(view) for view in views]
        loss = criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
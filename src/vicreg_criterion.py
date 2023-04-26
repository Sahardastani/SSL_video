import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import VICRegLoss
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform

from datasets.ucf101 import DoubleTransformBatchDataset
from models.vicreg import VICReg

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = VICReg(backbone)
model.to(device)

criterion = VICRegLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

# Define the transformations you want to apply
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])

transform2 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(degrees=45),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create a list of image paths and a dataset that uses the two different transformations
img_paths = ["flow-frames/frame_0010.png", "flow-frames/frame_0011.png"]
dataset = DoubleTransformBatchDataset(img_paths, transform1=transform1, transform2=transform2)

# Create a dataloader that uses the dataset, with a batch size of 2
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for x0, x1 in dataloader:
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

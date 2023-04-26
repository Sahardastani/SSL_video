import os
import cv2
import yaml
import warnings
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models

import sys
sys.path.insert(0, 'src')
sys.path.insert(1, 'src/datasets')
sys.path.insert(2, 'src/models')

import ucf101
import resnet, feature_extractor
from __init__ import top_dir, data_dir, configs_dir

# Define config, device, and ignore warnings
config = yaml.load(open("./src/configs/config.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
warnings.filterwarnings('ignore')

# load the model from checkpoint
model = resnet.SimSiam(models.__dict__['resnet50'], 2048, 512)
torch.save({'model_state_dict': model.state_dict()}, config['checkpoint']['simsiam'])
checkpoint = torch.load(config['checkpoint']['simsiam'], map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
new_model = feature_extractor.FeatureExtractor_simsiam(model.to(device)).to(device)

# Define the transform(s) to be applied to the video tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the dataset
video_dataset = ucf101.VideoDataset(data_dir(), transform=transform)

# Create a dataloader for the dataset
video_dataloader = DataLoader(video_dataset, shuffle=False)

std_list = []
all_video = {}
loss = nn.MSELoss()
for i, batch in enumerate(video_dataloader):
    extracted_features = {}
    for idx, frame in enumerate(batch[0]):
        frame = frame.to(device).float()
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            feature = new_model(frame)
        extracted_features[idx] = feature.to("cpu")
        del feature
        torch.cuda.empty_cache()

    # Compute MSE losses between each two consecutive frame for a videos
    loss_list=[]
    for k in range(len(extracted_features)-1):
        output = loss(extracted_features[k], extracted_features[k+1]).item()
        loss_list.append(output)
    if len(loss_list) == 50:
        std_list.append(loss_list)

    all_video[video_dataset.video_files[i]] = extracted_features

    print(f'video {i}: len(loss_list) = {len(loss_list)}')

# plot the stds over all videos
final_x = [key for key in extracted_features]
final_x.pop()
final_y = np.std(std_list, axis=0)
print(f'len(std_list) = {len(std_list)}, len(x) = {len(final_x)}, len(y) = {len(final_y)}')
plt.plot(final_x,final_y)
plt.xlabel('Frames')
plt.ylabel('Losses')
plt.title('Std of MSE losses between each two consecutive frame for all videos')
plt.savefig('./src/visualization/std_simsiam.png')
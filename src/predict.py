import os
import cv2
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
from datasets.ucf101 import VideoDataset
from models.resnet_simclr import ResNetSimCLR
from models.feature_extractor import FeatureExtractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--pre_trained_model', 
                    type=str, 
                    default='/home/sdastani/scratch/resnet18/checkpoint_0100.pth.tar', 
                    help='The directory of pre-trained model.')

parser.add_argument('--videos', 
                    type=str, 
                    default='/home/sdastani/scratch/UCF101',
                    help='The direcotry of videos.')
args = parser.parse_args()


# load the model from checkpoint
model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, args.pre_trained_model)
checkpoint = torch.load(args.pre_trained_model, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
new_model = FeatureExtractor(model.to(device))


# Define the transform(s) to be applied to the video tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the dataset
video_dataset = VideoDataset(args.videos, transform=transform)

# Create a dataloader for the dataset
video_dataloader = DataLoader(video_dataset, shuffle=True)

# breakpoint()
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
    std_list.append(loss_list)

    all_video[video_dataset.video_files[i]] = extracted_features

    # Plot MSE losses between each two consecutive frame for a videos
    x = [key for key in extracted_features]
    x.pop()
    y = loss_list
    plt.plot(x,y)
    plt.xlabel('Frames')
    plt.ylabel('Losses')
    plt.title(f'Std of MSE losses between each two consecutive frame for {video_dataset.video_files[i]}')
    plt.savefig(f'./src/visualization/video_{i}.png')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    print(f'video {i} is finished.')

# plot the stds over all videos
final_x = [key for key in extracted_features]
final_x.pop()
final_y = np.std(std_list, axis=0)
plt.plot(final_x,final_y)
plt.xlabel('Frames')
plt.ylabel('Losses')
plt.title('Std of MSE losses between each two consecutive frame for all videos')
plt.savefig('./src/visualization/std.png')
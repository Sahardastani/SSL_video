import os
import cv2
import argparse
import warnings
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
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

parser.add_argument('--frames', 
                    type=str, 
                    default='./frames/v_IceDancing_g01_c01.avi', #'/home/sdastani/scratch/ucf101/subset/v_IceDancing_g01_c01.avi', 
                    help='The direcotry of folder containing frames of each video.')

args = parser.parse_args()


# load the model from checkpoint
model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, args.pre_trained_model)
checkpoint = torch.load(args.pre_trained_model, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
new_model = FeatureExtractor(model)
model.to(device)

# read and transform each frame
extracted_features = {}
for frame in os.listdir(args.frames):
    frame_path = os.path.join(args.frames, frame)
    img = cv2.imread(frame_path)
    # breakpoint()
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.to(device).float()
    feature = new_model(img)
    extracted_features[int(frame.split('.')[0][5:])] = feature

# sort the keys (which are frames names)
myKeys = list(extracted_features.keys())
myKeys.sort()
sorted_dict = {i: extracted_features[i] for i in myKeys}

# compute the MSE loss between consecutive frames in input video
loss_list=[]
loss = nn.MSELoss()
for i in range(len(sorted_dict)-1):
    output = loss(sorted_dict[i], sorted_dict[i+1]).item()
    loss_list.append(output)

# plot the MSEs
x = [key for key in sorted_dict]
x.pop()
y = [key for key in loss_list]
plt.plot(x,y)
plt.xlabel('Frames')
plt.ylabel('Losses')
plt.title('MSE loss between each two consecutive frame in one video (IceDancing_g01_c01)')
plt.savefig('./src/visualization/MSE.png')



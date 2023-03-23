import os
import cv2
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

config = {
    'pre_trained_model':{'dir': '/home/sdastani/scratch/resnet18/checkpoint_0100.pth.tar'},
    'frames': {'dir':'/home/sdastani/scratch/ucf101/subset'}
}

# load the model from checkpoint
model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, config['pre_trained_model']['dir'])
checkpoint = torch.load(config['pre_trained_model']['dir'], map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
new_model = FeatureExtractor(model)
model.to(device)

# compute the MSE loss between consecutive frames in each video and do std over all of them
all_video = {}
loss_dict = {}
std_list = []
loss = nn.MSELoss()
for video in os.listdir(config['frames']['dir']):
    extracted_features = {}
    for frame in os.listdir(os.path.join(config['frames']['dir'], video)):
        frame_path = os.path.join(config['frames']['dir'], video, frame)
        img = cv2.imread(frame_path)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img = img.unsqueeze(0)
        img = img.to(device).float()
        feature = new_model(img)
        if int(frame.split('.')[0][5:]) < 20:
            extracted_features[int(frame.split('.')[0][5:])] = feature

    myKeys = list(extracted_features.keys())
    myKeys.sort()
    sorted_dict = {i: extracted_features[i] for i in myKeys}

    all_video[video[2:-4]] = sorted_dict

    loss_list=[]
    for i in range(len(sorted_dict)-1):
        output = loss(sorted_dict[i], sorted_dict[i+1]).item()
        loss_list.append(output)
        loss_dict[video[2:-4]] = loss_list

    x = [key for key in sorted_dict]
    x.pop()
    y = [key for key in loss_list]
    std_list.append(y)
    print(f'video {video} is finished.')

# plot the stds
final_x = [key for key in sorted_dict]
final_y = np.std(std_list, axis=0)
plt.plot(x,y)
plt.xlabel('Frames')
plt.ylabel('Losses')
plt.title('Std of MSE losses between each two consecutive frame for 21 videos')
plt.savefig('/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/src/visualization/std.png')



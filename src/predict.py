import cv2
import torch 
import torch.nn as nn
import numpy as np
import torchvision.models as models
from models.resnet_simclr import ResNetSimCLR
from models.feature_extractor import FeatureExtractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

config = {
    'pre_trained_model':{'dir': '/home/sdastani/scratch/resnet18/checkpoint_0100.pth.tar'},
    'first_frame':{'dir': '/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/frames/frame0.jpg'},
    'second_frame':{'dir': '/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/frames/frame1.jpg'},
}
# breakpoint()
model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, config['pre_trained_model']['dir'])
checkpoint = torch.load(config['pre_trained_model']['dir'], map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
new_model = FeatureExtractor(model)

model.to(device)
img1 = cv2.imread(config['first_frame']['dir'])
img1 = torch.from_numpy(img1)
img1 = img1.permute(2, 0, 1)
img1 = img1.unsqueeze(0)
img1 = img1.to(device).float()
feature1 = new_model(img1)

img2 = cv2.imread(config['second_frame']['dir'])
img2 = torch.from_numpy(img2)
img2 = img2.permute(2, 0, 1)
img2 = img2.unsqueeze(0)
img2 = img2.to(device).float()
feature2 = new_model(img2)

loss = nn.MSELoss()
output = loss(feature1, feature2)

print('distance:', output)


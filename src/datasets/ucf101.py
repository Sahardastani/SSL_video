import os
import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Download 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip' and unzip it

# Define the dataset class
class ucf101(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get a list of all the video files in the directory
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.avi')]
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        # Open the video file
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        video = cv2.VideoCapture(video_path)
        
        # Read each frame of the video and store them in a list
        frames = []
        while True:
            ret, frame = video.read()
            if not ret or len(frames) >= 51:
                break
            frames.append(frame)
        
        # Convert the list of frames to a tensor
        video_tensor = torch.stack([transforms.ToTensor()(frame) for frame in frames])
        
        # Apply any specified transforms to the video tensor
        if self.transform:
            video_tensor = self.transform(video_tensor)
            
        # Close the video file and return the tensor
        video.release()
        return video_tensor
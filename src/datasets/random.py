import torch
from torch.utils.data import Dataset, DataLoader
import random

# Create a custom dataset
class RandomDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random data here (replace this with your actual data generation logic)
        random_data = torch.randn(10)  # Example: generating a random tensor of size 10
        return random_data
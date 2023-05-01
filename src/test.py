import os

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from src import configs_dir


# The flow is come from pwc NVIDIA model (https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/tmp)

def warp_image(image, flow):
    """Warp an image using optical flow.

    Args:
        image (torch.Tensor): A tensor of shape (batch_size, channels, height, width)
            containing the input image.
        flow (torch.Tensor): A tensor of shape (batch_size, 2, height, width) containing
            the optical flow vectors.

    Returns:
        warped_image (torch.Tensor): A tensor of the same shape as `image` containing the
            warped image.
    """
    # Create a grid of pixel coordinates
    batch_size, _, height, width = image.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Add optical flow to the grid
    grid = grid + flow

    # Warp the image using the grid
    warped_image = F.grid_sample(image, grid.permute(0, 2, 3, 1), align_corners=True)
    return warped_image


# Define config
config_file = os.path.join(configs_dir(), "config.yaml")
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

# Load input frames and optical flow
frame1 = TF.to_tensor(Image.open(config['warping']['frame1'])).unsqueeze(0)
frame2 = TF.to_tensor(Image.open(config['warping']['frame2'])).unsqueeze(0)
test = TF.to_tensor(Image.open('/home/sdastani/scratch/new/SSL_video/further/000000.flo.png')).unsqueeze(0)
flow = torch.tensor(cv2.readOpticalFlow(config['warping']['flow'])).permute(2, 0, 1).unsqueeze(0)

# Warp frame1 to get frame2
warped_frame1 = warp_image(frame1, flow)

# Compute the difference between the warped frame1 and frame2
diff = (warped_frame1 - frame2).abs().sum(dim=1, keepdim=True)

# Save the result
result = TF.to_pil_image(diff[0])
result.save('src/visualization/warped_image.png')

import av 
import torch
from torchvision import transforms
from torchvision.datasets import UCF101

config = {
    'data':{'dir':'/home/sdastani/scratch/ucf101'},
    'label':{'dir':'/home/sdastani/scratch/ucfTrainTestlist'},
    'frames_per_clip':{'value': 5},
    'step_between_clips':{'value': 1},
    'batch_size':{'value': 32}
}
# breakpoint()
tfs = transforms.Compose([
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

train_dataset = UCF101(config['data']['dir'], 
                        config['label']['dir'], 
                        frames_per_clip=config['frames_per_clip']['value'],
                       step_between_clips=config['step_between_clips']['value'], 
                       train=True, 
                       transform=tfs)

test_dataset = UCF101(config['data']['dir'], 
                        config['label']['dir'], 
                        frames_per_clip=config['frames_per_clip']['value'],
                      step_between_clips=config['step_between_clips']['value'], 
                      train=False, 
                      transform=tfs)


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=config['batch_size']['value'], 
                                            shuffle=True,
                                           collate_fn=custom_collate)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=config['batch_size']['value'], 
                                            shuffle=True,
                                            collate_fn=custom_collate)

print(f"Total number of train samples: {len(train_dataset)}")
print(f"Total number of test samples: {len(test_dataset)}")
print(f"Total number of (train) batches: {len(train_loader)}")
print(f"Total number of (test) batches: {len(test_loader)}")
print()
import cv2
import torch 
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
# from exceptions.exceptions import InvalidBackboneError
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)



class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		
        # Extract resnet18 backbone Layers
    self.backbone = nn.Sequential(
        model.backbone.conv1,
        model.backbone.bn1,
        model.backbone.relu,
        model.backbone.maxpool,
        model.backbone.layer1,
        model.backbone.layer2,
        model.backbone.layer3,
        model.backbone.layer4
    )

		# Extract resnet18 Average Pooling Layer
    self.pooling = model.backbone.avgpool

		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()

		# Extract the fully-connected layer from resnet18
    self.fc = model.backbone.fc
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.backbone(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 


model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, '/home/sdastani/scratch/resnet18/checkpoint_0100.pth.tar')
checkpoint = torch.load('/home/sdastani/scratch/resnet18/checkpoint_0100.pth.tar', map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
new_model = FeatureExtractor(model)

model.to(device)
img1 = cv2.imread('/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/SimCLR_feature_extractor/frame0.jpg')
img1 = torch.from_numpy(img1)
img1 = img1.permute(2, 0, 1)
img1 = img1.unsqueeze(0)
img1 = img1.to(device).float()
feature1 = new_model(img1)

img2 = cv2.imread('/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/SimCLR_feature_extractor/frame1.jpg')
img2 = torch.from_numpy(img2)
img2 = img2.permute(2, 0, 1)
img2 = img2.unsqueeze(0)
img2 = img2.to(device).float()
feature2 = new_model(img2)

print(feature1)
print(feature2)
# breakpoint()
print('model successfuly loaded.')
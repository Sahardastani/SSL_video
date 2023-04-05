import torch 
import torch.nn as nn

# Extract feature from an input image in resnet from simclr
class FeatureExtractor_simclr(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor_simclr, self).__init__()
		
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

# Extract feature from an input image in resnet from byol
class FeatureExtractor_byol(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor_byol, self).__init__()
		
        # Extract resnet18 backbone Layers
    self.backbone = nn.Sequential(
        model.encoder[0],
        model.encoder[1],
        model.encoder[2],
        model.encoder[3],
        model.encoder[4],
        model.encoder[5],
        model.encoder[6],
        model.encoder[7]
    )

		# Extract resnet18 Average Pooling Layer
    self.pooling = model.encoder[8]

		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()

		# Extract the fully-connected layer from resnet18
    self.fc = nn.Linear(in_features=512, out_features=128, bias=True)
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.backbone(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 

# Extract feature from an input image in resnet from simsiam
class FeatureExtractor_simsiam(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor_simsiam, self).__init__()
		
        # Extract resnet50 backbone Layers
    self.backbone = nn.Sequential(
        model.encoder.conv1,
        model.encoder.bn1,
        model.encoder.relu,
        model.encoder.maxpool,
        model.encoder.layer1,
        model.encoder.layer2,
        model.encoder.layer3,
        model.encoder.layer4
    )

		# Extract resnet50 Average Pooling Layer
    self.pooling = model.encoder.avgpool

		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()

		# Extract the fully-connected layer from resnet50
    self.fc = nn.Linear(in_features=2048, out_features=512, bias=True)
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.backbone(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 

# Extract feature from an input image in resnet from swav
class FeatureExtractor_swav(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor_swav, self).__init__()
		
        # Extract resnet50 backbone Layers
    self.backbone = nn.Sequential(
        model.padding,
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )

		# Extract resnet50 Average Pooling Layer
    self.pooling = model.avgpool

		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()

		# Extract the fully-connected layer from resnet50
    self.fc = nn.Linear(in_features=2048, out_features=512, bias=True)
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.backbone(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 
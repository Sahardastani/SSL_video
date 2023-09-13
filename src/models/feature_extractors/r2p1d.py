from dataclasses import dataclass

import torch.nn
from torchvision.models.video import r2plus1d_18


@dataclass
class MyOutLayers:
    layer0_out: torch.Tensor
    layer1_out: torch.Tensor
    layer2_out: torch.Tensor
    layer3_out: torch.Tensor
    layer4_out: torch.Tensor 
    layer_pool_out: torch.Tensor

class OurVideoResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = r2plus1d_18()

    def forward(self, x):
        layer0_out= self.backbone.stem(x)
        layer1_out = self.backbone.layer1(layer0_out)
        layer2_out = self.backbone.layer2(layer1_out)
        layer3_out = self.backbone.layer3(layer2_out)
        layer4_out = self.backbone.layer4(layer3_out)
        layer_pool_out= self.backbone.avgpool(layer4_out)

                             #input=input,          #torch.Size([6, 3, 8, 112, 112]) 4
        output = MyOutLayers(layer0_out=layer0_out, #torch.Size([6, 64, 8, 56, 56]) 4
                             layer1_out=layer1_out, #torch.Size([6, 64, 8, 56, 56]) 4
                             layer2_out=layer2_out, #torch.Size([6, 128, 4, 28, 28]) 2
                             layer3_out=layer3_out, #torch.Size([6, 256, 2, 14, 14]) 1
                             layer4_out=layer4_out, #torch.Size([6, 512, 1, 7, 7]) 1
                             layer_pool_out=layer_pool_out) #torch.Size([6, 512, 1, 1, 1])

        return output

# if __name__ == '__main__':

#     myresnet = OurVideoResNet()
#     x = torch.randn(2, 3, 4, 112, 112)
#     out = myresnet(x)
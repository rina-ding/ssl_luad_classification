import torchvision
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes, device, num_channel, backbone = torchvision.models.resnet18()):
        super().__init__()
        self.backbone = backbone
        self.backbone.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1])) # Removing the FC layer
        self.predictor = nn.Sequential(nn.Flatten(), nn.Dropout(p = 0.2), nn.Linear(512, num_classes))
        self.encoder = nn.Sequential(self.backbone, self.predictor)
    
    def forward(self, x):
        output = self.encoder(x)
        
        return output

class ResNetEncoder(ResNet):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(block = BasicBlock, layers = [2, 2, 2, 2])
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(p = 0.3), nn.Linear(512, 6))

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            self.fc
        ]

    def forward(self, x):
        stages = self.get_stages()
        x = stages[0](x)
        x = stages[1](x)
        x = stages[2](x)
        x = stages[3](x)
        x = stages[4](x)
        x = stages[5](x)
        x = stages[6](x)
        x = stages[7](x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

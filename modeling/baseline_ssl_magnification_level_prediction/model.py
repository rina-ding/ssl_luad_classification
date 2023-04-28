import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes, backbone = torchvision.models.resnet18(), num_input_channel = 3):
        super().__init__()
        self.backbone = backbone
        self.backbone.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1])) # Removing the FC layer
        self.predictor = nn.Sequential(nn.Flatten(), nn.Dropout(p = 0.2), nn.Linear(512, num_classes)) # 512 for resnet18, 2048 for resnet50
        self.encoder = nn.Sequential(self.backbone, self.predictor)
    
    def forward(self, x):
        output = self.encoder(x)
        
        return output

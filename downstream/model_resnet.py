import torchvision
import torch
import torch.nn as nn

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes, backbone = torchvision.models.resnet18(pretrained=False)):
        super().__init__()
        self.backbone = backbone
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1])) 
        self.predictor = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.2), nn.Linear(512, num_classes))
        self.encoder = nn.Sequential(self.backbone, self.predictor)
    
    def forward(self, x):
        output = self.encoder(x)
        
        return output
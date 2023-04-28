
import torch.nn as nn
import segmentation_models_pytorch as smp

class EncoderDecoderUNET(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderDecoderUNET, self).__init__()
        self.unet = smp.Unet(encoder_name="resnet18", in_channels=in_channel, classes=out_channel)

    def forward(self, x):
        unet_output = self.unet(x)
        return unet_output


import torch.nn as nn
import torchvision
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights


class CustomeEfficientNet(nn.Module):
    def __init__(self, model_name = 'efficientnet_v2_l', pretrained = True, num_classes = 10):
        super().__init__()
        if model_name == 'efficientnet_v2_l':
            if pretrained:
                self.model = torchvision.models.efficientnet_v2_l(weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.efficientnet_v2_l(weights = None)
            self.transform = EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()

    
    def get_transform(self):
        return self.transform

    def forward(self, x):
        return self.model(x)
from models.efficientnet import CustomEfficientNet
from models.mobilenet import CustomMobileNet
import torch.nn as nn

class base_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None

    def forward(self, x):
        return self.model(x)

    def get_transform(self):
        return self.transform

def build_model(model_name: str, pretrained = True, n_classes = 10):
    if model_name == 'efficientnet_v2_l':
        return CustomEfficientNet(pretrained = pretrained, num_classes = n_classes)
    elif model_name == 'mobilenet_v2':
        return CustomMobileNet(pretrained = pretrained, num_classes = n_classes)

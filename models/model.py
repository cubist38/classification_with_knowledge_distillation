from models.efficientnet import CustomEfficientNet
from models.mobilenet import CustomMobileNet
import torch.nn as nn

def build_model(model_name: str, pretrained = True, n_classes = 10):
    if model_name == 'efficientnet_v2_l':
        return CustomEfficientNet(pretrained = pretrained, num_classes = n_classes)
    elif model_name == 'mobilenet_v2':
        return CustomMobileNet(pretrained = pretrained, num_classes = n_classes)

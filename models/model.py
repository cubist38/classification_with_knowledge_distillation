from models.efficientnet import CustomEfficientNet
from models.mobilenet import CustomMobileNet
import torch.nn as nn

def build_model(model_name: str, pretrained = True, n_classes = 10):
    model_type = model_name.split('_')[0]
    if model_type == 'efficientnet':
        return CustomEfficientNet(pretrained = pretrained, num_classes = n_classes)
    elif model_type == 'mobilenet':
        return CustomMobileNet(pretrained = pretrained, num_classes = n_classes)

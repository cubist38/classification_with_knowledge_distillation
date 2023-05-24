import torch.nn as nn
import torchvision
from torchvision.models.efficientnet import (EfficientNet_V2_L_Weights, 
                                                EfficientNet_V2_S_Weights,
                                                EfficientNet_B7_Weights,
                                                EfficientNet_B4_Weights,)

from models.base_model import BaseModel


class CustomEfficientNet(BaseModel):
    def __init__(self, model_name = 'efficientnet_v2_l', pretrained = True, num_classes = 10):
        super().__init__()
        if model_name == 'efficientnet_v2_l':
            if pretrained:
                self.model = torchvision.models.efficientnet_v2_l(weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.efficientnet_v2_l(weights = None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1280, out_features = num_classes, bias=True),
            )
            self.transform = EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
        elif model_name == 'efficientnet_v2_s':
            if pretrained:
                self.model = torchvision.models.efficientnet_v2_s(weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.efficientnet_v2_s(weights = None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features = num_classes, bias=True),
            )
            self.transform = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
        elif model_name == 'efficientnet_b7':
            if pretrained:
                self.model = torchvision.models.efficientnet_b7(weights = EfficientNet_B7_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.efficientnet_b7(weights = None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=2560, out_features = num_classes, bias=True),
            )
        elif model_name == 'efficientnet_b4':
            if pretrained:
                self.model = torchvision.models.efficientnet_b4(weights = EfficientNet_B4_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.efficientnet_b4(weights = None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1792, out_features = num_classes, bias=True),
            )
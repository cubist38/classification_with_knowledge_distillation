import torch.nn as nn
import torchvision
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights
from base_model import BaseModel


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

    
    def get_transform(self):
        return self.transform

    def forward(self, x):
        return self.model(x)
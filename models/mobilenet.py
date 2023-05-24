import torchvision
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from models.base_model import BaseModel
import torch.nn as nn

class CustomMobileNet(BaseModel):
    def __init__(self, model_name = 'mobilenetv2', pretrained = True, num_classes = 10):
        super().__init__()
        if model_name == 'mobilenet_v2':
            if pretrained:
                self.model = torchvision.models.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V2)
            else:
                self.model = torchvision.models.mobilenet_v2(weights = None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features = num_classes, bias=True),
            )
            self.transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
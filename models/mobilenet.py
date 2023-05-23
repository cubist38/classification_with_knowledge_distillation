import torchvision
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torch.nn as nn

class MyMobileNet(nn.Module):
    def __init__(self, model_name = 'mobilenetv2', weights = MobileNet_V2_Weights.IMAGENET1K_V2, num_classes = 10):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(weights = weights)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features = num_classes, bias=True),
        )
    
    def forward(self, x):
        return self.model(x)

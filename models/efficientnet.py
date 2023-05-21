from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class MyEfficientNet(nn.Module):
    def __init__(self, model_name = 'efficientnet-b0', num_classes = 10):
        super().__init__()
        self.model_name = model_name
        self.model = EfficientNet.from_pretrained(self.model_name, num_classes = num_classes)
    
    def image_size(self):
        return EfficientNet.get_image_size(self.model_name)

    def forward(self, x):
        return self.model(x)
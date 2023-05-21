from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, model_name = 'efficientnet-b0', num_classes = 10):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name, num_classes = num_classes)

    def forward(self, x):
        return self.model(x)
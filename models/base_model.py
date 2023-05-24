import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None

    def forward(self, x):
        assert self.model is not None, 'You have to define model in your custom model class'
        return self.model(x)

    def get_transform(self):
        assert self.transform is not None, 'You have to define model in your custom model class'
        return self.transform
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None

    def forward(self, x):
        return self.model(x)

    def get_transform(self):
        return self.transform
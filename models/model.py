import torchvision
from efficientnet import MyEfficientNet
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


def build_model(model_name: str, n_classes = 10):
    if model_name == 'efficientnet-b4':
        return MyEfficientNet(model_name = model_name, num_classes = n_classes)
    elif model_name == 'mobilenet_v2':
        return torchvision.models.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V2)
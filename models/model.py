from models.efficientnet import MyEfficientNet
from models.mobilenet import MyMobileNet

def build_model(model_name: str, n_classes = 10):
    if model_name == 'efficientnet-b4':
        return MyEfficientNet(model_name = model_name, num_classes = n_classes)
    elif model_name == 'mobilenet_v2':
        return MyMobileNet(num_classes = n_classes)
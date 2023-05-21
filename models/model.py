from efficientnet import *

def build_model(model_name: str, n_classes = 10):
    return MyEfficientNet(model_name = model_name, num_classes = n_classes)
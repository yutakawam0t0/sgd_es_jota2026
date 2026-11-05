from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.wideresnet import WideResNet40_4, WideResNet16_8, WideResNet28_10, WideResNet28_12


def select_model(model_name, num_classes=100):
    """Select model based on specified model name"""
    if model_name == "resnet18":
        return resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        return resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        return resnet50(num_classes=num_classes)
    elif model_name == "resnet101":
        return resnet101(num_classes=num_classes)
    elif model_name == "resnet152":
        return resnet152(num_classes=num_classes)
    elif model_name == "WideResNet40_4":
        return WideResNet40_4(num_classes=num_classes)
    elif model_name == "WideResNet16_8":
        return WideResNet16_8(num_classes=num_classes)
    elif model_name == "WideResNet28_10":
        return WideResNet28_10(num_classes=num_classes)
    elif model_name == "WideResNet28_12":
        return WideResNet28_12(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name {model_name}")

"""Model factories shared by training and checkpoint-only evaluation."""
from RAHFL.Network.Models_Def.resnet import (
    ResNet10, ResNet12, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
)
from RAHFL.Network.Models_Def.shufflenet import ShuffleNetG2, ShuffleNetG3
from RAHFL.Network.Models_Def.mobilnet_v2 import MobileNetV2

MODEL_FACTORIES = {
    "ResNet10": ResNet10, "ResNet12": ResNet12, "ResNet18": ResNet18,
    "ResNet34": ResNet34, "ResNet50": ResNet50, "ResNet101": ResNet101,
    "ResNet152": ResNet152, "ShuffleNet": ShuffleNetG2,
    "ShuffleNetG2": ShuffleNetG2, "ShuffleNetG3": ShuffleNetG3,
    "Mobilenetv2": MobileNetV2, "MobileNetV2": MobileNetV2,
}
DEFAULT_MODELS = ("ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2")


def create_model(name, num_classes=10):
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name](num_classes=num_classes)

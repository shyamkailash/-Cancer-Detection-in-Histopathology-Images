"""
ResNet-18 model for PCam binary classification.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(num_classes: int = 2, pretrained: bool = True):
    """
    Create a ResNet-18 classifier for histopathology images.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    # Replace the original 1000-class ImageNet classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
from __future__ import annotations
import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(num_classes: int, feature_extract: bool = True):
    """
    EfficientNet-B0 with ImageNet weights and replace final classifier for `num_classes`.
    If feature_extract=True, only the final linear layer is trainable.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if feature_extract:
        for name, p in model.named_parameters():
            p.requires_grad = ("classifier.1" in name)

    return model

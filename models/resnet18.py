from __future__ import annotations
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes: int, feature_extract: bool = True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    if feature_extract:
        for name, p in model.named_parameters():
            p.requires_grad = ("fc" in name)
    return model

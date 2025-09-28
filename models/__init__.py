from __future__ import annotations
from typing import Callable, Dict

from .resnet18 import build_resnet18
from .efficientnet_b0 import build_efficientnet_b0
from .hog_svm import build_hog_svm

# Simple registry of model builders
MODEL_REGISTRY = {
    "resnet18": build_resnet18,
    "efficientnet_b0": build_efficientnet_b0,
    "hog_svm": build_hog_svm,
}

__all__ = ["MODEL_REGISTRY"]

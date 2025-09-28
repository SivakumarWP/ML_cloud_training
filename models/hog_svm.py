# from __future__ import annotations
# import numpy as np
# from typing import Tuple
# from skimage.feature import hog
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
#
# # HOG params can be exposed via config if you want
# HOG_IMG_SIZE = (256, 256)      # (H, W)
# HOG_PIX_PER_CELL = (16, 16)
# HOG_CELLS_PER_BLOCK = (2, 2)
# HOG_ORIENTATIONS = 9
# HOG_BLOCK_NORM = "L2-Hys"
#
# def hog_extract(img_gray_uint8: np.ndarray) -> np.ndarray:
#     # img_gray_uint8 shape (H, W), dtype uint8
#     return hog(
#         img_gray_uint8,
#         orientations=HOG_ORIENTATIONS,
#         pixels_per_cell=HOG_PIX_PER_CELL,
#         cells_per_block=HOG_CELLS_PER_BLOCK,
#         block_norm=HOG_BLOCK_NORM,
#         feature_vector=True,
#     ).astype(np.float32)
#
# def build_hog_svm(num_classes: int, **kwargs) -> Pipeline:
#     return Pipeline([
#         ("scaler", StandardScaler(with_mean=True, with_std=True)),
#         ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced")),
#     ])

# models/hog_svm.py
# (Fix: provide build_hog_svm for MODEL_REGISTRY; keep HOG params/utilities consistent.)
# Based on your existing HOG helpers here. :contentReference[oaicite:1]{index=1}

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

# HOG params (keep identical in train & inference)
HOG_IMG_SIZE = (256, 256)  # (H, W)
_HOG_KW = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True,
)

def hog_extract(gray_uint8: np.ndarray) -> np.ndarray:
    return hog(gray_uint8, **_HOG_KW)

# def make_hog_svm_pipeline() -> Pipeline:
#     """
#     Default HOG+LinearSVC pipeline. StandardScaler(with_mean=False) is appropriate
#     for sparse-like HOG vectors. Class weights bias toward C0 to reduce C0→C1 flips.
#     """
#     return Pipeline([
#         ("scaler", StandardScaler(with_mean=False)),
#         ("clf", LinearSVC(C=1.0, class_weight={0: 1.5, 1: 1.0}, random_state=42)),
#     ])

def make_hog_svm_pipeline() -> Pipeline:
    """
    Default HOG+LinearSVC pipeline. StandardScaler(with_mean=False) is appropriate
    for sparse-like HOG vectors. Class weights bias toward C0 to reduce C0→C1 flips.
    Increased max_iter and tightened tol to address Liblinear convergence.
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LinearSVC(
            C=1.0,
            # class_weight={0: 1.5, 1: 1.0},
            class_weight={0: 2.0, 1: 1.0},
            random_state=42,
            max_iter=10000,   # <--- added
            tol=1e-4          # <--- added (optional tweak)
        )),
    ])


# === Required by models/__init__.py ===
def build_hog_svm(num_classes: int, **kwargs) -> Pipeline:
    """
    Registry-compatible builder. num_classes is accepted for signature parity,
    but the LinearSVC is binary; if you extend to >2 classes, adjust here.
    """
    return make_hog_svm_pipeline()

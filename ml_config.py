from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

# # ====== GLOBAL PATHS ======
# DATA_ROOT = Path("/Users/sivakumarvaradharajan/PycharmProjects/ML_development_pipeline/ml_model/models")#Path("data/processed")
# IMAGES_DIR = Path("/Users/sivakumarvaradharajan/PycharmProjects/ML_development_pipeline/ml_model/models/dummy_training")#DATA_ROOT / "ML_processed_image"
# METADATA_TSV = Path("/Users/sivakumarvaradharajan/PycharmProjects/ML_development_pipeline/ml_model/models/dummy_training/")#DATA_ROOT / "metadata.tsv"
# OUTPUTS_DIR = Path("outputs")

# point DATA_ROOT anywhere you like (optional)
DATA_ROOT = Path("/Users/sivakumarvaradharajan/PycharmProjects/ML_development_pipeline/ml_model/models/latest")

# folder that contains the images referenced by the metadata
IMAGES_DIR = Path("/Users/sivakumarvaradharajan/Desktop/Dataset_T3.0/final_bucket")
# ^ adjust if your images are directly under dummy_training (then drop /images)

# the *file* (not folder) for your metadata: csv or tsv both fine if loader supports it
METADATA_TSV = Path("/Users/sivakumarvaradharajan/Desktop/Dataset_T3.0/final_bucket/pair_log.csv")
#                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                <-- must be a file path, e.g. metadata.csv or metadata.tsv

OUTPUTS_DIR = Path("outputs")

# ====== CLASS / LABEL CONFIG ======
@dataclass
class ClassConfig:
    class_names: List[str] = field(default_factory=lambda: ["C0", "C1"])#["C0", "C1", "C2", "C3"])
    @property
    def class_map(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.class_names)}
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

# ====== TRAINING CONFIG (ResNet-18) ======
@dataclass
class TrainResNet18Config:
    images_dir: Path = IMAGES_DIR
    metadata_tsv: Path = METADATA_TSV
    out_dir: Path = OUTPUTS_DIR / "resnet18"
    img_size: int = 224
    batch_size: int = 64 #32
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    feature_extract: bool = True  # freeze backbone except final FC
    # split ratios
    split_train: float = 0.8
    split_val: float = 0.2
    split_test: float = 0.0

# SINGLETONS
CLASSES = ClassConfig()
RESNET18 = TrainResNet18Config()

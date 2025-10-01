# diff2_dataset.py
from pathlib import Path
from typing import List, Tuple, Dict
import csv, random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def load_items(metadata_csv: Path, images_dir: Path, class_map: Dict[str, int]) -> List[Tuple[Path, int]]:
    """
    Robust loader:
    - Supports columns: processed_seq_name, processed_seq_path, processed_path, written_path
    - If an absolute path from the CSV doesn't exist in this environment,
      it falls back to images_dir / basename(path).
    - Accepts labels from the 'label' column and maps via class_map.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv does not exist: {metadata_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

    items: List[Tuple[Path, int]] = []
    with metadata_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # file is comma-separated
        header = reader.fieldnames or []

        # possible path sources in priority order
        path_cols = [
            "processed_seq_name",   # filename like 4.png
            "processed_seq_path",   # absolute path on source machine
            "processed_path",       # absolute/relative path
            "written_path",         # absolute/relative path
        ]
        path_cols = [c for c in path_cols if c in header]
        if not path_cols:
            raise RuntimeError(f"No usable path column found. Header: {header}")

        for row in reader:
            lbl = (row.get("label") or "").strip()
            if lbl not in class_map:
                continue

            candidate_paths: List[Path] = []

            # 1) If we have a filename-only column, try images_dir / filename
            name = (row.get("processed_seq_name") or "").strip().strip("/\\")
            if name:
                candidate_paths.append(images_dir / name)

            # 2) For any path-like columns: try the path as-is, and also images_dir / basename
            for col in ("processed_seq_path", "processed_path", "written_path"):
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                p = Path(raw)
                candidate_paths.append(p)
                candidate_paths.append(images_dir / p.name)

            # pick the first candidate that exists
            p_final = next((p for p in candidate_paths if p.exists()), None)
            if p_final is None:
                continue  # skip rows whose files we can't resolve

            y = class_map[lbl]
            items.append((p_final, y))

    if not items:
        raise RuntimeError(
            "No items loaded. None of the resolved file paths exist. "
            "Check that your images are under IMAGES_DIR and filenames in the CSV match."
        )
    return items


def stratified_split(
    items: List[Tuple[Path, int]], train=0.7, val=0.15, test=0.15, seed: int = 42
):
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    rng = random.Random(seed)
    train_set: List[Tuple[Path, int]] = []
    val_set: List[Tuple[Path, int]] = []
    test_set: List[Tuple[Path, int]] = []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        if n_tr + n_va > n:
            n_va = max(0, n - n_tr)
        n_te = n - n_tr - n_va
        train_set.extend(lst[:n_tr])
        val_set.extend(lst[n_tr:n_tr+n_va])
        test_set.extend(lst[n_tr+n_va:])
    rng.shuffle(train_set); rng.shuffle(val_set); rng.shuffle(test_set)
    return train_set, val_set, test_set


class Diff2Dataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], img_size: int, train: bool):
        self.items = items
        if train:
            self.tfm = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("L").convert("RGB")
        x = self.tfm(img)
        return x, y

# train.py
# Unified trainer (ResNet18 / HOG+SVM) with:
# - Train/Validation only (NO TEST)
# - Letterbox preprocessing for DL models
# - Early stopping + best/last checkpoints
# - Macro & per-class metrics each epoch
# - MPS/CUDA-friendly AMP + optional cosine LR
# - New: --dropout, --loss {ce,focal,bce,wbce}, --focal-gamma, --pos-weight, --warmup-steps

from __future__ import annotations
import argparse, random, time, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
import joblib

from ml_config import CLASSES, RESNET18
from diff2_dataset import load_items, stratified_split
from models import MODEL_REGISTRY
from models.hog_svm import hog_extract, HOG_IMG_SIZE
from skimage.transform import resize as sk_resize


# ------------------ utils ------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------ Letterbox Dataset (DL models) ------------------
class LetterboxSquare:
    def __init__(self, target: int = 224, pad_color=(255, 255, 255)):
        self.target = int(target)
        self.pad_color = pad_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.target / max(w, h)
        new_w, new_h = int(round(w * s)), int(round(h * s))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        pad_l = (self.target - new_w) // 2
        pad_t = (self.target - new_h) // 2
        pad_r = self.target - new_w - pad_l
        pad_b = self.target - new_h - pad_t
        return ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=self.pad_color)


class Diff2LetterboxDataset(Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, items: List[Tuple[Path, int]], img_size: int, train: bool):
        self.items = items
        self.letterbox = LetterboxSquare(target=img_size, pad_color=(255, 255, 255))
        self.train = train

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("L").convert("RGB")
        img = self.letterbox(img)
        if self.train and random.random() < 0.20:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        return x, y


# ------------------ New: Loss helpers & dropout/warmup ------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)  # (N,)
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1).clamp_(1e-6, 1.0)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def make_final_dropout(model: nn.Module, p: float, num_classes: int):
    if p <= 0:
        return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p), nn.Linear(in_feat, num_classes))
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_feat = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p), nn.Linear(in_feat, num_classes))
        return model
    # Fallback: wrap last Linear
    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, nn.Linear):
            in_feat = mod.in_features
            parent = model
            *parents, last = name.split(".")
            for part in parents:
                parent = getattr(parent, part)
            setattr(parent, last, nn.Sequential(nn.Dropout(p), nn.Linear(in_feat, num_classes)))
            return model
    return model


def compute_pos_weight_from_labels(y: np.ndarray) -> float:
    num_pos = (y == 1).sum()
    num_neg = (y == 0).sum()
    if num_pos == 0:
        return 1.0
    return float(num_neg / max(1, num_pos))


def linear_warmup_lr(base_lrs, step_idx: int, warmup_steps: int):
    if warmup_steps <= 0:
        return base_lrs
    scale = min(1.0, (step_idx + 1) / float(warmup_steps))
    return [lr * scale for lr in base_lrs]


# ------------------ HOG helpers ------------------
def to_gray_uint8(p: Path, size_hw=HOG_IMG_SIZE) -> np.ndarray:
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != size_hw:
        arr = (sk_resize(arr, size_hw, preserve_range=True, anti_aliasing=True)).astype(np.uint8)
    return arr


def build_hog_dataset(items: List[Tuple[Path, int]]) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for p, y in items:
        g = to_gray_uint8(p, size_hw=HOG_IMG_SIZE)
        feats.append(hog_extract(g))
        labels.append(y)
    X = np.vstack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ------------------ Train/Eval (DL) ------------------
def train_epoch_torch(model, loader, optim, device, amp_device_type: str, criterion, warmup_steps: int, base_lrs: list, global_step: int):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    scaler = torch.cuda.amp.GradScaler() if amp_device_type == "cuda" else None

    for x, y in loader:
        x = x.to(device, non_blocking=(amp_device_type == "cuda"))
        y = y.to(device, non_blocking=(amp_device_type == "cuda"))
        optim.zero_grad(set_to_none=True)

        # Warmup step LRs
        if warmup_steps > 0:
            new_lrs = linear_warmup_lr(base_lrs, global_step, warmup_steps)
            for pg, lr in zip(optim.param_groups, new_lrs):
                pg["lr"] = lr

        use_amp = amp_device_type in ("cuda", "mps")
        ctx = torch.autocast(device_type=amp_device_type, dtype=torch.float16, enabled=use_amp)
        with ctx:
            logits = model(x)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                pos_logits = logits[:, 1] if logits.size(1) > 1 else logits.squeeze(1)
                loss = criterion(pos_logits, y.float())
                pred = (torch.sigmoid(pos_logits) >= 0.5).long()
            else:
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)

        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        else:
            loss.backward(); optim.step()

        loss_sum += float(loss.item()) * y.size(0)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        global_step += 1

    return loss_sum / max(1, total), correct / max(1, total), global_step


@torch.no_grad()
def evaluate_torch(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            pos_logits = logits[:, 1] if logits.size(1) > 1 else logits.squeeze(1)
            loss = criterion(pos_logits, y.float())
            pred = (torch.sigmoid(pos_logits) >= 0.5).long()
        else:
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
        loss_sum += float(loss.item()) * y.size(0)
        correct += int((pred == y).sum().item()); total += y.size(0)
        all_y.append(y.cpu().numpy()); all_p.append(pred.cpu().numpy())
    all_y = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    all_p = np.concatenate(all_p) if all_p else np.array([], dtype=np.int64)
    acc = correct / max(1, total)
    if all_y.size:
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_y, all_p, labels=list(range(len(CLASSES.class_names))),
            average="macro", zero_division=0
        )
        cm = confusion_matrix(all_y, all_p, labels=list(range(len(CLASSES.class_names))))
    else:
        prec = rec = f1 = 0.0
        cm = np.zeros((len(CLASSES.class_names), len(CLASSES.class_names)), dtype=int)
    return (loss_sum / max(1, total), acc, prec, rec, f1, cm)


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser("Trainer (train/val only)")
    ap.add_argument("--model", type=str, default="resnet18", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=RESNET18.lr)
    ap.add_argument("--weight-decay", type=float, default=RESNET18.weight_decay)
    ap.add_argument("--num-workers", type=int, default=RESNET18.num_workers)
    ap.add_argument("--seed", type=int, default=RESNET18.seed)
    ap.add_argument("--feature-extract", action="store_true", default=RESNET18.feature_extract)
    ap.add_argument("--img-size", type=int, default=RESNET18.img_size)
    ap.add_argument("--out-dir", type=str, default=str(RESNET18.out_dir.parent))
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--use-cosine", action="store_true")
    ap.add_argument("--amp", action="store_true")
    # New flags
    ap.add_argument("--dropout", type=float, default=0.0, help="Final-layer dropout prob; 0 disables.")
    ap.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "bce", "wbce"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--pos-weight", type=float, default=None, help="For wbce; if unset auto-compute from train labels.")
    ap.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps (0 disables).")
    # HOG-only param kept for completeness (not used in DL path)
    ap.add_argument("--target-recall", type=float, default=0.97)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    amp_device_type = device.type
    print(f"[INFO] Using device: {device}")

    images_dir = Path(RESNET18.images_dir)
    meta_tsv = Path(RESNET18.metadata_tsv)
    base_out = Path(args.out_dir); base_out.mkdir(parents=True, exist_ok=True)

    # Data split (TRAIN/VAL ONLY)
    items = load_items(meta_tsv, images_dir, CLASSES.class_map)
    tr, va, _ = stratified_split(
        items,
        train=RESNET18.split_train,
        val=1.0 - RESNET18.split_train,
        test=0.0,
        seed=args.seed
    )

    model_name = args.model.lower()

    # -------- HOG + SVM path (validation only) --------
    if model_name in ("hog_svm",):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        out_dir = base_out / "hog_svm"
        out_dir.mkdir(parents=True, exist_ok=True)

        X_tr, y_tr = build_hog_dataset(tr)
        X_va, y_va = build_hog_dataset(va)

        base = LinearSVC(
            C=1.0,
            class_weight={0: 2.0, 1: 1.0},
            random_state=42,
            max_iter=50000,
            tol=1e-3,
        )
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            CalibratedClassifierCV(base, method="sigmoid", cv=5),
        )
        clf.fit(X_tr, y_tr)

        print("=== Validation (default 0.5) ===")
        yva_pred_default = clf.predict(X_va)
        print(classification_report(y_va, yva_pred_default, target_names=CLASSES.class_names, digits=4))

        target_recall = float(args.target_recall)
        probs_va = clf.predict_proba(X_va)[:, 1]
        prec, rec, thr = precision_recall_curve(y_va, probs_va)
        prec, rec, thr = prec[:-1], rec[:-1], thr
        if np.any(rec >= target_recall):
            idxs = np.where(rec >= target_recall)[0]
            best_idx = idxs[np.argmax(prec[idxs])]
        else:
            f1 = (2 * prec * rec) / (prec + rec + 1e-12)
            best_idx = int(np.nanargmax(f1))
        prob_threshold = float(thr[best_idx])
        print(f"[VAL] chosen prob_threshold={prob_threshold:.4f} "
              f"(P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}) at target_recall={target_recall:.3f}")

        joblib.dump({
            "pipeline": clf,
            "class_names": CLASSES.class_names,
            "class_map": CLASSES.class_map,
            "prob_threshold": prob_threshold,
            "target_recall": target_recall,
        }, out_dir / "hog_svm_model.joblib")

        with (out_dir / "metrics.txt").open("w") as f:
            # use balanced accuracy proxy if you want; here we skip and only print reports
            f.write("best_val_acc=\n")

        return

    # -------- Torch models (resnet18 etc.) --------
    out_dir = base_out / model_name; out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Diff2LetterboxDataset(tr, img_size=args.img_size, train=True)
    val_ds   = Diff2LetterboxDataset(va, img_size=args.img_size, train=False)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_mem)

    # Build model
    factory = MODEL_REGISTRY[model_name]
    model = factory(num_classes=CLASSES.num_classes, feature_extract=args.feature_extract).to(device)

    # Optional final-layer dropout
    if args.dropout and args.dropout > 0.0:
        model = make_final_dropout(model, args.dropout, CLASSES.num_classes).to(device)

    # Optimizer / Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_cosine else None

    # Criterion
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.loss in ("bce", "wbce"):
        assert CLASSES.num_classes == 2, "--loss bce/wbce requires binary classes (num_classes == 2)."
        pos_weight_t = None
        if args.loss == "wbce":
            if args.pos_weight is not None:
                pw = float(args.pos_weight)
            else:
                train_labels = np.array([y for _, y in tr], dtype=np.int64)
                pw = compute_pos_weight_from_labels(train_labels)
            pos_weight_t = torch.tensor([pw], dtype=torch.float32, device=device)
            print(f"[INFO] WBCE pos_weight={float(pos_weight_t.item()):.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        raise ValueError(f"Unknown loss {args.loss}")

    # AMP scaler (optional)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None  # (unused but safe)

    # Training loop
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    global_step = 0
    best_val_acc = 0.0
    best_epoch = 0
    patience = args.early_stop_patience
    wait = 0
    best_path = out_dir / f"{model_name}_best.pt"
    last_path = out_dir / f"{model_name}_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, global_step = train_epoch_torch(
            model, train_loader, optimizer, device, amp_device_type,
            criterion, args.warmup-steps if hasattr(args, "warmup-steps") else args.warmup_steps,  # safety
            base_lrs, global_step
        )

        if scheduler is not None:
            scheduler.step()

        va_loss, va_acc, va_prec, va_rec, va_f1, va_cm = evaluate_torch(model, val_loader, device, criterion)
        dt = time.time() - t0

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} "
              f"(P {va_prec:.4f} R {va_rec:.4f} F1 {va_f1:.4f}) | {dt:.1f}s")

        # Save last checkpoint each epoch
        torch.save({
            "model": model.state_dict(),
            "class_names": CLASSES.class_names,
            "class_map": CLASSES.class_map
        }, last_path)

        # best checkpoint on val acc
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "class_names": CLASSES.class_names,
                "class_map": CLASSES.class_map
            }, best_path)
            wait = 0
        else:
            wait += 1

        # early stopping
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_acc={best_val_acc:.4f}).")
            break

    # Load best for final export
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    # Save metrics (VAL ONLY)
    with (out_dir / "metrics.txt").open("w") as f:
        f.write(f"best_val_acc={best_val_acc:.6f}\n")
        f.write(f"val_precision_macro={va_prec:.6f}\n")
        f.write(f"val_recall_macro={va_rec:.6f}\n")
        f.write(f"val_f1_macro={va_f1:.6f}\n")

    # Optionally dump confusion matrix
    with (out_dir / "val_confusion_matrix.txt").open("w") as f:
        f.write("labels=" + ",".join(CLASSES.class_names) + "\n")
        for row in va_cm:
            f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()

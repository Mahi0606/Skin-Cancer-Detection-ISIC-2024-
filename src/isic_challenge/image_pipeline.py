"""
image_pipeline.py — Phase 2 CNN training for ISIC 2024 Skin Cancer Detection.

Architecture:  EfficientNet-B4-NS  (NoisyStudent pretrained via timm)
Loss:          Focal Loss  FL(p_t) = −α_t · (1−p_t)^γ · log(p_t)
Optimizer:     AdamW  (weight-decoupled Adam, Loshchilov & Hutter 2017)
LR Schedule:   Linear Warmup (1 epoch) → Cosine Annealing
Augmentation:  Albumentations pipeline + MixUp
Data Format:   HDF5 (preferred) or JPEG directory
Validation:    pAUC (TPR ≥ 0.88) aligned with ISIC 2024 competition metric
Interpretability: Grad-CAM for TP/TN/FP/FN analysis
"""

from __future__ import annotations

import io
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    warnings.warn("timm not installed — install with `pip install timm`")

from isic_challenge.metrics import compute_pauc

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# § 1  FOCAL LOSS
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Sigmoid-activated Focal Loss for binary classification.

    Mathematics
    -----------
    Given probability  p = σ(logit),  let  p_t = p  if y=1  else  1-p.

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    The factor  (1 - p_t)^γ  down-weights well-classified (easy) examples so
    that the gradient budget concentrates on hard, rare positives.

    Parameters
    ----------
    gamma : float
        Focusing parameter γ ≥ 0.  γ=0 reduces to weighted BCE.
        γ=2 recommended (Lin et al., 2017, RetinaNet).
    alpha : float | None
        Prior probability of the positive class.
        If None, uses ``pos_weight`` passed at call time instead.
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Binary cross-entropy (numerically stable)
        if pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        # p_t : probability of "true" class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # Modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Class balance weight α_t
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# § 2  MIXUP AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp data augmentation (Zhang et al., 2018).

    Returns
    -------
    mixed_x       : λ·x_i + (1−λ)·x_j
    y_a, y_b      : original and permuted labels
    lam           : mixing coefficient drawn from Beta(α,α)
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    **criterion_kwargs,
) -> torch.Tensor:
    """Blended loss for MixUp: λ·L(y_a) + (1−λ)·L(y_b)."""
    return (
        lam * criterion(pred, y_a, **criterion_kwargs)
        + (1.0 - lam) * criterion(pred, y_b, **criterion_kwargs)
    )


# ──────────────────────────────────────────────────────────────────────────────
# § 3  AUGMENTATION PIPELINES
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 256) -> A.Compose:
    """
    Training augmentation — 9 Albumentations transforms justified for dermoscopy:

    1.  HorizontalFlip / VerticalFlip  — no orientation constraint on lesions
    2.  ShiftScaleRotate               — random positioning during TBP capture
    3.  RandomBrightnessContrast       — lighting & scanner variation
    4.  HueSaturationValue             — color calibration differences
    5.  CoarseDropout                  — regularization, forces attention spread
    6.  GaussianBlur                   — autofocus / motion artifacts
    7.  Sharpen (ImageFilter)          — some scanners over-sharpen
    8.  GridDistortion                 — minor lens distortion
    9.  Normalize (ImageNet μ, σ)      — align with pretrained backbone stats
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=90,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=img_size // 16,
                max_width=img_size // 16,
                fill_value=0,
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                ],
                p=0.3,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Validation: deterministic resize + normalize only."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def visualize_augmentations(
    dataset,
    n_samples: int = 6,
    n_aug: int = 4,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> "matplotlib.figure.Figure":
    """
    Render a (n_samples × n_aug) grid of training augmentations.

    Each row shows a different sample; each column is an independent augmentation
    of that sample.  Useful for verifying that transforms are visually reasonable
    before full training.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx_list = rng.choice(len(dataset), size=n_samples, replace=False).tolist()

    # Temporarily switch to train augmentation
    orig_transform = dataset.transform
    dataset.transform = get_train_transforms(
        getattr(dataset, "img_size", 256)
    )

    fig, axes = plt.subplots(n_samples, n_aug, figsize=(n_aug * 3, n_samples * 3))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    unnorm_mean = np.array([0.485, 0.456, 0.406])
    unnorm_std = np.array([0.229, 0.224, 0.225])

    for row, sample_idx in enumerate(idx_list):
        for col in range(n_aug):
            img_tensor, label, _ = dataset[sample_idx]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = img_np * unnorm_std + unnorm_mean
            img_np = np.clip(img_np, 0, 1)
            ax = axes[row, col]
            ax.imshow(img_np)
            if col == 0:
                ax.set_ylabel(
                    f"{'Malignant' if label == 1 else 'Benign'}",
                    fontsize=9,
                    rotation=0,
                    ha="right",
                    va="center",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Training Augmentation Samples — Each Column is an Independent Aug",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    dataset.transform = orig_transform
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# § 4  DATASETS  (HDF5-first with JPEG fallback)
# ──────────────────────────────────────────────────────────────────────────────

class IsicDataset(Dataset):
    """
    Loads ISIC 2024 lesion images from HDF5 (preferred) or JPEG files.

    The official ISIC 2024 dataset ships as ``train-image.hdf5``.
    Each HDF5 key is an ``isic_id`` mapping to the raw JPEG byte-sequence stored
    as a uint8 array.  This reader avoids extracting all images to disk.

    If ``hdf5_path`` is None or the file is missing, falls back to individual
    JPEG files at ``img_dir/{isic_id}.jpg``.

    Parameters
    ----------
    df           : DataFrame with columns ``isic_id`` and ``target``
    hdf5_path    : path to ``train-image.hdf5`` (or None)
    img_dir      : directory containing ``<isic_id>.jpg`` files (fallback)
    transform    : Albumentations ``Compose`` transform
    img_size     : target pixel resolution
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hdf5_path: Optional[str],
        img_dir: Optional[str],
        transform: Optional[A.Compose] = None,
        img_size: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

        # Try to open HDF5 once to validate; keep closed between __getitem__ calls
        # (forking-safe: each worker opens its own handle)
        self._use_hdf5 = (
            hdf5_path is not None and os.path.exists(hdf5_path)
        )
        self._hdf5_handle: Optional[h5py.File] = None

    # ── HDF5 helpers (fork-safe lazy open) ────────────────────────────────────

    def _open_hdf5(self) -> None:
        if self._hdf5_handle is None:
            self._hdf5_handle = h5py.File(self.hdf5_path, "r")

    def __del__(self):
        if self._hdf5_handle is not None:
            try:
                self._hdf5_handle.close()
            except Exception:
                pass

    # ── Core interface ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, isic_id: str) -> np.ndarray:
        """Return H×W×3 uint8 numpy array."""
        if self._use_hdf5:
            self._open_hdf5()
            try:
                raw_bytes = self._hdf5_handle[isic_id][()]  # uint8 JPEG bytes
                img = np.array(Image.open(io.BytesIO(raw_bytes.tobytes())))
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 4:
                    img = img[:, :, :3]
                return img
            except KeyError:
                pass  # fall through to JPEG file

        # JPEG fallback
        if self.img_dir is not None:
            path = os.path.join(self.img_dir, f"{isic_id}.jpg")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Last resort: blank image (avoids crash on missing file)
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        isic_id = str(row["isic_id"])
        label = float(row["target"])

        img = self._load_image(isic_id)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, torch.tensor(label, dtype=torch.float32), isic_id


# ──────────────────────────────────────────────────────────────────────────────
# § 5  EfficientNet-B4-NS  MODEL
# ──────────────────────────────────────────────────────────────────────────────

class EfficientNetClassifier(nn.Module):
    """
    Binary classifier wrapping a timm EfficientNet backbone.

    Head architecture:
        Global-Average-Pool  →  Dropout(p)  →  Linear(features, 1)

    The final linear layer outputs a raw logit (no sigmoid) for compatibility
    with FocalLoss / BCEWithLogitsLoss.

    Parameters
    ----------
    model_name : str
        Any timm model name, e.g. ``'tf_efficientnet_b4_ns'``.
    pretrained : bool
        Whether to initialise from pretrained ImageNet weights.
    dropout : float
        Dropout rate before the classification head.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b4_ns",
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("Install timm: pip install timm")

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,        # remove original head
            global_pool="avg",   # built-in global average pooling
        )
        n_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)           # (B, n_features)
        logits = self.head(features).squeeze(-1)  # (B,)
        return logits

    def get_cam_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_maps, logits) for Grad-CAM computation."""
        # Store activations from the last conv block
        target_layer = self.backbone.conv_head \
            if hasattr(self.backbone, "conv_head") \
            else list(self.backbone.children())[-2]

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def fwd_hook(module, inp, out):
            activations.append(out)

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        logits = self.forward(x)
        h_fwd.remove()
        h_bwd.remove()

        return activations[0] if activations else None, logits


# ──────────────────────────────────────────────────────────────────────────────
# § 6  GRAD-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Map (Selvaraju et al., 2017).

    Produces a saliency map showing which spatial regions of the lesion image
    most influenced the model's malignancy prediction.

    Usage
    -----
    ::

        cam = GradCAM(model, target_layer=model.backbone.conv_head)
        heatmap = cam(image_tensor)   # (H, W) float in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._hook_fwd = target_layer.register_forward_hook(self._save_activation)
        self._hook_bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._hook_fwd.remove()
        self._hook_bwd.remove()

    def __call__(
        self,
        image: torch.Tensor,
        class_idx: int = 0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        image     : (1, C, H, W) tensor, normalised
        class_idx : 0 for binary sigmoid output

        Returns
        -------
        heatmap   : (H, W) float in [0, 1]
        """
        self.model.zero_grad()
        image = image.requires_grad_(True)

        output = self.model(image)           # scalar logit
        output.backward(torch.ones_like(output))

        # α_k = global-average-pooled gradient
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (h, w)
        cam = F.relu(cam)

        # Resize to input resolution
        h, w = image.shape[-2:]
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max > cam_min:
            cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
        else:
            cam_resized = np.zeros_like(cam_resized)

        return cam_resized


def overlay_gradcam(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay heatmap on the original image (uint8 RGB output)."""
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    overlay = (1 - alpha) * image_np + alpha * colored_rgb
    return overlay.astype(np.uint8)


def plot_gradcam_panel(
    model: nn.Module,
    dataset: IsicDataset,
    indices: List[int],
    target_layer: nn.Module,
    device: torch.device,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Render a panel of Grad-CAM overlays for selected samples.

    Each column shows: original image | heatmap overlay | sigmoid score.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    cam = GradCAM(model, target_layer)
    model.eval()

    unnorm_mean = np.array([0.485, 0.456, 0.406])
    unnorm_std = np.array([0.229, 0.224, 0.225])

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        img_tensor, label, isic_id = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        heatmap = cam(inp)

        # Unnormalise image for display
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * unnorm_std + unnorm_mean
        img_np = np.clip(img_np, 0, 1)

        overlay = overlay_gradcam((img_np * 255).astype(np.uint8), heatmap)

        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(
            f"{'Malignant' if label == 1 else 'Benign'} | ID: {isic_id[:8]}",
            fontsize=8,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Grad-CAM  | Score: {score:.3f}", fontsize=8)
        axes[i, 1].axis("off")

        if titles and i < len(titles):
            fig.text(
                0.02,
                1 - (i + 0.5) / n,
                titles[i],
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle("Grad-CAM: Original (left) | Saliency Overlay (right)", fontsize=12)
    plt.tight_layout()

    cam.remove_hooks()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# § 7  LEARNING-RATE SCHEDULE  (Warmup + Cosine)
# ──────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup (``warmup_steps`` steps → base LR)
    followed by cosine annealing over remaining steps.

    This prevents large gradient updates at the start of fine-tuning when
    the randomly-initialised classification head interacts with the pretrained
    backbone.

    Parameters
    ----------
    optimizer    : torch Optimizer
    warmup_steps : number of steps in warmup phase (usually 1 epoch)
    total_steps  : total training steps
    eta_min      : minimum LR at end of cosine cycle
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self._step = 0
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        lrs = self._get_lrs()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def _get_lrs(self) -> List[float]:
        s = self._step
        ws = self.warmup_steps
        ts = self.total_steps
        lrs = []
        for base_lr in self._base_lrs:
            if s <= ws:
                lr = base_lr * s / max(ws, 1)
            else:
                progress = (s - ws) / max(ts - ws, 1)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                    1 + np.cos(np.pi * progress)
                )
            lrs.append(lr)
        return lrs

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ──────────────────────────────────────────────────────────────────────────────
# § 8  TRAINING LOOP  (single fold)
# ──────────────────────────────────────────────────────────────────────────────

def _build_dataloader(
    df: pd.DataFrame,
    hdf5_path: Optional[str],
    img_dir: Optional[str],
    transform: A.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    img_size: int,
) -> DataLoader:
    ds = IsicDataset(
        df=df,
        hdf5_path=hdf5_path,
        img_dir=img_dir,
        transform=transform,
        img_size=img_size,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,  # drop last incomplete batch only during training
    )


def fit_image_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg,                     # TrainingConfig
    paths: Dict[str, str],
    device: Optional[torch.device] = None,
    return_model: bool = False,
) -> Dict[str, Any]:
    """
    Train an EfficientNet-B4-NS model on one cross-validation fold.

    Returns
    -------
    dict with keys:
        'oof_preds'   : np.ndarray of sigmoid scores for val_df rows
        'val_pAUC'    : partial AUC (TPR ≥ 0.88) on validation set
        'val_AUC'     : full ROC-AUC on validation set
        'history'     : list of per-epoch dicts {epoch, train_loss, val_pAUC, val_auc, lr}
        'model'       : trained model (only if return_model=True)
        'best_epoch'  : epoch with best val_pAUC
        'n_train'     : number of training samples used
        'n_val'       : number of validation samples
    """
    from sklearn.metrics import roc_auc_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Negative subsampling (training set only) ───────────────────────────
    neg_mask = train_df["target"] == 0
    pos_mask = ~neg_mask
    neg_df = train_df[neg_mask].sample(
        frac=cfg.img_neg_subsample_fraction,
        random_state=cfg.seed + fold,
    )
    sampled_train_df = pd.concat([train_df[pos_mask], neg_df]).sample(
        frac=1.0, random_state=cfg.seed + fold
    )

    n_pos = pos_mask.sum()
    n_neg = len(neg_df)
    imbalance_ratio = n_neg / max(n_pos, 1)

    # ── 2. pos_weight for focal loss ─────────────────────────────────────────
    pos_weight = torch.tensor([imbalance_ratio], dtype=torch.float32).to(device)

    # ── 3. Data loaders ───────────────────────────────────────────────────────
    train_loader = _build_dataloader(
        df=sampled_train_df,
        hdf5_path=paths.get("train_hdf5"),
        img_dir=paths.get("train_img_dir"),
        transform=get_train_transforms(cfg.img_size),
        batch_size=cfg.img_batch_size,
        shuffle=True,
        num_workers=cfg.img_num_workers,
        pin_memory=cfg.img_pin_memory if torch.cuda.is_available() else False,
        img_size=cfg.img_size,
    )
    val_loader = _build_dataloader(
        df=val_df,
        hdf5_path=paths.get("train_hdf5"),
        img_dir=paths.get("train_img_dir"),
        transform=get_val_transforms(cfg.img_size),
        batch_size=cfg.img_batch_size * 2,
        shuffle=False,
        num_workers=cfg.img_num_workers,
        pin_memory=False,
        img_size=cfg.img_size,
    )

    # ── 4. Model, loss, optimizer, schedule ──────────────────────────────────
    model = EfficientNetClassifier(
        model_name=cfg.img_model,
        pretrained=True,
        dropout=cfg.img_dropout,
    ).to(device)

    criterion = FocalLoss(gamma=cfg.img_focal_gamma)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.img_lr,
        weight_decay=cfg.img_weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg.img_epochs * steps_per_epoch
    warmup_steps = cfg.img_warmup_epochs * steps_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=cfg.img_lr * 0.01,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ── 5. Training epochs ────────────────────────────────────────────────────
    best_pAUC = -1.0
    best_state = None
    patience_counter = 0
    history: List[Dict] = []

    epoch_bar = tqdm(
        range(1, cfg.img_epochs + 1),
        desc=f"  Fold {fold + 1}",
        leave=False,
        disable=not cfg.img_show_progress,
    )

    for epoch in epoch_bar:
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"    Ep{epoch:02d} train",
            leave=False,
            disable=not cfg.img_show_progress,
        )

        for images, labels, _ in batch_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # MixUp
            if cfg.img_mixup_alpha > 0.0:
                images, lbl_a, lbl_b, lam = mixup_data(
                    images, labels, alpha=cfg.img_mixup_alpha
                )
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = mixup_criterion(
                        criterion, logits, lbl_a, lbl_b, lam,
                        pos_weight=pos_weight,
                    )
            else:
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = criterion(logits, labels, pos_weight=pos_weight)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_accum += loss.item()
            n_batches += 1

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_train_loss = train_loss_accum / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        all_logits: List[float] = []
        all_labels: List[float] = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                all_logits.extend(logits.cpu().float().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        val_scores = torch.sigmoid(torch.tensor(all_logits)).numpy()
        val_labels = np.array(all_labels)

        val_pauc = compute_pauc(val_labels, val_scores)
        try:
            val_auc = roc_auc_score(val_labels, val_scores)
        except ValueError:
            val_auc = 0.5

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        record = dict(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_pAUC=val_pauc,
            val_AUC=val_auc,
            lr=current_lr,
            elapsed_s=elapsed,
        )
        history.append(record)

        # ── Early stopping ────────────────────────────────────────────────────
        if val_pauc > best_pAUC:
            best_pAUC = val_pauc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_bar.set_postfix(
            pAUC=f"{val_pauc:.4f}",
            best=f"{best_pAUC:.4f}",
            pat=patience_counter,
        )

        if cfg.img_log_each_epoch:
            print(
                f"  Fold {fold+1} | Ep {epoch:02d}/{cfg.img_epochs} | "
                f"loss={avg_train_loss:.4f} | "
                f"pAUC={val_pauc:.4f} | AUC={val_auc:.4f} | "
                f"LR={current_lr:.2e} | {elapsed:.0f}s"
            )

        if patience_counter >= cfg.img_early_stop_patience:
            print(f"  → Early stopping at epoch {epoch}")
            break

    # ── 6. Restore best weights → final OOF predictions ──────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    oof_logits: List[float] = []
    with torch.no_grad():
        for images, _, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
            oof_logits.extend(logits.cpu().float().numpy().tolist())

    oof_preds = torch.sigmoid(torch.tensor(oof_logits)).numpy()

    # ── 7. Save checkpoint ────────────────────────────────────────────────────
    os.makedirs(paths.get("model_dir", "outputs/models"), exist_ok=True)
    ckpt_path = os.path.join(
        paths.get("model_dir", "outputs/models"),
        f"phase2_{cfg.img_model}_fold{fold}.pt",
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_pAUC": best_pAUC,
            "cfg_model": cfg.img_model,
            "cfg_img_size": cfg.img_size,
        },
        ckpt_path,
    )

    result = dict(
        oof_preds=oof_preds,
        val_pAUC=best_pAUC,
        val_AUC=val_auc,
        history=history,
        best_epoch=best_epoch,
        n_train=len(sampled_train_df),
        n_val=len(val_df),
    )
    if return_model:
        result["model"] = model

    return result


# ──────────────────────────────────────────────────────────────────────────────
# § 9  FULL CROSS-VALIDATION  LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_image_cv(
    df: pd.DataFrame,
    fold_iterator,           # iterable of (train_idx, val_idx) arrays
    cfg,                     # TrainingConfig
    paths: Dict[str, str],
    device: Optional[torch.device] = None,
    verbose: bool = True,
    skip_folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run full K-fold cross-validation for the image model.

    Parameters
    ----------
    df            : full dataset DataFrame with columns 'isic_id', 'target', 'patient_id'
    fold_iterator : list of (train_idx, val_idx) — use the same folds as Phase 1
    cfg           : TrainingConfig
    paths         : dict from cfg.paths(root)
    device        : torch device (auto-detected if None)

    Returns
    -------
    dict with keys:
        'img_oof'     : float64 array (len == len(df)) of OOF sigmoid scores
        'fold_results': list of per-fold result dicts from fit_image_fold
        'oof_pAUC'    : pAUC computed over the full OOF array
        'oof_AUC'     : AUC computed over the full OOF array
    """
    from sklearn.metrics import roc_auc_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if skip_folds is None:
        skip_folds = []

    img_oof = np.zeros(len(df), dtype=np.float64)
    fold_results: List[Dict] = []

    fold_bar = tqdm(
        list(enumerate(fold_iterator)),
        desc="Phase 2 — Image CV",
        disable=not verbose,
    )

    for fold, (train_idx, val_idx) in fold_bar:

        if fold in skip_folds:
            print(f"⏭️ Skipping fold {fold+1}/{cfg.n_folds} (already trained)")
            
            ckpt_path = os.path.join(
                paths.get("model_dir", "outputs/models"),
                f"phase2_{cfg.img_model}_fold{fold}.pt",
            )
            ckpt = torch.load(ckpt_path, map_location=device)

            model = EfficientNetClassifier(
                model_name=ckpt["cfg_model"],
                pretrained=False,
                dropout=0.4,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            val_df = df.iloc[val_idx].reset_index(drop=True)
            val_loader = _build_dataloader(
                df=val_df,
                hdf5_path=paths.get("train_hdf5"),
                img_dir=paths.get("train_img_dir"),
                transform=get_val_transforms(ckpt.get("cfg_img_size", cfg.img_size)),
                batch_size=cfg.img_batch_size * 2,
                shuffle=False,
                num_workers=cfg.img_num_workers,
                pin_memory=False,
                img_size=ckpt.get("cfg_img_size", cfg.img_size),
            )

            oof_logits = []
            with torch.no_grad():
                for images, _, _ in val_loader:
                    images = images.to(device, non_blocking=True)
                    logits = model(images)
                    oof_logits.extend(logits.cpu().float().numpy().tolist())

            oof_preds = torch.sigmoid(torch.tensor(oof_logits)).numpy()
            img_oof[val_idx] = oof_preds  # ← fills the zeros for skipped folds

            fold_results.append({
                "oof_preds": oof_preds,
                "val_pAUC": ckpt.get("best_pAUC", float("nan")),
                "val_AUC": float("nan"),   # not stored in ckpt, acceptable
                "history": [],
                "best_epoch": ckpt.get("best_epoch", -1),
                "n_train": -1,
                "n_val": len(val_idx),
            })
            continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(
            f"\n{'─'*60}\n"
            f"  Fold {fold + 1}/{cfg.n_folds} | "
            f"train={len(train_idx):,} val={len(val_idx):,}\n"
            f"{'─'*60}"
        )

        result = fit_image_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            cfg=cfg,
            paths=paths,
            device=device,
        )
        fold_results.append(result)

        img_oof[val_idx] = result["oof_preds"]

        fold_bar.set_postfix(
            pAUC=f"{result['val_pAUC']:.4f}",
            AUC=f"{result['val_AUC']:.4f}",
        )

    oof_pauc = compute_pauc(df["target"].values, img_oof)
    try:
        oof_auc = roc_auc_score(df["target"].values, img_oof)
    except ValueError:
        oof_auc = 0.5

    if verbose:
        print(
            f"\n{'='*60}\n"
            f"  Image CV complete\n"
            f"  OOF pAUC = {oof_pauc:.4f}\n"
            f"  OOF AUC  = {oof_auc:.4f}\n"
            f"{'='*60}"
        )

    return dict(
        img_oof=img_oof,
        fold_results=fold_results,
        oof_pAUC=oof_pauc,
        oof_AUC=oof_auc,
    )


# ──────────────────────────────────────────────────────────────────────────────
# § 10  LATE FUSION UTILITY  (kept for Phase 3 compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def optimize_late_fusion(
    tabular_oof: np.ndarray,
    img_oof: np.ndarray,
    y: np.ndarray,
    n_steps: int = 101,
) -> Tuple[float, float, float]:
    """
    Grid-search for optimal weight w in:
        final = w * tabular_oof + (1 - w) * img_oof

    Optimises pAUC (TPR ≥ 0.88).

    Returns
    -------
    (best_w, best_pAUC, tabular-only_pAUC)
    """
    tabular_only_pauc = compute_pauc(y, tabular_oof)

    best_w = 1.0
    best_pauc = tabular_only_pauc

    for w in np.linspace(0.0, 1.0, n_steps):
        blend = w * tabular_oof + (1.0 - w) * img_oof
        pauc = compute_pauc(y, blend)
        if pauc > best_pauc:
            best_pauc = pauc
            best_w = w

    return best_w, best_pauc, tabular_only_pauc


# ──────────────────────────────────────────────────────────────────────────────
# § 11  LEARNING CURVE PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    fold_results: List[Dict],
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot training loss, validation pAUC, and LR schedule across all folds.

    Each fold is shown as a separate line; the mean ± 1 std band is overlaid
    in grey for pAUC.
    """
    import matplotlib.pyplot as plt

    # ADD: filter out skipped folds with empty history
    active_results = [res for res in fold_results if res.get("history")]
    
    if not active_results:
        print("No training history available to plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.tab10

    pauc_curves = []

    for i, res in enumerate(active_results):
        hist = res["history"]
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        val_pauc = [h["val_pAUC"] for h in hist]
        lrs = [h["lr"] for h in hist]
        color = cmap(i)

        axes[0].plot(epochs, train_loss, color=color, label=f"Fold {i+1}")
        axes[1].plot(epochs, val_pauc, color=color, label=f"Fold {i+1}")
        axes[2].plot(epochs, lrs, color=color, label=f"Fold {i+1}")

        pauc_curves.append(val_pauc)

    # Mean ± std band for pAUC
    max_ep = max(len(c) for c in pauc_curves)
    padded = [c + [c[-1]] * (max_ep - len(c)) for c in pauc_curves]
    arr = np.array(padded)
    mean_pauc = arr.mean(axis=0)
    std_pauc = arr.std(axis=0)
    ep_range = np.arange(1, max_ep + 1)
    axes[1].fill_between(
        ep_range,
        mean_pauc - std_pauc,
        mean_pauc + std_pauc,
        alpha=0.2,
        color="grey",
        label="Mean ± 1 std",
    )

    for ax, title, ylabel in zip(
        axes,
        ["Training Loss (Focal)", "Validation pAUC (TPR ≥ 0.88)", "Learning Rate"],
        ["Loss", "pAUC", "LR"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"EfficientNet-B4-NS — Learning Curves ({len(fold_results)}-Fold CV)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_lr_schedule(
    cfg,
    steps_per_epoch: int = 100,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Visualise the Warmup + CosineAnnealing LR schedule."""
    import matplotlib.pyplot as plt

    optimizer = AdamW([torch.zeros(1)], lr=cfg.img_lr)
    total_steps = cfg.img_epochs * steps_per_epoch
    warmup_steps = cfg.img_warmup_epochs * steps_per_epoch

    sched = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    lrs = []
    for _ in range(total_steps):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(1, total_steps + 1)
    ax.plot(steps / steps_per_epoch, lrs, color="steelblue", linewidth=2)
    ax.axvline(
        x=cfg.img_warmup_epochs,
        color="tomato",
        linestyle="--",
        label=f"Warmup end (epoch {cfg.img_warmup_epochs})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(
        f"Linear Warmup ({cfg.img_warmup_epochs} ep) → Cosine Annealing\n"
        f"Base LR = {cfg.img_lr:.1e}  |  η_min = {cfg.img_lr * 0.01:.1e}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

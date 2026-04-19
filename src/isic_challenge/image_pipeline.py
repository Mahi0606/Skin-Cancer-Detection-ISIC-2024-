"""Phase 2 — EfficientNet image training (patient-aligned CV) + late fusion with tabular OOF."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sklearn.metrics import roc_auc_score

from isic_challenge.metrics import compute_pauc

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):  # type: ignore[misc]
        return it


def _log(msg: str) -> None:
    from datetime import datetime

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:
    A = None
    ToTensorV2 = None
    _ALBUMENTATIONS_ERR = e
else:
    _ALBUMENTATIONS_ERR = None

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset

    import timm
except ImportError as e:
    torch = None
    _TORCH_ERR = e
else:
    _TORCH_ERR = None


def _require_torch_stack() -> None:
    if torch is None:
        raise ImportError(
            "Phase 2 image training requires PyTorch and timm. "
            "Install with: pip install -e '.[image]'"
        ) from _TORCH_ERR
    if A is None or ToTensorV2 is None:
        raise ImportError(
            "Install albumentations for image augmentations: pip install -e '.[image]'"
        ) from _ALBUMENTATIONS_ERR


def find_image_path(isic_id: str, train_img_dir: str) -> str | None:
    """Resolve `train_img_dir/{isic_id}.jpg` (common ISIC / Kaggle layouts)."""
    if not train_img_dir or not os.path.isdir(train_img_dir):
        return None
    sid = str(isic_id).strip()
    base = Path(train_img_dir)
    for name in (sid, sid.upper(), sid.lower()):
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
            p = base / f"{name}{ext}"
            if p.is_file():
                return str(p)
    return None


def image_coverage_fraction(
    df_feat: pd.DataFrame,
    train_img_dir: str,
    *,
    max_check: int = 5000,
    seed: int = 42,
) -> float:
    """Approximate fraction of rows with a resolvable image file (sampled)."""
    rng = np.random.default_rng(seed)
    n = len(df_feat)
    if n == 0:
        return 0.0
    take = min(n, max_check)
    idx = rng.choice(np.arange(n), size=take, replace=False)
    ids = df_feat["isic_id"].astype(str).values[idx]
    ok = sum(1 for i in ids if find_image_path(i, train_img_dir) is not None)
    return float(ok) / float(take)


def get_torch_device() -> "torch.device":
    _require_torch_stack()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def effective_val_batch_size(batch_size: int, device: "torch.device") -> int:
    if device.type == "cpu":
        return min(batch_size, 32)
    if device.type == "mps":
        return min(batch_size, 48)
    return batch_size


def build_transforms(img_size: int, train: bool) -> A.Compose:
    _require_torch_stack()
    assert A is not None and ToTensorV2 is not None
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.05, rotate_limit=12, border_mode=4, p=0.35
                ),
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35),
                A.HueSaturationValue(
                    hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=8, p=0.25
                ),
                norm,
                ToTensorV2(),
            ]
        )
    return A.Compose([A.Resize(img_size, img_size), norm, ToTensorV2()])


class IsicImageDataset(Dataset):
    """Index rows of df_feat; load JPEGs from ``train_img_dir``."""

    def __init__(
        self,
        row_indices: np.ndarray,
        df_feat: pd.DataFrame,
        y: np.ndarray,
        train_img_dir: str,
        transform: A.Compose,
        img_size: int,
    ) -> None:
        _require_torch_stack()
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.isic_ids = df_feat["isic_id"].astype(str).values
        self.y = y
        self.train_img_dir = train_img_dir
        self.transform = transform
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int):
        from PIL import Image

        row = int(self.row_indices[i])
        sid = self.isic_ids[row]
        p = find_image_path(sid, self.train_img_dir)
        if p is None:
            rgb = np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)
        else:
            rgb = np.array(Image.open(p).convert("RGB"))
        aug = self.transform(image=rgb)
        x = aug["image"]
        t = torch.as_tensor(self.y[row], dtype=torch.float32)
        return x, t


def subsample_train_indices_majority(
    train_idx: np.ndarray,
    y: np.ndarray,
    neg_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep all positives; subsample negatives to ``neg_fraction`` of negatives in this train fold."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    y_tr = y[train_idx]
    pos = train_idx[y_tr == 1]
    neg = train_idx[y_tr == 0]
    if neg_fraction >= 1.0 or len(neg) == 0:
        return train_idx
    n_keep = max(1, int(np.ceil(len(neg) * float(neg_fraction))))
    n_keep = min(n_keep, len(neg))
    neg_keep = rng.choice(neg, size=n_keep, replace=False)
    return np.sort(np.unique(np.concatenate([pos, neg_keep])))


def optimize_late_fusion(
    tabular_oof: np.ndarray,
    img_oof: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 101,
) -> tuple[str, float, float, dict]:
    """
    Search 1D blend on OOF (stacked predictions only).

    Returns (mode, weight_on_tabular, best_pauc, details).
    """
    tabular_oof = np.asarray(tabular_oof, dtype=np.float64)
    img_oof = np.asarray(img_oof, dtype=np.float64)
    y = np.asarray(y)
    eps = 1e-7
    lt = logit(np.clip(tabular_oof, eps, 1.0 - eps))
    li = logit(np.clip(img_oof, eps, 1.0 - eps))

    best_prob_w, best_prob = _grid_pauc(y, tabular_oof, img_oof, use_logit=False, steps=steps)
    best_logit_w, best_logit = _grid_pauc(y, lt, li, use_logit=True, steps=steps)

    if best_logit > best_prob + 1e-12:
        return (
            "logit",
            float(best_logit_w),
            float(best_logit),
            {"prob_pauc": best_prob, "logit_pauc": best_logit, "prob_w": best_prob_w},
        )
    return (
        "prob",
        float(best_prob_w),
        float(best_prob),
        {"prob_pauc": best_prob, "logit_pauc": best_logit, "logit_w": best_logit_w},
    )


def _grid_pauc(y, a, b, *, use_logit: bool, steps: int) -> tuple[float, float]:
    best_w = 0.5
    best = -1.0
    for w in np.linspace(0.0, 1.0, steps):
        if use_logit:
            pred = expit(w * a + (1.0 - w) * b)
        else:
            pred = w * a + (1.0 - w) * b
        p = compute_pauc(y, pred)
        if p > best + 1e-15 or (abs(p - best) <= 1e-15 and w >= best_w):
            best = p
            best_w = w
    return best_w, best


def fused_predictions(
    tabular_oof: np.ndarray,
    img_oof: np.ndarray,
    *,
    mode: str,
    weight_on_tabular: float,
) -> np.ndarray:
    w = float(weight_on_tabular)
    tabular_oof = np.asarray(tabular_oof, dtype=np.float64)
    img_oof = np.asarray(img_oof, dtype=np.float64)
    if mode == "logit":
        eps = 1e-7
        lt = logit(np.clip(tabular_oof, eps, 1.0 - eps))
        li = logit(np.clip(img_oof, eps, 1.0 - eps))
        return expit(w * lt + (1.0 - w) * li)
    return w * tabular_oof + (1.0 - w) * img_oof


def _train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler: GradScaler | None,
    criterion,
    device: "torch.device",
    use_cuda_amp: bool,
    *,
    show_progress: bool = True,
    desc: str = "train",
) -> float:
    _require_torch_stack()
    model.train()
    tot = 0.0
    n = 0
    it = loader
    if show_progress:
        it = tqdm(
            loader,
            desc=desc,
            leave=False,
            mininterval=0.4,
            unit="batch",
        )
    for xb, yb in it:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        if use_cuda_amp and scaler is not None:
            with autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tot += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def _predict_val_probs(
    model,
    loader: DataLoader,
    device: "torch.device",
    use_cuda_amp: bool,
    *,
    show_progress: bool = True,
    desc: str = "val infer",
) -> np.ndarray:
    _require_torch_stack()
    model.eval()
    outs: list[float] = []
    it = loader
    if show_progress:
        it = tqdm(loader, desc=desc, leave=False, mininterval=0.4, unit="batch")
    for xb, _ in it:
        xb = xb.to(device, non_blocking=True)
        if use_cuda_amp:
            with autocast():
                logits = model(xb)
        else:
            logits = model(xb)
        p = torch.sigmoid(logits).squeeze(-1).float().cpu().numpy()
        outs.extend(p.tolist())
    return np.asarray(outs, dtype=np.float64)


def fit_image_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    df_feat: pd.DataFrame,
    y: np.ndarray,
    train_img_dir: str,
    model_name: str,
    img_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    neg_subsample_fraction: float,
    early_stop_patience: int,
    seed: int,
    model_dir: str | None,
    fold_id: int,
    num_workers: int,
    n_folds: int = 5,
    show_progress: bool = True,
    log_each_epoch: bool = True,
) -> np.ndarray:
    """Train timm model on ``train_idx``; return probabilities for rows ``val_idx`` (same order)."""
    _require_torch_stack()
    device = get_torch_device()
    rng = np.random.default_rng(seed)
    train_fit = subsample_train_indices_majority(train_idx, y, neg_subsample_fraction, rng)

    tr_tf = build_transforms(img_size, train=True)
    va_tf = build_transforms(img_size, train=False)
    ds_tr = IsicImageDataset(train_fit, df_feat, y, train_img_dir, tr_tf, img_size)
    ds_va = IsicImageDataset(val_idx, df_feat, y, train_img_dir, va_tf, img_size)

    pin = device.type == "cuda"
    bs_val = effective_val_batch_size(batch_size, device)
    _dl_common = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        _dl_common["persistent_workers"] = True
        _dl_common["prefetch_factor"] = 2
    loader_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **_dl_common,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=bs_val,
        shuffle=False,
        **_dl_common,
    )

    n_pos_tr = int((y[train_fit] == 1).sum())
    n_neg_tr = int((y[train_fit] == 0).sum())
    n_pos_va = int((y[val_idx] == 1).sum())
    _log(
        f"Phase2 CNN fold {fold_id + 1}/{n_folds} | {model_name} | device={device} | "
        f"train_rows={len(train_fit):,} (malignant={n_pos_tr}, benign={n_neg_tr}) | "
        f"val_rows={len(val_idx):,} (malignant={n_pos_va})"
    )

    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    model = model.to(device)

    n_pos = max(1, n_pos_tr)
    n_neg = max(1, n_neg_tr)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    use_cuda_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_cuda_amp)

    best_pauc = -1.0
    best_state = None
    bad = 0
    ep_done = 0

    epoch_range = range(epochs)
    if show_progress:
        epoch_range = tqdm(
            epoch_range,
            desc=f"Fold {fold_id + 1}/{n_folds} epochs",
            leave=True,
            unit="epoch",
        )

    for ep in epoch_range:
        train_loss = _train_one_epoch(
            model,
            loader_tr,
            optimizer,
            scaler,
            criterion,
            device,
            use_cuda_amp,
            show_progress=show_progress,
            desc=f"train ep{ep + 1}/{epochs}",
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        val_p = _predict_val_probs(
            model,
            loader_va,
            device,
            use_cuda_amp,
            show_progress=show_progress,
            desc=f"val ep{ep + 1}",
        )
        vp = compute_pauc(y[val_idx], val_p)
        ep_done = ep + 1
        if vp > best_pauc + 1e-7:
            best_pauc = vp
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            improved = True
        else:
            improved = False
            bad += 1
        if show_progress and hasattr(epoch_range, "set_postfix"):
            epoch_range.set_postfix(
                val_pauc=f"{vp:.4f}",
                best_pauc=f"{best_pauc:.4f}",
                loss=f"{train_loss:.4f}",
                lr=f"{lr_now:.2e}",
                bad_epochs=bad,
            )
        if log_each_epoch:
            _log(
                f"  epoch {ep + 1}/{epochs} | train_loss={train_loss:.5f} | val_pAUC={vp:.5f} | "
                f"best_val_pAUC={best_pauc:.5f} | lr={lr_now:.2e} | "
                f"{'new best' if improved else f'no improve ({bad}/{early_stop_patience})'}"
            )
        if bad >= early_stop_patience:
            if log_each_epoch:
                _log(f"  early stopping at epoch {ep_done} (patience={early_stop_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _log(f"Fold {fold_id + 1} final val pass (best checkpoint, {ep_done} epochs trained)")
    preds = _predict_val_probs(
        model,
        loader_va,
        device,
        use_cuda_amp,
        show_progress=show_progress,
        desc="val final",
    )

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        safe = model_name.replace("/", "_").replace(".", "_")
        path = os.path.join(model_dir, f"phase2_{safe}_fold{fold_id + 1}.pt")
        torch.save({"state_dict": best_state or model.state_dict(), "fold": fold_id}, path)

    return preds


def run_phase2_image_oof_and_fusion(
    *,
    root: str,
    cfg,
    pred_data: dict,
) -> dict:
    """
    Build ``img_oof`` with the same folds as Phase 1, optimize late fusion vs ``tabular_oof``.

    Expects ``pred_data`` from Notebook 4 (``fold_iterator``, ``tabular_oof``, ``y``, ``df_feat``).
    Returns a **new** dict suitable for ``model_predictions.pkl`` (does not mutate input).
    """
    _require_torch_stack()
    from isic_challenge.config import TrainingConfig

    if not isinstance(cfg, TrainingConfig):
        raise TypeError("cfg must be a TrainingConfig instance")

    paths = cfg.paths(root)
    train_img_dir = paths["train_img_dir"]
    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(
            f"Image folder not found: {train_img_dir!r}. "
            "Place Kaggle training images under this path (see README)."
        )

    fold_iterator = pred_data.get("fold_iterator")
    if not fold_iterator:
        raise KeyError("pred_data must contain fold_iterator from Notebook 4")

    df_feat = pred_data["df_feat"]
    y = np.asarray(pred_data["y"])
    tabular_oof = np.asarray(pred_data["tabular_oof"], dtype=np.float64)

    cov = image_coverage_fraction(df_feat, train_img_dir)
    if cov < 0.5:
        import warnings

        warnings.warn(
            f"Low image file coverage (~{cov:.0%} in sample). Check train_img_dir and file names.",
            stacklevel=2,
        )

    img_oof = np.zeros(len(y), dtype=np.float64)
    fold_meta: list[dict] = []

    fold_list = list(fold_iterator)
    n_folds = len(fold_list)
    _log(
        f"Phase2 image OOF | {n_folds} folds | backbone={cfg.img_model} | "
        f"show_progress={cfg.img_show_progress}"
    )

    fold_iter = fold_list
    if cfg.img_show_progress:
        fold_iter = tqdm(fold_list, desc="Phase2 CV folds", unit="fold", leave=True)

    for fold_id, (train_idx, val_idx) in enumerate(fold_iter):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        seed = cfg.seed + 1000 * fold_id
        p_va = fit_image_fold(
            train_idx,
            val_idx,
            df_feat=df_feat,
            y=y,
            train_img_dir=train_img_dir,
            model_name=cfg.img_model,
            img_size=cfg.img_size,
            epochs=cfg.img_epochs,
            batch_size=cfg.img_batch_size,
            lr=cfg.img_lr,
            weight_decay=cfg.img_weight_decay,
            neg_subsample_fraction=cfg.img_neg_subsample_fraction,
            early_stop_patience=cfg.img_early_stop_patience,
            seed=seed,
            model_dir=paths["model_dir"],
            fold_id=fold_id,
            num_workers=cfg.img_num_workers,
            n_folds=n_folds,
            show_progress=cfg.img_show_progress,
            log_each_epoch=cfg.img_log_each_epoch,
        )
        img_oof[val_idx] = p_va
        try:
            va_auc = float(roc_auc_score(y[val_idx], p_va))
        except ValueError:
            va_auc = float("nan")
        fold_meta.append(
            {
                "fold": fold_id + 1,
                "val_pauc": float(compute_pauc(y[val_idx], p_va)),
                "val_auc": va_auc,
            }
        )
        _log(
            f"Fold {fold_id + 1}/{n_folds} OOF slice done | val_pAUC={fold_meta[-1]['val_pauc']:.5f} | "
            f"val_ROC_AUC={va_auc}"
        )

    _log(f"Searching late fusion ({cfg.img_fusion_grid_steps} grid steps, pAUC objective)…")
    mode, w_tab, best_pauc, fus_detail = optimize_late_fusion(
        tabular_oof, img_oof, y, steps=cfg.img_fusion_grid_steps
    )
    final_oof = fused_predictions(tabular_oof, img_oof, mode=mode, weight_on_tabular=w_tab)
    _log(
        f"Fusion picked: mode={mode!r} | w_tabular={w_tab:.4f} | OOF pAUC={best_pauc:.5f} | detail={fus_detail}"
    )

    out = dict(pred_data)
    out["img_oof"] = img_oof
    out["has_img"] = True
    out["phase"] = "multimodal_phase2"
    out["final_oof"] = final_oof
    out["fusion_mode"] = mode
    out["fusion_weight_tabular"] = w_tab
    out["fusion_detail"] = fus_detail
    out["img_model"] = cfg.img_model
    out["phase2_fold_metrics"] = fold_meta
    return out

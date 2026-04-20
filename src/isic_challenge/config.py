"""Central training hyperparameters and paths.

Phase 1 — LightGBM + XGBoost tabular ensemble (Notebook 4)
Phase 2 — EfficientNet-B4-NS CNN with focal loss, MixUp, warmup LR (Notebook 6)
Phase 3 — Multimodal late fusion (Notebook 7, planned)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    Central hyperparameter store for all phases.

    **Phase 1:** metadata-only LightGBM + XGBoost + weighted tabular blend.
    **Phase 2:** EfficientNet-B4-NS CNN on lesion crops + Focal loss + MixUp +
                 Warmup-CosineAnnealing LR.  Late fusion of CNN OOF + tabular OOF
                 deferred to Phase 3.
    """

    # ── General ────────────────────────────────────────────────────────────────
    seed: int = 42
    n_folds: int = 5
    majority_subsample_fraction: float = 1.0

    # ── Phase 1 — Gradient Boosting ────────────────────────────────────────────
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 63
    lgb_max_depth: int = 8
    boost_early_stopping_rounds: int = 100
    boost_n_estimators: int = 2000
    boost_log_interval: int = 100

    # ── Phase 2 — CNN Image Model ──────────────────────────────────────────────
    # Architecture
    # tf_efficientnet_b4_ns = NoisyStudent pretrained EfficientNet-B4.
    # Chosen over B0 (underfits lesion detail at 256px) and B7 (too many params
    # for ~400 positives).  NoisyStudent pretraining bridges ImageNet→medical gap
    # better than vanilla supervised pretraining.
    img_model: str = "tf_efficientnet_b4_ns"
    img_size: int = 256            # 256 > 224 for TBP lesion crops; divisible by 32

    # Training
    img_epochs: int = 12
    img_batch_size: int = 32       # B4 needs more VRAM; halve vs B0
    img_lr: float = 2e-4           # slightly higher with warmup
    img_weight_decay: float = 0.01

    # Learning-rate schedule: LinearWarmup(warmup_epochs) → CosineAnnealing
    img_warmup_epochs: int = 1

    # Regularization
    img_dropout: float = 0.4       # classification-head dropout
    img_neg_subsample_fraction: float = 0.12  # keep 12 % negatives per fold
    img_early_stop_patience: int = 4

    # Loss — Focal Loss  FL(p_t) = −α_t (1−p_t)^γ log(p_t)
    img_focal_gamma: float = 2.0   # γ: down-weights easy negatives
    # img_focal_alpha: auto-computed from imbalance ratio inside fit_image_fold

    # Augmentation
    img_mixup_alpha: float = 0.4   # MixUp Beta-distribution α; 0 = no mixup

    # Data / IO
    img_num_workers: int = 0
    img_pin_memory: bool = True

    # Late-fusion (grid search on OOF pAUC) — kept for backward compat
    img_fusion_grid_steps: int = 101

    # Logging
    img_show_progress: bool = True
    img_log_each_epoch: bool = True

    # ── Paths ──────────────────────────────────────────────────────────────────
    def paths(self, root: str) -> dict:
        import os

        ds = os.path.join(root, "ISIC 2024 Skin Cancer Challenge Dataset")
        return {
            "root": root,
            "prep_pickle": os.path.join(root, "outputs", "preprocessed_data.pkl"),
            "fig_dir": os.path.join(root, "outputs", "figures"),
            "model_dir": os.path.join(root, "outputs", "models"),
            "data_dir": ds,
            # Real ISIC 2024 ships images as an HDF5 archive (preferred)
            "train_hdf5": os.path.join(ds, "train-image.hdf5"),
            # Fallback: extracted JPEG directory
            "train_img_dir": os.path.join(ds, "train-image", "image"),
        }

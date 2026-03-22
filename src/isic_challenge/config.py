"""Central training hyperparameters and paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Notebook 4 hyperparameters.

    **Phase 1:** metadata-only LightGBM + XGBoost + tabular blend.
    **Phase 2 (planned):** image model + fusion; image fields reserved for that track.
    """

    seed: int = 42
    n_folds: int = 5
    majority_subsample_fraction: float = 1.0

    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 63
    lgb_max_depth: int = 8
    boost_early_stopping_rounds: int = 100
    boost_n_estimators: int = 2000
    boost_log_interval: int = 100

    # Phase 2 — image track
    img_size: int = 224
    img_epochs: int = 10
    img_batch_size: int = 64
    img_lr: float = 1e-4
    img_num_workers: int = 0

    def paths(self, root: str) -> dict:
        import os

        ds = os.path.join(root, "ISIC 2024 Skin Cancer Challenge Dataset")
        return {
            "root": root,
            "prep_pickle": os.path.join(root, "outputs", "preprocessed_data.pkl"),
            "fig_dir": os.path.join(root, "outputs", "figures"),
            "model_dir": os.path.join(root, "outputs", "models"),
            "data_dir": ds,
            "train_hdf5": os.path.join(ds, "train-image.hdf5"),
            "train_img_dir": os.path.join(ds, "train-image", "image"),
        }

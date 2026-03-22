"""Cross-validation helpers: fold-safe imputation, site residuals, optional majority subsampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def site_residual_matrix(
    df_feat: pd.DataFrame,
    train_idx: np.ndarray,
    raw_cols: list,
    site_col: str = "anatom_site_general",
) -> np.ndarray:
    dtr = df_feat.iloc[train_idx]
    n = len(df_feat)
    k = len(raw_cols)
    mat = np.zeros((n, k), dtype=np.float64)
    sites = df_feat[site_col].astype(str)
    for j, col in enumerate(raw_cols):
        site_mean = dtr.groupby(site_col)[col].mean()
        g_mean = float(np.nanmean(pd.to_numeric(dtr[col], errors="coerce")))
        mapped = sites.map(site_mean)
        mapped = pd.to_numeric(mapped, errors="coerce").fillna(g_mean)
        vals = pd.to_numeric(df_feat[col], errors="coerce").fillna(g_mean)
        mat[:, j] = (vals - mapped).values
    return mat


def fold_impute_median(X_df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    imp.fit(X_df.iloc[train_idx])
    out = pd.DataFrame(imp.transform(X_df), columns=X_df.columns, index=X_df.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def subsample_majority_train_indices(
    train_idx: np.ndarray,
    y: np.ndarray,
    majority_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if majority_fraction >= 1.0:
        return train_idx
    y_t = np.asarray(y)[train_idx]
    pos_mask = y_t == 1
    neg_mask = ~pos_mask
    pos_idx = train_idx[pos_mask]
    neg_idx = train_idx[neg_mask]
    if len(neg_idx) == 0:
        return train_idx
    n_keep = max(1, int(np.ceil(len(neg_idx) * majority_fraction)))
    n_keep = min(n_keep, len(neg_idx))
    chosen_neg = rng.choice(neg_idx, size=n_keep, replace=False)
    return np.sort(np.unique(np.concatenate([pos_idx, chosen_neg])))

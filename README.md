# Skin cancer detection — ISIC 2024 Challenge

**Phase 1 (this repo):** train strong **metadata-only** models (LightGBM + XGBoost + weighted tabular ensemble) with patient-grouped CV, fold-safe preprocessing, and competition **pAUC (≥88% TPR)**.

**Phase 2:** **lesion image** training (pretrained EfficientNet via timm) with the **same CV folds** as Phase 1, then **late fusion** (probability or logit blend) with the tabular stack for further gains.

## Project phases

| Phase | Scope | Notebooks / code |
|--------|--------|------------------|
| **1** | Tabular only | `03` → `04` (LGB, XGB, blend) → `05` |
| **2** | Images + fusion | `06_Image_Training_and_Fusion.ipynb` (`isic_challenge.image_pipeline`); `pip install -e ".[image]"` or use a `requirements.txt` env with PyTorch/timm |

## Dataset snapshot

| Property | Value |
|----------|--------|
| Rows (train metadata) | 401,059 |
| Malignant (target=1) | 393 |
| Approx. imbalance | ~1020:1 benign:malignant |

## Repository layout

```
Skin-Cancer-Detection-ISIC-2024-/
├── src/isic_challenge/          # Importable helpers (config, metrics, CV utils) — required
├── 01_Literature_Review.ipynb
├── 02_EDA.ipynb
├── 03_Feature_Engineering.ipynb
├── 04_Model_Training.ipynb      # Phase 1: tabular CV + ensemble only
├── 06_Image_Training_and_Fusion.ipynb  # Phase 2: CNN OOF + late fusion → updates model_predictions.pkl
├── 05_Evaluation_and_Analysis.ipynb
├── pyproject.toml               # pip install -e .
├── requirements.txt
├── outputs/
│   ├── figures/                 # Notebook plots
│   ├── models/                  # Saved checkpoints (Phase 2); notebooks use this path
│   ├── preprocessed_data.pkl    # from Notebook 3
│   └── model_predictions.pkl    # from Notebook 4
├── venv/                        # local env (gitignored)
└── ISIC 2024 Skin Cancer Challenge Dataset/   # not in git; see .gitignore
```

**Note:** If you have empty top-level folders such as `models/` or `training/`, they are not used by the notebooks (`TrainingConfig` writes under `outputs/models/`). You can delete those folders or repurpose them; keeping a single `outputs/` tree avoids confusion.

## Setup

**Phase 1 (CPU-friendly):**

```bash
pip install -e .
# or: pip install -r requirements.txt   # adds PyTorch/timm for Phase 2 experiments
```

Notebook 4 prepends `src` to `sys.path`, so you can run without installing if the repo root is the working directory.

**macOS + LightGBM:** if you see `libomp` errors, run `brew install libomp` once.

## How to run

1. Place the ISIC 2024 challenge files under `ISIC 2024 Skin Cancer Challenge Dataset/` (see `.gitignore` for expected names).
2. Run in order: `02_EDA` → `03_Feature_Engineering` → `04_Model_Training` → (`06_Image_Training_and_Fusion` for Phase 2) → `05_Evaluation_and_Analysis`.
3. `01` is narrative (literature); run cells optionally.

**Google Colab:** upload the project folder to Drive, match `ROOT` in the notebooks to your path, run the same order.

## Phase 1 methodology (tabular)

- **CV:** `StratifiedGroupKFold` on `patient_id` (fallback: `StratifiedKFold`).
- **Imbalance:** full training rows per fold + `scale_pos_weight`; optional per-fold majority subsample via `TrainingConfig.majority_subsample_fraction`.
- **Leakage control:** median imputation and site-residual features fit **only on each fold’s training indices** (aligned with Notebook 3).
- **Ensemble:** grid search on OOF predictions to blend LightGBM vs XGBoost. **`final_oof`** in `outputs/model_predictions.pkl` is the tabular blend after Phase 1, and the **tabular+image fusion** after Phase 2 (Notebook 6).

## Primary metric

**Partial AUC above 88% TPR** — ISIC 2024 primary leaderboard definition ([Challenge-2024-Metrics](https://github.com/ISIC-Research/Challenge-2024-Metrics)).

## License

Uses the ISIC 2024 dataset under the licenses stated by the challenge organizers and data contributors.

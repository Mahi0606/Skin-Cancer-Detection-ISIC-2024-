"""Phase-2 EfficientNet K-fold ensemble inference for uploaded images."""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from isic_challenge.image_pipeline import EfficientNetClassifier, get_val_transforms  # noqa: E402


def default_model_dir() -> Path:
    override = os.environ.get("SKINCD_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / "outputs" / "models").resolve()


def _sort_fold_checkpoints(paths: List[Path]) -> List[Path]:
    def key(p: Path) -> Tuple[int, str]:
        m = re.search(r"fold(\d+)\.pt$", p.name, re.I)
        return (int(m.group(1)) if m else 0, p.name)

    return sorted(paths, key=key)


def discover_checkpoints(model_dir: Path) -> List[Path]:
    if not model_dir.is_dir():
        return []
    paths = list(model_dir.glob("phase2_*_fold*.pt"))
    return _sort_fold_checkpoints(paths)


def bytes_to_rgb_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


class EfficientNetFoldEnsemble:
    """Lazy-loaded ensemble of per-fold Phase-2 checkpoints (mean logit → sigmoid)."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or default_model_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: List[torch.nn.Module] | None = None
        self._transform = None
        self._checkpoint_paths: List[Path] = []

    def checkpoint_paths(self) -> List[Path]:
        if not self._checkpoint_paths:
            self._checkpoint_paths = discover_checkpoints(self.model_dir)
        return self._checkpoint_paths

    def _load(self) -> None:
        if self._models is not None:
            return
        paths = self.checkpoint_paths()
        if not paths:
            raise FileNotFoundError(
                f"No Phase-2 checkpoints matching phase2_*_fold*.pt under {self.model_dir}"
            )

        models: List[torch.nn.Module] = []
        img_size: int | None = None
        for p in paths:
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            cfg_model = ckpt["cfg_model"]
            state = ckpt["model_state_dict"]
            fold_img_size = int(ckpt.get("cfg_img_size", 256))
            if img_size is None:
                img_size = fold_img_size
            elif fold_img_size != img_size:
                raise ValueError(
                    f"Inconsistent cfg_img_size across folds ({img_size} vs {fold_img_size})"
                )

            model = EfficientNetClassifier(
                model_name=cfg_model,
                pretrained=False,
                dropout=0.4,
            ).to(self.device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)

        self._models = models
        self._transform = get_val_transforms(img_size or 256)

    def predict_malignant_prob(self, image_bytes: bytes) -> Tuple[float, dict]:
        """
        Returns P(malignant) in [0, 1] and metadata for the API response.
        """
        self._load()
        assert self._models is not None and self._transform is not None

        try:
            arr = bytes_to_rgb_array(image_bytes)
        except Exception as e:
            raise ValueError("Could not decode image bytes") from e

        tensor = self._transform(image=arr)["image"]
        batch = tensor.unsqueeze(0).to(self.device, non_blocking=True)

        logits: List[float] = []
        with torch.inference_mode():
            use_amp = self.device.type == "cuda"
            for model in self._models:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logit = model(batch).squeeze(0).float().cpu().item()
                logits.append(logit)

        mean_logit = float(np.mean(logits))
        p_mal = float(torch.sigmoid(torch.tensor(mean_logit)).item())

        meta = {
            "branch": "image",
            "ensemble_folds": len(logits),
            "mean_logit": mean_logit,
        }
        return p_mal, meta

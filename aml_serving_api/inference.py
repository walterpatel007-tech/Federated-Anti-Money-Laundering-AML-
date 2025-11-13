"""Model loading and inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from aml_federated_training.lightning_module import AMLClassifier


DEFAULT_MODEL_PATH = Path("artifacts/global_model.ckpt")
MODEL_CONFIG_PATH = Path("config/model_config.yaml")


def _load_model_hparams() -> dict:
    if MODEL_CONFIG_PATH.exists():
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    raise FileNotFoundError(
        "Model hyperparameters could not be inferred. Provide config/model_config.yaml."
    )


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> AMLClassifier:
    """Load a trained :class:`AMLClassifier` checkpoint."""

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Run federated training first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters") or _load_model_hparams()
    model = AMLClassifier(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def score_transaction(model: AMLClassifier, features: Iterable[float]) -> float:
    """Return the suspicious score for a transaction given numeric features."""

    with torch.no_grad():
        array = np.array(list(features), dtype=np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0)
        logits = model(tensor).squeeze(0)
        score = torch.sigmoid(logits).item()
        return float(score)

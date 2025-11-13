"""Model loading and inference utilities."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import yaml

from aml_federated_training.lightning_module import AMLClassifier


DEFAULT_MODEL_PATH = Path("artifacts/global_model.ckpt")
MODEL_CONFIG_PATH = Path("config/model_config.yaml")

_MODEL_SIGNATURE = inspect.signature(AMLClassifier.__init__)
_MODEL_INIT_FIELDS = {
    name
    for name, parameter in _MODEL_SIGNATURE.parameters.items()
    if name != "self"
}
_REQUIRED_MODEL_FIELDS = {
    name
    for name, parameter in _MODEL_SIGNATURE.parameters.items()
    if name != "self" and parameter.default is inspect._empty
}


def _extract_model_kwargs(config: Mapping[str, object]) -> dict[str, object]:
    """Filter a configuration mapping to the kwargs accepted by :class:`AMLClassifier`."""

    config_dict = dict(config)
    missing = _REQUIRED_MODEL_FIELDS - config_dict.keys()
    if missing:
        formatted = ", ".join(sorted(missing))
        raise KeyError(f"Missing required model hyperparameters: {formatted}")
    return {key: config_dict[key] for key in _MODEL_INIT_FIELDS if key in config_dict}


def _load_model_hparams() -> dict:
    if MODEL_CONFIG_PATH.exists():
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            raw_config = yaml.safe_load(fh) or {}
            return _extract_model_kwargs(raw_config)
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
    hparams = checkpoint.get("hyper_parameters")
    if hparams:
        model_kwargs = _extract_model_kwargs(hparams)
    else:
        model_kwargs = _load_model_hparams()
    model = AMLClassifier(**model_kwargs)
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

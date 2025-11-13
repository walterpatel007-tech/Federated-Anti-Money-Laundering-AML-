"""Flower client definition for the AML federated setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import flwr as fl
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import TransactionDataset
from .evaluation import compute_metrics
from .lightning_module import AMLClassifier


@dataclass
class ClientConfig:
    """Configuration for a Flower client representing a single bank."""

    name: str
    data_path: Path
    feature_columns: list[str]
    label_column: str
    batch_size: int
    local_epochs: int


def _state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().cpu().numpy() for value in state_dict.values()]


def _numpy_to_state_dict(parameters: Iterable, model: AMLClassifier) -> None:
    state_dict = model.state_dict()
    for value, (key, tensor) in zip(parameters, state_dict.items()):
        state_dict[key] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(state_dict)


def _create_dataloaders(client_cfg: ClientConfig) -> tuple[DataLoader, DataLoader]:
    frame = pd.read_csv(client_cfg.data_path)
    dataset = TransactionDataset(frame, client_cfg.feature_columns, client_cfg.label_column)
    val_size = max(int(0.1 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset)
        val_size = len(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=client_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=client_cfg.batch_size)
    return train_loader, val_loader


def build_client(model: AMLClassifier, client_cfg: ClientConfig) -> fl.client.NumPyClient:
    """Instantiate a Flower client for a specific bank."""

    train_loader, val_loader = _create_dataloaders(client_cfg)

    class LightningClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # pragma: no cover - Flower callback
            return _state_dict_to_numpy(model.state_dict())

        def fit(self, parameters, fit_config):  # pragma: no cover - Flower callback
            _numpy_to_state_dict(parameters, model)
            fit_config = fit_config or {}
            local_epochs = fit_config.get("local_epochs") or fit_config.get("epochs") or client_cfg.local_epochs
            trainer = pl.Trainer(
                max_epochs=local_epochs,
                accelerator="cpu",
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            return self.get_parameters({}), len(train_loader.dataset), {}

        def evaluate(self, parameters, eval_config):  # pragma: no cover - Flower callback
            _numpy_to_state_dict(parameters, model)
            model.eval()
            criterion = torch.nn.BCEWithLogitsLoss()
            losses = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for features, labels in val_loader:
                    logits = model(features)
                    loss = criterion(logits, labels.float())
                    losses.append(loss.item())
                    all_labels.append(labels)
                    all_preds.append(torch.sigmoid(logits))
            if losses:
                avg_loss = sum(losses) / len(losses)
                labels_tensor = torch.cat(all_labels)
                preds_tensor = torch.cat(all_preds)
                metrics = compute_metrics(labels_tensor, preds_tensor)
            else:
                avg_loss = 0.0
                metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}
            return float(avg_loss), len(val_loader.dataset), metrics

    return LightningClient()

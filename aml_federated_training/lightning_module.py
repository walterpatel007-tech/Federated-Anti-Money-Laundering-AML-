"""PyTorch Lightning module for AML classification."""

from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class AMLClassifier(pl.LightningModule):
    """Simple feedforward network for AML detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        learning_rate: float,
        weight_decay: float = 0.0,
        class_weight: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weight = class_weight

        pos_weight = torch.tensor(class_weight) if class_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_accuracy = BinaryAccuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.network(x).squeeze(-1)

    def _step(self, batch: Any, stage: str) -> Dict[str, torch.Tensor]:
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels.float())
        preds = torch.sigmoid(logits)
        metrics = {
            f"{stage}_loss": loss,
        }
        if stage == "train":
            self.train_precision(preds, labels.int())
            self.train_recall(preds, labels.int())
            metrics["train_precision"] = self.train_precision.compute()
            metrics["train_recall"] = self.train_recall.compute()
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.val_precision(preds, labels.int())
            self.val_recall(preds, labels.int())
            self.val_f1(preds, labels.int())
            self.val_accuracy(preds, labels.int())
            metrics.update(
                {
                    "val_precision": self.val_precision.compute(),
                    "val_recall": self.val_recall.compute(),
                    "val_f1": self.val_f1.compute(),
                    "val_accuracy": self.val_accuracy.compute(),
                }
            )
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        return metrics

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        metrics = self._step(batch, "train")
        return metrics["train_loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # pragma: no cover - logging only
        self._step(batch, "val")

    def configure_optimizers(self):  # pragma: no cover - Lightning API
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

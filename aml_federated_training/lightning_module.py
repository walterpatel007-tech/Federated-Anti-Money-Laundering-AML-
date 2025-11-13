"""PyTorch Lightning module for AML classification."""

from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
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
        self.save_hyperparameters()
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

        if class_weight is not None:
            pos_weight = torch.tensor(float(class_weight), dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_accuracy = BinaryAccuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.network(x).squeeze(-1)

    def _step(self, batch: Any, stage: str) -> Tensor:
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels.float())
        preds = torch.sigmoid(logits)
        label_int = labels.int()

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=label_int.size(0),
        )

        if stage == "train":
            self.log(
                "train_precision",
                self.train_precision(preds, label_int),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=label_int.size(0),
            )
            self.log(
                "train_recall",
                self.train_recall(preds, label_int),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=label_int.size(0),
            )
        else:
            self.log(
                "val_precision",
                self.val_precision(preds, label_int),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=label_int.size(0),
            )
            self.log(
                "val_recall",
                self.val_recall(preds, label_int),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=label_int.size(0),
            )
            self.log(
                "val_f1",
                self.val_f1(preds, label_int),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=label_int.size(0),
            )
            self.log(
                "val_accuracy",
                self.val_accuracy(preds, label_int),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=label_int.size(0),
            )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # pragma: no cover - logging only
        self._step(batch, "val")

    def on_train_epoch_end(self) -> None:  # pragma: no cover - Lightning hook
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self) -> None:  # pragma: no cover - Lightning hook
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_accuracy.reset()

    def configure_optimizers(self):  # pragma: no cover - Lightning API
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

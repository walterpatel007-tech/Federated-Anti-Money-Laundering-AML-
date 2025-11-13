"""Dataset helpers for federated AML training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class TransactionDataset(Dataset):
    """PyTorch dataset representing AML transactions."""

    def __init__(self, frame: pd.DataFrame, feature_columns: Iterable[str], label_column: str):
        self.frame = frame.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.label_column = label_column

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.frame)

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        row = self.frame.iloc[idx]
        features = np.asarray(row[self.feature_columns], dtype=np.float32)
        label = float(row[self.label_column])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)


@dataclass
class DataModuleConfig:
    """Configuration for the :class:`TransactionDataModule`."""

    path: Path
    feature_columns: list[str]
    label_column: str
    batch_size: int = 64
    num_workers: int = 0
    val_split: float = 0.1


class TransactionDataModule(pl.LightningDataModule):
    """Thin data module around :class:`TransactionDataset`."""

    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.config = config
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        frame = pd.read_csv(self.config.path)
        dataset = TransactionDataset(frame, self.config.feature_columns, self.config.label_column)
        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size
        if val_size == 0 or train_size == 0:
            self._train_dataset = dataset
            self._val_dataset = dataset
        else:
            self._train_dataset, self._val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

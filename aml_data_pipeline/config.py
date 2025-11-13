"""Configuration helpers for the AML data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import yaml


@dataclass
class BankFilter:
    """Filter definition for a specific bank partition."""

    bank_id: Optional[str] = None
    country: Optional[str] = None

    def matches(self, record: Mapping[str, object]) -> bool:
        """Return ``True`` when the record satisfies the filter criteria."""

        if self.bank_id is not None and record.get("bank_id") != self.bank_id:
            return False
        if self.country is not None and record.get("country") != self.country:
            return False
        return True


@dataclass
class DataConfig:
    """Top-level configuration for data ingestion and partitioning."""

    raw_data_dir: Path
    processed_data_dir: Path
    bank_splits_dir: Path
    bank_partitions: Dict[str, BankFilter]
    index_column: str
    timestamp_column: str
    label_column: str
    feature_columns: list[str]

    @classmethod
    def from_dict(cls, cfg: Mapping[str, object]) -> "DataConfig":
        bank_filters = {
            name: BankFilter(**(definition.get("filter", {})))
            for name, definition in cfg.get("bank_partitions", {}).items()
        }
        return cls(
            raw_data_dir=Path(cfg["raw_data_dir"]),
            processed_data_dir=Path(cfg["processed_data_dir"]),
            bank_splits_dir=Path(cfg["bank_splits_dir"]),
            bank_partitions=bank_filters,
            index_column=str(cfg.get("index_column", "transaction_id")),
            timestamp_column=str(cfg.get("timestamp_column", "timestamp")),
            label_column=str(cfg.get("label_column", "is_suspicious")),
            feature_columns=list(cfg.get("feature_columns", [])),
        )


def load_data_config(path: str | Path) -> DataConfig:
    """Load a :class:`DataConfig` from ``path``."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg_dict = yaml.safe_load(fh)
    return DataConfig.from_dict(cfg_dict)

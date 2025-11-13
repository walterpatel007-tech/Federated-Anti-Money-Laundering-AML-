"""Utility functions used across the data pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence


LOGGER = logging.getLogger(__name__)


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` if it does not exist and return it as :class:`Path`."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_csv_files(directory: str | Path) -> Sequence[Path]:
    """Return all CSV file paths contained within ``directory``."""

    directory_path = Path(directory)
    if not directory_path.exists():
        LOGGER.warning("Directory %s does not exist", directory_path)
        return []
    return sorted(p for p in directory_path.iterdir() if p.suffix.lower() == ".csv")


def log_partition_summary(partitions: Iterable[tuple[str, int]]) -> None:
    """Log a summary of the number of records written per partition."""

    summary = ", ".join(f"{name}: {count}" for name, count in partitions)
    LOGGER.info("Created bank partitions -> %s", summary)

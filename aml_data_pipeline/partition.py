"""Partitioning utilities for splitting transactions by bank."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import BankFilter
from .utils import ensure_directory, log_partition_summary


def _filter_dataframe(df: pd.DataFrame, bank_filter: BankFilter) -> pd.DataFrame:
    """Apply ``bank_filter`` to ``df`` and return the filtered frame."""

    mask = pd.Series(True, index=df.index)
    if bank_filter.bank_id is not None:
        mask &= df["bank_id"] == bank_filter.bank_id
    if bank_filter.country is not None and "country" in df.columns:
        mask &= df["country"] == bank_filter.country
    return df.loc[mask].copy()


def partition_by_bank(
    df: pd.DataFrame,
    output_dir: str | Path,
    bank_column: str,
    bank_filters: Iterable[tuple[str, BankFilter]] | None = None,
) -> dict[str, Path]:
    """Partition ``df`` into separate CSVs per bank."""

    output_directory = ensure_directory(output_dir)
    written_paths: dict[str, Path] = {}
    partition_counts: list[tuple[str, int]] = []

    if bank_filters is None:
        bank_filters = [(bank, BankFilter(bank_id=bank)) for bank in sorted(df[bank_column].unique())]

    for name, bank_filter in bank_filters:
        partition_df = _filter_dataframe(df, bank_filter)
        if partition_df.empty:
            continue
        partition_path = output_directory / f"{name}.csv"
        partition_df.to_csv(partition_path, index=False)
        written_paths[name] = partition_path
        partition_counts.append((name, len(partition_df)))

    log_partition_summary(partition_counts)
    return written_paths

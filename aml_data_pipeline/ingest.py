"""Functions responsible for ingesting raw AML transaction data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .schemas import TransactionRecord
from .utils import ensure_directory, list_csv_files


def _load_single_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with timestamp parsing."""

    return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)


def load_raw_transactions(directory: str | Path) -> pd.DataFrame:
    """Load and validate all CSV files in ``directory``."""

    csv_paths: List[Path] = list_csv_files(directory)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {directory!r}")

    frames = [_load_single_csv(path) for path in csv_paths]
    combined = pd.concat(frames, ignore_index=True)

    records: Iterable[TransactionRecord] = (
        TransactionRecord(**row)
        for row in combined.to_dict(orient="records")
    )
    validated_rows = [record.dict() for record in records]
    validated_df = pd.DataFrame(validated_rows)
    validated_df.sort_values("timestamp", inplace=True)
    validated_df.reset_index(drop=True, inplace=True)
    return validated_df


def write_processed_data(df: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    """Persist the processed dataframe to ``output_dir``."""

    ensure_directory(output_dir)
    output_path = Path(output_dir) / filename
    df.to_csv(output_path, index=False)
    return output_path

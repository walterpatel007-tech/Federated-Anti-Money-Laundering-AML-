"""Feature engineering utilities for AML transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with normalized numeric fields."""

    normalized = df.copy()
    if "origin_balance" in normalized.columns:
        normalized["origin_balance"] = normalized["origin_balance"].fillna(method="ffill")
    if "destination_balance" in normalized.columns:
        normalized["destination_balance"] = normalized["destination_balance"].fillna(
            method="ffill"
        )
    if "amount" in normalized.columns:
        normalized["amount"] = normalized["amount"].clip(lower=0.0)
        normalized["amount_log"] = np.log(normalized["amount"] + 1)
    if "timestamp" in normalized.columns:
        normalized["hour_of_day"] = normalized["timestamp"].dt.hour
    return normalized


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer simple features used by the baseline model."""

    engineered = normalize_transactions(df)
    if "transaction_type" in engineered.columns:
        engineered["transaction_type_encoded"] = (
            engineered["transaction_type"].astype("category").cat.codes
        )
    if "customer_risk_score" not in engineered.columns:
        engineered["customer_risk_score"] = 0.5
    for balance_col in ["origin_balance", "destination_balance"]:
        if balance_col not in engineered.columns:
            engineered[balance_col] = 0.0
    return engineered

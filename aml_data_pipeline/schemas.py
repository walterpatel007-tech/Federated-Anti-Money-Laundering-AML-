"""Data validation schemas for AML transactions."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


class TransactionRecord(BaseModel):
    """Representation of a single AML transaction record."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    origin_customer_id: str = Field(..., description="Originating customer ID")
    destination_customer_id: str = Field(..., description="Receiving customer ID")
    amount: float = Field(..., ge=0.0, description="Transaction amount")
    currency: str = Field(..., min_length=3, max_length=3)
    transaction_type: Optional[str] = Field(None, description="Type or channel")
    bank_id: str = Field(..., description="Owning bank identifier")
    country: Optional[str] = Field(None, description="Country code")
    customer_risk_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Normalized risk score"
    )
    origin_balance: Optional[float] = Field(None, ge=0.0)
    destination_balance: Optional[float] = Field(None, ge=0.0)
    is_suspicious: int = Field(..., ge=0, le=1)

    class Config:
        extra = "allow"

    @validator("currency")
    def uppercase_currency(cls, value: str) -> str:
        return value.upper()


class TransactionBatch(BaseModel):
    """Wrapper schema for multiple transactions."""

    records: list[TransactionRecord]

    def to_pandas(self):  # pragma: no cover - thin convenience wrapper
        """Convert the batch of records to a :class:`pandas.DataFrame`."""

        import pandas as pd

        return pd.DataFrame([record.dict() for record in self.records])

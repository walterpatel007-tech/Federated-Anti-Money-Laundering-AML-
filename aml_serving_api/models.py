"""Pydantic models for the AML serving API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime
    amount: float = Field(..., ge=0.0)
    origin_customer_id: str
    destination_customer_id: str
    origin_balance: Optional[float] = Field(default=None, ge=0.0)
    destination_balance: Optional[float] = Field(default=None, ge=0.0)
    hour_of_day: Optional[int] = Field(default=None, ge=0, le=23)
    transaction_type_encoded: Optional[int] = Field(default=None, ge=0)
    customer_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TransactionResponse(BaseModel):
    transaction_id: str
    suspicious_score: float = Field(..., ge=0.0, le=1.0)
    is_suspicious: bool
    model_version: str

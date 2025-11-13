"""FastAPI application exposing the AML model for inference."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException

from aml_data_pipeline.config import load_data_config
from aml_serving_api.models import TransactionRequest, TransactionResponse

from .inference import DEFAULT_MODEL_PATH, load_model, score_transaction

app = FastAPI(title="Federated AML Scoring API", version="0.1.0")


@lru_cache()
def get_model():  # pragma: no cover - simple cache wrapper
    return load_model(DEFAULT_MODEL_PATH)


@lru_cache()
def get_feature_order() -> List[str]:
    data_cfg = load_data_config("config/data_config.yaml")
    return data_cfg.feature_columns


@app.get("/health")
def health() -> dict[str, str]:  # pragma: no cover - trivial endpoint
    return {"status": "ok"}


@app.post("/predict_transaction", response_model=TransactionResponse)
def predict_transaction(request: TransactionRequest) -> TransactionResponse:
    model = get_model()
    feature_order = get_feature_order()
    features = []
    for column in feature_order:
        if not hasattr(request, column):
            raise HTTPException(status_code=400, detail=f"Missing feature: {column}")
        value = getattr(request, column)
        if value is None:
            raise HTTPException(status_code=400, detail=f"Feature {column} cannot be null")
        features.append(value)
    score = score_transaction(model, features)
    return TransactionResponse(
        transaction_id=request.transaction_id,
        suspicious_score=score,
        is_suspicious=score >= 0.5,
        model_version=app.version,
    )


@app.get("/model/metadata")
def model_metadata() -> dict[str, str]:  # pragma: no cover - simple metadata
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "feature_order": ",".join(get_feature_order()),
        "model_version": app.version,
    }

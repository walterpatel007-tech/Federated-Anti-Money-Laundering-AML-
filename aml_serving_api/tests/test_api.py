"""Basic tests for the AML FastAPI service."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from aml_serving_api.main import app

client = TestClient(app)


def _mock_model():
    class DummyModel:
        def __call__(self, tensor):
            import torch

            return torch.tensor(0.0)

    return DummyModel()


@patch("aml_serving_api.main.get_model", side_effect=_mock_model)
@patch("aml_serving_api.main.get_feature_order", return_value=["amount", "hour_of_day"])
def test_predict_transaction(mock_features, mock_model):
    payload = {
        "transaction_id": "tx123",
        "timestamp": datetime.utcnow().isoformat(),
        "amount": 1000.0,
        "origin_customer_id": "c1",
        "destination_customer_id": "c2",
        "hour_of_day": 12,
    }
    response = client.post("/predict_transaction", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx123"
    assert 0.0 <= body["suspicious_score"] <= 1.0
    assert body["model_version"] == app.version

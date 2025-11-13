"""Entry point for Flower-based federated AML training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import flwr as fl
import numpy as np

from aml_data_pipeline import load_data_config
from aml_data_pipeline.config import DataConfig

from .client_flower import ClientConfig, build_client
from .lightning_module import AMLClassifier
from .server_flower import build_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated AML training")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--federated-config", type=Path, required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_model(model_cfg: Dict) -> AMLClassifier:
    return AMLClassifier(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg.get("hidden_dims", [32, 16]),
        learning_rate=model_cfg.get("learning_rate", 1e-3),
        weight_decay=model_cfg.get("weight_decay", 0.0),
        class_weight=model_cfg.get("class_weight"),
        dropout=model_cfg.get("dropout", 0.0),
    )


def main() -> None:
    args = parse_args()
    data_config: DataConfig = load_data_config(args.data_config)
    model_cfg = _load_yaml(args.model_config)
    federated_cfg = _load_yaml(args.federated_config)

    np.random.seed(federated_cfg.get("seed", 7))

    client_configs: list[ClientConfig] = []
    for name in data_config.bank_partitions.keys():
        partition_path = data_config.bank_splits_dir / f"{name}.csv"
        client_configs.append(
            ClientConfig(
                name=name,
                data_path=partition_path,
                feature_columns=data_config.feature_columns,
                label_column=data_config.label_column,
                batch_size=model_cfg.get("batch_size", 64),
                local_epochs=model_cfg.get("local_epochs", 1),
            )
        )

    def client_fn(cid: str) -> fl.client.NumPyClient:
        idx = int(cid)
        cfg = client_configs[idx]
        model = build_model(model_cfg)
        return build_client(model, cfg)

    strategy = build_strategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_configs),
        config=fl.server.ServerConfig(num_rounds=federated_cfg.get("num_rounds", 3)),
        strategy=strategy,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

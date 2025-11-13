"""Flower server utilities for coordinating AML federated learning."""

from __future__ import annotations

from typing import Callable, Dict

import flwr as fl


def build_strategy(evaluate_fn: Callable | None = None) -> fl.server.strategy.FedAvg:
    """Create a FedAvg strategy with optional evaluation function."""

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
        evaluate_metrics_aggregation_fn=_aggregate_metrics,
        evaluate_fn=evaluate_fn,
    )


def _aggregate_metrics(metrics: list[tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated: Dict[str, float] = {}
    for num_examples, client_metrics in metrics:
        weight = num_examples / total_examples
        for name, value in client_metrics.items():
            aggregated[name] = aggregated.get(name, 0.0) + weight * value
    return aggregated

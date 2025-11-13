"""Federated training stack built with Flower and PyTorch Lightning."""

from .dataset import TransactionDataModule
from .lightning_module import AMLClassifier
from .train_federated import main as run_federated_training

__all__ = ["TransactionDataModule", "AMLClassifier", "run_federated_training"]

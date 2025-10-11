from model.registry import REGISTRY, get_model_spec

from .evaluation import evaluate_model
from .train import train_and_compare
from .tune import tune_model

__all__ = [
    "train_and_compare",
    "tune_model",
    "evaluate_model",
    "get_model_spec",
    "REGISTRY",
]

"""mlbench package exports."""

from .utils.registry import (
    register_model,
    get_model,
    available_models,
    model_entry,
)
from .data.registry import (
    register_dataset,
    get_dataset,
    available_datasets,
    dataset_entry,
)

__all__ = [
    "register_model",
    "get_model",
    "available_models",
    "model_entry",
    "register_dataset",
    "get_dataset",
    "available_datasets",
    "dataset_entry",
]

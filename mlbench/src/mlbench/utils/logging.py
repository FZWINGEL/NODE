from __future__ import annotations

import logging
import mlflow
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging for mlbench."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_params_to_mlflow(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow with proper flattening."""
    flattened = _flatten_dict(params)
    mlflow.log_params(flattened)


def log_metrics_to_mlflow(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow."""
    mlflow.log_metrics(metrics, step=step)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary for MLflow logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Union
from sklearn.metrics import r2_score


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Squared Error."""
    return float(torch.mean((predictions - targets) ** 2).item())


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return float(torch.mean(torch.abs(predictions - targets)).item())


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return float(torch.sqrt(torch.mean((predictions - targets) ** 2)).item())


def r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """R-squared coefficient of determination."""
    # Convert to numpy for sklearn
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    return float(r2_score(target_np, pred_np))


def mape(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = targets != 0
    if not mask.any():
        return float("inf")
    
    return float(torch.mean(torch.abs((targets[mask] - predictions[mask]) / targets[mask])).item() * 100)


def smape(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(predictions) + torch.abs(targets)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    if not mask.any():
        return float("inf")
    
    return float(torch.mean(numerator[mask] / denominator[mask]).item() * 100)


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive metrics for predictions vs targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "mse": mse(predictions, targets),
        "mae": mae(predictions, targets),
        "rmse": rmse(predictions, targets),
        "r2": r2(predictions, targets),
        "mape": mape(predictions, targets),
        "smape": smape(predictions, targets),
    }
    
    return metrics


def compute_soh_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute SoH-specific metrics.
    
    Args:
        predictions: SoH predictions (0-1 range)
        targets: Ground truth SoH values (0-1 range)
        
    Returns:
        Dictionary of SoH-specific metrics
    """
    # Basic metrics
    metrics = compute_metrics(predictions, targets)
    
    # SoH-specific metrics
    # Capacity retention error
    capacity_error = torch.abs(predictions - targets)
    metrics["capacity_error"] = float(torch.mean(capacity_error).item())
    
    # End-of-life prediction error (assuming SoH < 0.8 is EOL)
    eol_threshold = 0.8
    eol_mask = targets < eol_threshold
    if eol_mask.any():
        eol_error = torch.abs(predictions[eol_mask] - targets[eol_mask])
        metrics["eol_error"] = float(torch.mean(eol_error).item())
    else:
        metrics["eol_error"] = 0.0
    
    # Prediction accuracy within tolerance
    tolerance = 0.05  # 5% tolerance
    within_tolerance = torch.abs(predictions - targets) <= tolerance
    metrics["accuracy_5pct"] = float(torch.mean(within_tolerance.float()).item())
    
    return metrics

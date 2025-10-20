from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_training_curves(
    history: List[Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves.
    
    Args:
        history: List of epoch metrics
        save_path: Path to save the plot
        title: Plot title
    """
    if not history:
        return
    
    # Extract metrics
    epochs = [h["epoch"] for h in history]
    train_losses = [h.get("train_loss", 0) for h in history]
    val_losses = [h.get("val_mse", 0) for h in history]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation MSE", linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_parity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    metric_name: str = "SoH",
) -> None:
    """Plot predictions vs targets (parity plot).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        save_path: Path to save the plot
        title: Plot title
        metric_name: Name of the metric being plotted
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(target_np, pred_np, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(target_np.min(), pred_np.min())
    max_val = max(target_np.max(), pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(target_np, pred_np)
    
    ax.set_xlabel(f"True {metric_name}")
    ax.set_ylabel(f"Predicted {metric_name}")
    ax.set_title(title or f"{metric_name} Predictions vs Targets (R² = {r2:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_residuals(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    metric_name: str = "SoH",
) -> None:
    """Plot prediction residuals.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        save_path: Path to save the plot
        title: Plot title
        metric_name: Name of the metric being plotted
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # Calculate residuals
    residuals = pred_np - target_np
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs predictions
    ax1.scatter(pred_np, residuals, alpha=0.6, s=20)
    ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax1.set_xlabel(f"Predicted {metric_name}")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predictions")
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title or f"{metric_name} Prediction Residuals", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: List[float],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    top_k: int = 20,
) -> None:
    """Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        save_path: Path to save the plot
        title: Plot title
        top_k: Number of top features to show
    """
    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_scores,
    })
    
    # Sort by importance and take top k
    df_sorted = df.sort_values("importance", ascending=True).tail(top_k)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title(title or f"Top {top_k} Feature Importance")
    ax.grid(True, alpha=0.3)
    
    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_battery_degradation(
    cycle_numbers: List[int],
    soh_values: List[float],
    predictions: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    battery_id: Optional[str] = None,
) -> None:
    """Plot battery degradation curve.
    
    Args:
        cycle_numbers: List of cycle numbers
        soh_values: List of SoH values
        predictions: Optional predictions to overlay
        save_path: Path to save the plot
        title: Plot title
        battery_id: Battery identifier
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual SoH
    ax.plot(cycle_numbers, soh_values, "o-", label="Actual SoH", linewidth=2, markersize=4)
    
    # Plot predictions if provided
    if predictions is not None:
        ax.plot(cycle_numbers, predictions, "s-", label="Predicted SoH", linewidth=2, markersize=4)
    
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("State of Health (SoH)")
    ax.set_title(title or f"Battery Degradation Curve{f' - {battery_id}' if battery_id else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()


def create_evaluation_plots(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_dir: Union[str, Path],
    history: Optional[List[Dict[str, float]]] = None,
    model_name: str = "model",
    dataset_name: str = "dataset",
) -> None:
    """Create comprehensive evaluation plots.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        history: Training history (optional)
        output_dir: Directory to save plots
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training curves
    if history:
        plot_training_curves(
            history,
            save_path=output_path / "training_curves.png",
            title=f"{model_name} Training Curves",
        )
    
    # Parity plot
    plot_parity(
        predictions,
        targets,
        save_path=output_path / "parity_plot.png",
        title=f"{model_name} Predictions vs Targets",
    )
    
    # Residuals plot
    plot_residuals(
        predictions,
        targets,
        save_path=output_path / "residuals.png",
        title=f"{model_name} Prediction Residuals",
    )
    
    print(f"Evaluation plots saved to: {output_path}")

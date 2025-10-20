from __future__ import annotations

import json
import pickle
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import torch
import yaml
from ..utils.env import get_device_info
from ..utils.version import get_version_info


def compute_dataset_fingerprint(data_dir: str, **params: Any) -> str:
    """Compute a fingerprint for the dataset based on files and parameters."""
    fingerprint_data = {
        "data_dir": data_dir,
        "params": params,
        "files": [],
    }
    
    # Add file information
    data_path = Path(data_dir)
    if data_path.exists():
        for file_path in sorted(data_path.glob("*.mat")):
            if file_path.is_file():
                stat = file_path.stat()
                fingerprint_data["files"].append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
    
    # Create hash
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


def save_artifact_bundle(
    save_dir: Union[str, Path],
    model: torch.nn.Module,
    config: Dict[str, Any],
    scaler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    plots_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Save a complete artifact bundle for a training run.
    
    Args:
        save_dir: Directory to save artifacts
        model: Trained model
        config: Experiment configuration
        scaler: Fitted scaler (optional)
        metadata: Dataset metadata (optional)
        metrics: Training metrics (optional)
        plots_dir: Directory containing plots (optional)
        
    Returns:
        Path to the saved artifact bundle
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = save_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save configuration
    config_path = save_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = save_path / "scaler.pkl"
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f)
    
    # Save metadata
    if metadata is not None:
        metadata_path = save_path / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
    
    # Save metrics
    if metrics is not None:
        metrics_path = save_path / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
    
    # Copy plots if provided
    if plots_dir is not None:
        plots_src = Path(plots_dir)
        plots_dst = save_path / "plots"
        plots_dst.mkdir(exist_ok=True)
        
        for plot_file in plots_src.glob("*.png"):
            import shutil
            shutil.copy2(plot_file, plots_dst / plot_file.name)
    
    # Create fingerprint
    fingerprint = {
        "timestamp": datetime.now().isoformat(),
        "version_info": get_version_info(),
        "device_info": get_device_info(),
        "config_hash": hashlib.sha256(str(config).encode()).hexdigest()[:16],
    }
    
    fingerprint_path = save_path / "fingerprint.json"
    with fingerprint_path.open("w") as f:
        json.dump(fingerprint, f, indent=2)
    
    return str(save_path)


def load_artifact_bundle(artifact_path: Union[str, Path]) -> Dict[str, Any]:
    """Load an artifact bundle from disk.
    
    Args:
        artifact_path: Path to artifact bundle directory or model.pt file
        
    Returns:
        Dictionary containing loaded artifacts
    """
    artifact_path = Path(artifact_path)
    
    # Handle both directory and model.pt file paths
    if artifact_path.is_file() and artifact_path.name == "model.pt":
        bundle_dir = artifact_path.parent
    else:
        bundle_dir = artifact_path
    
    bundle = {}
    
    # Load model
    model_path = bundle_dir / "model.pt"
    if model_path.exists():
        bundle["model_state_dict"] = torch.load(model_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load configuration
    config_path = bundle_dir / "config.yaml"
    if config_path.exists():
        with config_path.open("r") as f:
            bundle["config"] = yaml.safe_load(f)
    
    # Load scaler
    scaler_path = bundle_dir / "scaler.pkl"
    if scaler_path.exists():
        with scaler_path.open("rb") as f:
            bundle["scaler"] = pickle.load(f)
    
    # Load metadata
    metadata_path = bundle_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            bundle["metadata"] = json.load(f)
    
    # Load metrics
    metrics_path = bundle_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r") as f:
            bundle["metrics"] = json.load(f)
    
    # Load fingerprint
    fingerprint_path = bundle_dir / "fingerprint.json"
    if fingerprint_path.exists():
        with fingerprint_path.open("r") as f:
            bundle["fingerprint"] = json.load(f)
    
    return bundle


def load_model_from_bundle(artifact_path: Union[str, Path], model_class: torch.nn.Module) -> torch.nn.Module:
    """Load a model from an artifact bundle.
    
    Args:
        artifact_path: Path to artifact bundle
        model_class: Model class to instantiate
        
    Returns:
        Loaded model instance
    """
    bundle = load_artifact_bundle(artifact_path)
    
    # Create model instance
    model = model_class()
    
    # Load state dict
    if "model_state_dict" in bundle:
        model.load_state_dict(bundle["model_state_dict"])
    else:
        raise ValueError("No model state dict found in bundle")
    
    return model


def create_artifact_path(base_dir: str, model_name: str, dataset_name: str) -> str:
    """Create a standardized artifact path.
    
    Args:
        base_dir: Base directory for artifacts
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Path to artifact directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, model_name, dataset_name, timestamp)

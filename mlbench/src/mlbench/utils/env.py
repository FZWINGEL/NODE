from __future__ import annotations

import torch
import platform
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device and environment information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info.update({
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "device_memory": torch.cuda.get_device_properties(0).total_memory,
        })
    
    return info


def get_optimal_device() -> torch.device:
    """Get the optimal device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_deterministic(seed: int) -> None:
    """Setup deterministic behavior for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms if available
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    
    # Set cuDNN to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

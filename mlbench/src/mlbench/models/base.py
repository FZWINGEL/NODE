from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from ..data.types import Batch

class ForwardModel(nn.Module):
    """Base class for all forward models in mlbench.
    
    Models must implement:
    - forward(): returns dict with predictions (must include "soh" key)
    - compute_loss(): returns scalar loss tensor
    """
    registry_name: str | None = None

    def forward(self, batch: Batch, t_eval: torch.Tensor | None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Forward pass through the model.
        
        Args:
            batch: Input batch containing sequences and labels
            t_eval: Optional time evaluation points (for ODE models)
            
        Returns:
            Tuple of (outputs dict, trajectory dict)
            outputs must contain "soh" key with predictions
        """
        raise NotImplementedError

    def compute_loss(self, batch: Batch, outputs: Dict[str, torch.Tensor], traj: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for the given batch and outputs.
        
        Args:
            batch: Input batch with ground truth labels
            outputs: Model predictions from forward()
            traj: Trajectory information from forward()
            
        Returns:
            Scalar loss tensor
        """
        raise NotImplementedError

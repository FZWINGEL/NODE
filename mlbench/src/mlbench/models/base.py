from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from ..data.types import Batch

class ForwardModel(nn.Module):
	def forward(self, batch: Batch, t_eval: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
		raise NotImplementedError

	def compute_loss(self, batch: Batch, outputs: Dict[str, torch.Tensor], traj: Dict[str, Any]) -> torch.Tensor:
		raise NotImplementedError

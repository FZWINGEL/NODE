from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from ...utils.registry import register_model
from ..base import ForwardModel
from ..shared.mlp import MLP

class _Dyn(nn.Module):
	def __init__(self, dim: int, hidden: int = 64, layers: int = 2):
		super().__init__()
		self.f = MLP(dim, dim, hidden=hidden, layers=layers)

	def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return self.f(y)

@register_model(
    "node",
    description="Neural ODE with learned dynamics",
    default_config={
        "input_dim": 16,
        "state_dim": 32,
        "hidden": 64,
        "layers": 2,
        "output_dim": 2,
    },
    tags=("ode",),
)
class NODEModel(ForwardModel):
	def __init__(self, input_dim: int = 16, state_dim: int = 32, hidden: int = 64, layers: int = 2, output_dim: int = 2):
		super().__init__()
		self.encoder = nn.Linear(input_dim, state_dim)
		self.dyn = _Dyn(state_dim, hidden=hidden, layers=layers)
		self.head = nn.Linear(state_dim, output_dim)

	def forward(self, batch, t_eval: torch.Tensor | None = None):
		# simple pooling to init state
		x = batch.x_seq # [B, T, D]
		x_mean = x.mean(dim=1)
		y0 = self.encoder(x_mean)
		# Use Dopri5 solver for better accuracy
		t = torch.tensor([0.0, 1.0], device=x.device)
		yt = odeint(self.dyn, y0, t, method='dopri5', rtol=1e-5, atol=1e-6)
		y_last = yt[-1]
		pred = self.head(y_last)
		return {"soh_r": pred}, {"state_traj": yt}

	def compute_loss(self, batch, outputs, traj):
		y = batch.labels.get("soh_r")
		return torch.nn.functional.mse_loss(outputs["soh_r"], y)

from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from ...utils.registry import register_model
from ..base import ForwardModel
from ..shared.mlp import MLP

class _AugDyn(nn.Module):
	def __init__(self, dim_total: int, hidden: int = 64, layers: int = 2):
		super().__init__()
		self.f = MLP(dim_total, dim_total, hidden=hidden, layers=layers)

	def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return self.f(y)

@register_model(
    "anode",
    description="Augmented Neural ODE",
    default_config={
        "input_dim": 16,
        "state_dim": 32,
        "aug_dim": 8,
        "hidden": 64,
        "layers": 2,
        "output_dim": 2,
    },
    tags=("ode", "augmented"),
)
class ANODEModel(ForwardModel):
    def __init__(self, input_dim: int = 16, state_dim: int = 32, aug_dim: int = 8, hidden: int = 64, layers: int = 2, output_dim: int = 2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, state_dim)
        self.dyn = _AugDyn(state_dim + aug_dim, hidden=hidden, layers=layers)
        self.head = nn.Linear(state_dim, output_dim)
        self.aug_dim = aug_dim

    def forward(self, batch, t_eval: torch.Tensor | None = None):
        x = batch.x_seq
        x_mean = x.mean(dim=1)
        y0 = self.encoder(x_mean)
        B = y0.size(0)
        a0 = torch.zeros(B, self.aug_dim, device=y0.device, dtype=y0.dtype)
        z0 = torch.cat([y0, a0], dim=-1)
        t = torch.tensor([0.0, 1.0], device=y0.device)
        zt = odeint(self.dyn, z0, t, method="dopri5", rtol=1e-5, atol=1e-6)
        z_last = zt[-1]
        y_last = z_last[..., : y0.size(-1)]
        pred = self.head(y_last)
        return {"soh_r": pred}, {"state_traj": zt}

    def compute_loss(self, batch, outputs, traj):
        return torch.nn.functional.mse_loss(outputs["soh_r"], batch.labels["soh_r"])

from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from ...utils.registry import register_model
from ..base import ForwardModel
from ..shared.mlp import MLP

@register_model(
    "pcrnn",
    description="Physics-constrained RNN baseline",
    default_config={
        "input_dim": 16,
        "hidden_dim": 64,
        "num_layers": 2,
        "output_dim": 2,
    },
    tags=("sequence", "physics"),
)
class PCRNNModel(ForwardModel):
	def __init__(self, input_dim: int = 16, state_dim: int = 32, hidden: int = 64, layers: int = 2, output_dim: int = 2):
		super().__init__()
		self.encoder = nn.Linear(input_dim, state_dim)
		self.f = MLP(state_dim, state_dim, hidden=hidden, layers=layers)
		self.head = nn.Linear(state_dim, output_dim)

	def step(self, y: torch.Tensor, dt: float) -> torch.Tensor:
		f_y = self.f(y)
		y_hat = y + dt * f_y
		f_y_hat = self.f(y_hat)
		return y + 0.5 * dt * (f_y + f_y_hat)

	def forward(self, batch, t_eval: torch.Tensor | None = None):
		x = batch.x_seq  # [B, T, D]
		B, T, D = x.shape
		x0 = x[:, 0]
		y = self.encoder(x0)
		dt = 1.0 / max(T - 1, 1)
		for _ in range(T - 1):
			y = self.step(y, dt)
		pred = self.head(y)
		return {"soh_r": pred}, {}

	def compute_loss(self, batch, outputs, traj):
		return torch.nn.functional.mse_loss(outputs["soh_r"], batch.labels["soh_r"]) 

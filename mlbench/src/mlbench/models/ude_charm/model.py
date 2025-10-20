from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from ...utils.registry import register_model
from ..base import ForwardModel
from ..shared.mlp import MLP

class FPhys(nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.w_q = nn.Parameter(torch.tensor(0.1))
		self.w_r = nn.Parameter(torch.tensor(0.1))
		self.lin_s = nn.Linear(dim - 2, dim - 2)

	def forward(self, z: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
		# z: [B, dim] with first 2 dims ~ (q_like, r_like)
		B, D = z.shape
		q = z[:, :1]
		r = z[:, 1:2]
		s = z[:, 2:]
		dq = -torch.relu(self.w_q).expand_as(q)
		dr = torch.relu(self.w_r).expand_as(r)
		ds = self.lin_s(s)
		return torch.cat([dq, dr, ds], dim=-1)

class Gate(nn.Module):
	def __init__(self, in_dim: int):
		super().__init__()
		self.g = nn.Sequential(nn.Linear(in_dim, 32), nn.Tanh(), nn.Linear(32, 1))

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		return torch.sigmoid(self.g(z))

class TimeMap(nn.Module):
	def __init__(self):
		super().__init__()
		self.c = nn.Parameter(torch.tensor(1.0))

	def chi(self, t_like: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
		return (torch.relu(self.c) + 1e-3).expand_as(t_like)

class _CharmDyn(nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.fphys = FPhys(dim)
		self.resid = MLP(dim, dim, hidden=64, layers=2)
		self.gate = Gate(dim)
		self.tmap = TimeMap()

	def forward(self, tau: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
		# state = [z, t_physical]
		z = state[..., :-1]
		t = state[..., -1:]
		f = self.fphys(z)
		g = self.gate(z)
		res = self.resid(z)
		dz_dtau = f + g * res
		chi = self.tmap.chi(t)
		dt_dtau = 1.0 / chi
		return torch.cat([dz_dtau, dt_dtau], dim=-1)

@register_model(
    "ude_charm",
    description="CHARM-inspired universal differential equation",
    default_config={
        "input_dim": 16,
        "dim": 6,
        "output_dim": 2,
    },
    tags=("ude", "physics"),
)
class CHARMUDE(ForwardModel):
	def __init__(self, input_dim: int = 16, dim: int = 6, output_dim: int = 2):
		super().__init__()
		self.encoder = nn.Linear(input_dim, dim)
		self.dyn = _CharmDyn(dim)
		self.head = nn.Linear(dim, output_dim)

	def forward(self, batch, t_eval: torch.Tensor | None = None):
		x = batch.x_seq
		x_mean = x.mean(dim=1)
		z0 = self.encoder(x_mean)
		t0 = torch.zeros(z0.size(0), 1, device=z0.device, dtype=z0.dtype)
		state0 = torch.cat([z0, t0], dim=-1)
		tau = torch.tensor([0.0, 1.0], device=z0.device)
		traj = odeint(self.dyn, state0, tau, method='dopri5', rtol=1e-5, atol=1e-6)
		z_last = traj[-1][..., :-1]
		pred = self.head(z_last)
		return {"soh_r": pred}, {"state_traj": traj}

	def compute_loss(self, batch, outputs, traj):
		return torch.nn.functional.mse_loss(outputs["soh_r"], batch.labels["soh_r"]) 

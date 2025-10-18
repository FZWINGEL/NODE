from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, layers: int = 2):
		super().__init__()
		mods = []
		last = in_dim
		for _ in range(layers - 1):
			mods += [nn.Linear(last, hidden), nn.ReLU()]
			last = hidden
		mods += [nn.Linear(last, out_dim)]
		self.net = nn.Sequential(*mods)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

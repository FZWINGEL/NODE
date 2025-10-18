import torch
from torch import nn
from ...utils.registry import register
from ..base import ForwardModel

@register("lstm")
class LSTMModel(ForwardModel):
	def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 2):
		super().__init__()
		self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
		self.head = nn.Linear(hidden_dim, output_dim)

	def forward(self, batch, t_eval):
		x = batch.x_seq # [B, T, D]
		h, _ = self.encoder(x)
		pred = self.head(h[:, -1])
		return {"soh_r": pred}, {}

	def compute_loss(self, batch, outputs, traj):
		y = batch.labels.get("soh_r")
		return torch.nn.functional.mse_loss(outputs["soh_r"], y)

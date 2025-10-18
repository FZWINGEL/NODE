import math
import torch

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
	return math.sqrt(torch.mean((pred - target) ** 2).item())

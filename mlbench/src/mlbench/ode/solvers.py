from typing import Callable
import torch
from torchdiffeq import odeint_adjoint as odeint


def solve(dynamics: Callable, y0: torch.Tensor, t: torch.Tensor, method: str = "dopri5", rtol: float = 1e-5, atol: float = 1e-6, **kwargs):
	return odeint(dynamics, y0, t, method=method, rtol=rtol, atol=atol, **kwargs)

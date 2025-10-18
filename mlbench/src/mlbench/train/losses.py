from __future__ import annotations
import torch

def physics_penalty_monotone_soh(dq_like: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
	# placeholder: expects dq_like, penalize positives beyond eps
	return torch.nn.functional.relu(dq_like - eps).pow(2).mean()


def composite_loss(base: torch.Tensor, penalties: dict[str, torch.Tensor] | None = None, lambdas: dict[str, float] | None = None) -> torch.Tensor:
	if not penalties:
		return base
	lmb = lambdas or {}
	return base + sum(lmb.get(k, 0.0) * v for k, v in penalties.items())

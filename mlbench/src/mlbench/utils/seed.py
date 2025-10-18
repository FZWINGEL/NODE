from __future__ import annotations
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
	"""Set seeds across libraries for reproducible experiments.

	Parameters
	----------
	seed: int
		The random seed to set.
	deterministic: bool
		If True, configures torch backends for deterministic behavior.
	"""
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	if deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False



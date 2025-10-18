from __future__ import annotations
from typing import Any, Dict, Tuple

import math
import yaml


def load_space(config_path: str, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
	"""Load search space for a given model from YAML.

	Returns a tuple (model_space, train_space).
	"""
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	if model_name not in cfg:
		raise KeyError(f"No search space found for model '{model_name}' in {config_path}")
	entry = cfg[model_name]
	return entry.get("model", {}), entry.get("train", {})


# ---------- Adapters ----------

def suggest_with_optuna(trial, space: Dict[str, Any]) -> Dict[str, Any]:
	params: Dict[str, Any] = {}
	for name, spec in space.items():
		typ = spec.get("type")
		if typ == "choice":
			params[name] = trial.suggest_categorical(name, list(spec["values"]))
		elif typ == "int":
			params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
		elif typ == "float":
			params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
		elif typ == "loguniform":
			params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
		else:
			raise ValueError(f"Unsupported type for Optuna: {typ}")
	return params


def ax_parameters_from_space(space: Dict[str, Any]) -> list:
	params = []
	for name, spec in space.items():
		typ = spec.get("type")
		if typ == "choice":
			params.append({
				"name": name,
				"type": "choice",
				"values": list(spec["values"]),
			})
		elif typ in ("int", "float", "loguniform"):
			is_int = typ == "int"
			log_scale = typ == "loguniform"
			params.append({
				"name": name,
				"type": "range",
				"bounds": [float(spec["low"]), float(spec["high"])],
				"log_scale": log_scale,
				"value_type": "int" if is_int else "float",
			})
		else:
			raise ValueError(f"Unsupported type for Ax: {typ}")
	return params



from __future__ import annotations
from typing import Any, Dict, Tuple

import mlflow
import optuna

from ..train.run import run as train_run
from .search_spaces import load_space, suggest_with_optuna
from ..utils.seed import set_seed


def run_optuna(
	model: str,
	space_config_path: str,
	n_trials: int = 20,
	epochs: int = 10,
	base_seed: int = 0,
	data_name: str = "dummy",
	data_dir: str | None = None,
	window: int = 20,
	stride: int = 5,
	val_split: float = 0.2,
	study_name: str | None = None,
	direction: str = "minimize",
) -> Tuple[Dict[str, Any], float]:
	model_space, train_space = load_space(space_config_path, model)

	def objective(trial: optuna.trial.Trial) -> float:
		# Deterministic seed per trial
		trial_seed = base_seed + trial.number
		set_seed(trial_seed)
		# Sample params
		model_params = suggest_with_optuna(trial, model_space)
		train_params = suggest_with_optuna(trial, train_space)
		# Train one config; nested MLflow child run
		metric = train_run(
			model=model,
			epochs=epochs,
			lr=train_params.get("lr", 1e-3),
			batch_size=int(train_params.get("batch_size", 32)),
			seed=trial_seed,
			data_name=data_name,
			data_dir=data_dir,
			window=window,
			stride=stride,
			val_split=val_split,
			model_kwargs=model_params,
			nested_run=True,
			tags={
				"backend": "optuna",
				"trial_number": str(trial.number),
				"study_name": study_name or "default",
			},
		)
		# Record scalar in trial
		trial.set_user_attr("val_mse", metric)
		for k, v in {**{f"model.{k}": v for k, v in model_params.items()}, **{f"train.{k}": v for k, v in train_params.items()}}.items():
			trial.set_user_attr(k, v)
		return metric

	study = optuna.create_study(direction=direction, study_name=study_name)
	study.optimize(objective, n_trials=n_trials)
	best_val = study.best_value
	best_attrs = study.best_trial.user_attrs
	# Extract flattened params back
	best_model_params = {k.split("model.", 1)[1]: v for k, v in best_attrs.items() if k.startswith("model.")}
	best_train_params = {k.split("train.", 1)[1]: v for k, v in best_attrs.items() if k.startswith("train.")}
	return {"model": best_model_params, "train": best_train_params}, best_val



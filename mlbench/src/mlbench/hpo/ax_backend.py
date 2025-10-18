from __future__ import annotations
from typing import Any, Dict, Tuple

from ax.service.ax_client import AxClient

from ..train.run import run as train_run
from .search_spaces import load_space, ax_parameters_from_space
from ..utils.seed import set_seed


def run_ax(
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
	experiment_name: str | None = None,
	direction: str = "minimize",
) -> Tuple[Dict[str, Any], float]:
	model_space, train_space = load_space(space_config_path, model)
	parameters = ax_parameters_from_space({**{f"model.{k}": v for k, v in model_space.items()}, **{f"train.{k}": v for k, v in train_space.items()}})

	ax = AxClient()
	ax.create_experiment(
		name=experiment_name or f"ax_{model}",
		parameters=parameters,
		objective_name="val_mse",
		minimize=(direction == "minimize"),
	)

	best_val = float("inf")
	best_params: Dict[str, Any] = {}
	for t in range(n_trials):
		params, trial_index = ax.get_next_trial()
		trial_seed = base_seed + t
		set_seed(trial_seed)
		# split back to namespaces
		model_params = {k.split("model.", 1)[1]: v for k, v in params.items() if k.startswith("model.")}
		train_params = {k.split("train.", 1)[1]: v for k, v in params.items() if k.startswith("train.")}
		metric = train_run(
			model=model,
			epochs=epochs,
			lr=float(train_params.get("lr", 1e-3)),
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
				"backend": "ax",
				"trial_number": str(t),
				"experiment_name": experiment_name or "default",
			},
		)
		ax.complete_trial(trial_index=trial_index, raw_data=metric)
		if metric < best_val:
			best_val = metric
			best_params = params

	best_model_params = {k.split("model.", 1)[1]: v for k, v in best_params.items() if k.startswith("model.")}
	best_train_params = {k.split("train.", 1)[1]: v for k, v in best_params.items() if k.startswith("train.")}
	return {"model": best_model_params, "train": best_train_params}, best_val



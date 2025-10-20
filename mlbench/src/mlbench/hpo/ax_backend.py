from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

from ax.service.ax_client import AxClient

from ..train.run import load_experiment
from ..train.trainer import Trainer
from ..utils.config import experiment_from_mapping
from .search_spaces import load_space, ax_parameters_from_space
from ..utils.seed import set_seed


def run_ax(
    base_config_path: str,
    model: str,
    space_config_path: str,
    n_trials: int = 20,
    base_seed: int = 0,
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

    base_experiment = load_experiment(Path(base_config_path))
    best_val = float("inf")
    best_params: Dict[str, Any] = {}
    for t in range(n_trials):
        params, trial_index = ax.get_next_trial()
        trial_seed = base_seed + t
        set_seed(trial_seed)
        # split back to namespaces
        model_params = {k.split("model.", 1)[1]: v for k, v in params.items() if k.startswith("model.")}
        train_params = {k.split("train.", 1)[1]: v for k, v in params.items() if k.startswith("train.")}
        cfg_dict = {
            "model": {"name": model, "params": model_params},
            "dataset": {"name": base_experiment.dataset.name, "params": base_experiment.dataset.params},
            "optimizer": {"name": base_experiment.optimizer.name, "params": {**base_experiment.optimizer.params, **train_params}},
            "scheduler": {"name": base_experiment.scheduler.name, "params": base_experiment.scheduler.params},
            "training": {
                "epochs": base_experiment.training.epochs,
                "seed": trial_seed,
                "device": base_experiment.training.device,
                "gradient_clip_norm": base_experiment.training.gradient_clip_norm,
            },
            "tags": {**base_experiment.tags, "backend": "ax", "trial_number": str(t), "experiment_name": experiment_name or "default"},
        }
        experiment = experiment_from_mapping(cfg_dict)
        result = Trainer(experiment).fit()
        metric = result.best_val_metric
        ax.complete_trial(trial_index=trial_index, raw_data=metric)
        if metric < best_val:
            best_val = metric
            best_params = params

    best_model_params = {k.split("model.", 1)[1]: v for k, v in best_params.items() if k.startswith("model.")}
    best_train_params = {k.split("train.", 1)[1]: v for k, v in best_params.items() if k.startswith("train.")}
    return {"model": best_model_params, "train": best_train_params}, best_val



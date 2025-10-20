from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import optuna

from ..train.run import load_experiment
from ..train.trainer import Trainer
from ..utils.config import experiment_from_mapping
from .search_spaces import load_space, suggest_with_optuna
from ..utils.seed import set_seed


def run_optuna(
    base_config_path: str,
    model: str,
    space_config_path: str,
    n_trials: int = 20,
    base_seed: int = 0,
    study_name: str | None = None,
    direction: str = "minimize",
) -> Tuple[Dict[str, Any], float]:
    model_space, train_space = load_space(space_config_path, model)

    base_experiment = load_experiment(Path(base_config_path))

    def objective(trial: optuna.trial.Trial) -> float:
        # Deterministic seed per trial
        trial_seed = base_seed + trial.number
        set_seed(trial_seed)
        # Sample params
        model_params = suggest_with_optuna(trial, model_space)
        train_params = suggest_with_optuna(trial, train_space)
        # Train one config; nested MLflow child run
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
            "tags": {**base_experiment.tags, "backend": "optuna", "trial_number": str(trial.number), "study_name": study_name or "default"},
        }
        experiment = experiment_from_mapping(cfg_dict)
        result = Trainer(experiment).fit()
        metric = result.best_val_metric
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



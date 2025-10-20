from __future__ import annotations

import argparse
import os
import torch
import mlflow
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..utils.registry import get as get_model
from ..utils.mlflow_utils import start_run
from ..utils.seed import set_seed
from ..data.registry import get_dataset
# Import models to register them
import mlbench.models

from ..utils.config import ExperimentConfig, experiment_from_mapping
from .trainer import Trainer


def train_one_epoch(model, loader, opt, device):
	model.train()
	total = 0.0
	for batch in loader:
		# Move batch data to GPU
		batch.x_seq = batch.x_seq.to(device)
		batch.labels["soh_r"] = batch.labels["soh_r"].to(device)
		outputs, traj = model(batch, t_eval=None)
		loss = model.compute_loss(batch, outputs, traj)
		opt.zero_grad()
		loss.backward()
		# Clip gradients to prevent explosion
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		opt.step()
		total += float(loss.item())
	return total / max(1, len(loader))


def evaluate(model, loader, device):
	model.eval()
	mse = 0.0
	with torch.no_grad():
		for batch in loader:
			# Move batch data to GPU
			batch.x_seq = batch.x_seq.to(device)
			batch.labels["soh_r"] = batch.labels["soh_r"].to(device)
			outputs, traj = model(batch, t_eval=None)
			loss = model.compute_loss(batch, outputs, traj)
			mse += float(loss.item())
	return {"val_mse": mse / max(1, len(loader))}


def run(
	model: str = "lstm",
	epochs: int = 10,
	lr: float = 1e-3,
	batch_size: int = 32,
	seed: int = 0,
	data_name: str = "dummy",
	data_dir: str | None = None,
	window: int = 20,
	stride: int = 5,
	val_split: float = 0.2,
	model_kwargs: dict | None = None,
	nested_run: bool = False,
	tags: dict | None = None,
) -> float:
	# Set global seeds for reproducibility
	set_seed(seed)
	Model = get_model(model)
	m = Model(**(model_kwargs or {}))
	# Ensure model is on GPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	m = m.to(device)
	print(f"Using device: {device}")
	opt = AdamW(m.parameters(), lr=lr)
	sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
	builder = get_dataset(data_name)
	builder_kwargs = {"batch_size": batch_size, "val_split": val_split, "seed": seed}
	if data_name == "nasa":
		builder_kwargs.update({"data_dir": data_dir, "window": window, "stride": stride})
	elif data_name == "nasa_cc":
		builder_kwargs.update({"data_dir": data_dir, "window": window, "stride": stride})
	datamodule = builder(**builder_kwargs)
	train_loader = datamodule.train
	val_loader = datamodule.val
	with start_run(run_name=f"train_{model}", tags=tags, nested=nested_run):
		mlflow.log_params({
			"model": model,
			"epochs": epochs,
			"lr": lr,
			"batch_size": batch_size,
			"seed": seed,
			"data_name": data_name,
			"data_dir": data_dir or "<default>",
			"window": window,
			"stride": stride,
			"val_split": val_split,
		})
		# Log model constructor kwargs (flattened)
		if model_kwargs:
			flat_params = {f"model.{k}": v for k, v in model_kwargs.items()}
			mlflow.log_params(flat_params)
		best_val = float("inf")
		for ep in range(epochs):
			train_loss = train_one_epoch(m, train_loader, opt, device)
			val_metrics = evaluate(m, val_loader, device)
			sched.step(val_metrics["val_mse"])
			mlflow.log_metrics({"train_loss": train_loss, **val_metrics}, step=ep)
			print({"epoch": ep, "train_loss": train_loss, **val_metrics})
			if val_metrics["val_mse"] < best_val:
				best_val = val_metrics["val_mse"]
		# save model
		os.makedirs("artifacts", exist_ok=True)
		ckpt = os.path.join("artifacts", f"{model}.pt")
		torch.save(m.state_dict(), ckpt)
		mlflow.log_artifact(ckpt)
		mlflow.log_metric("best_val_mse", best_val)
	return best_val

def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if path.suffix == ".json":
            return json.load(handle)
        raise ValueError(f"Unsupported config file extension: {path.suffix}")


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mlbench model from config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON experiment config")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Override parameters (key=value pairs)",
    )
    return parser.parse_args()


def _parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value format")
        key, raw = item.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    return overrides


def load_experiment(path: Path, overrides: Dict[str, Any] | None = None) -> ExperimentConfig:
    data = _load_config(path)
    if overrides:
        for key, value in overrides.items():
            section, _, param = key.partition(".")
            if not param:
                data[section] = value
            else:
                if section not in data or not isinstance(data[section], dict):
                    data[section] = {}
                data[section][param] = value
    return experiment_from_mapping(data)


def main():
    args = _parse_cli()
    config_path = Path(args.config).resolve()
    overrides = _parse_overrides(args.override) if args.override else None
    exp_config = load_experiment(config_path, overrides)
    trainer = Trainer(exp_config)
    trainer.fit()


if __name__ == "__main__":
    main()

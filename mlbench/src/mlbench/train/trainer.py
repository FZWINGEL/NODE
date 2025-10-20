from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.optim import Optimizer
from ..utils.config import ExperimentConfig, flatten_mapping, merge_params
from ..utils.mlflow_utils import start_run
from ..utils.registry import get_model, get_model_entry
from ..data.registry import get_dataset


OPTIMIZER_BUILDERS: Dict[str, Any] = {
    "adamw": torch.optim.AdamW,
}

SCHEDULER_BUILDERS: Dict[str, Any] = {
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


@dataclass(slots=True)
class TrainerResult:
    best_val_metric: float
    history: Sequence[Dict[str, float]]


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = self._resolve_device(config.training.device)
        self.model = self._build_model(config)
        datamodule = self._build_dataloaders(config)
        self.train_loader = datamodule.train
        self.val_loader = datamodule.val
        self.optimizer = self._build_optimizer(config)
        self.scheduler = self._build_scheduler(config, self.optimizer)

    def _resolve_device(self, device_config: str | None) -> torch.device:
        if device_config:
            return torch.device(device_config)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_model(self, config: ExperimentConfig):
        entry = get_model(config.model.name)
        model_params = config.model.params
        model = entry(**model_params)
        return model.to(self.device)

    def _build_dataloaders(self, config: ExperimentConfig):
        builder = get_dataset(config.dataset.name)
        params = config.dataset.params
        module = builder(**params)
        if not hasattr(module, "train") or not hasattr(module, "val"):
            raise ValueError("Dataset builder must return a DataModule with 'train' and 'val' attributes")
        return module

    def _build_optimizer(self, config: ExperimentConfig) -> Optimizer:
        opt_name = config.optimizer.name.lower()
        try:
            opt_cls = OPTIMIZER_BUILDERS[opt_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported optimizer '{config.optimizer.name}'") from exc
        params = merge_params({"lr": 1e-3}, config.optimizer.params)
        return opt_cls(self.model.parameters(), **params)

    def _build_scheduler(self, config: ExperimentConfig, optimizer: Optimizer):
        sched_name = config.scheduler.name
        if not sched_name:
            return None
        try:
            sched_cls = SCHEDULER_BUILDERS[sched_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported scheduler '{config.scheduler.name}'") from exc
        params = merge_params({}, config.scheduler.params)
        return sched_cls(optimizer, **params)

    def _move_batch(self, batch):
        batch.x_seq = batch.x_seq.to(self.device)
        if batch.alpha_seq is not None:
            batch.alpha_seq = batch.alpha_seq.to(self.device)
        for key, value in batch.labels.items():
            batch.labels[key] = value.to(self.device)
        return batch

    def _train_one_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            batch = self._move_batch(batch)
            outputs, traj = self.model(batch, t_eval=None)
            loss = self.model.compute_loss(batch, outputs, traj)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.training.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
            self.optimizer.step()
            total += float(loss.detach().item())
        return total / max(1, len(self.train_loader))

    def _evaluate(self):
        self.model.eval()
        mse = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch(batch)
                outputs, traj = self.model(batch, t_eval=None)
                loss = self.model.compute_loss(batch, outputs, traj)
                mse += float(loss.detach().item())
        return {"val_mse": mse / max(1, len(self.val_loader))}

    def fit(self) -> TrainerResult:
        best_val = float("inf")
        history: list[Dict[str, float]] = []
        self.model.to(self.device)
        tags = dict(self.config.tags)
        with start_run(
            run_name=f"train_{self.config.model.name}",
            tags=tags,
            nested=False,
        ):
            for epoch in range(self.config.training.epochs):
                train_loss = self._train_one_epoch()
                val_metrics = self._evaluate()
                if self.scheduler:
                    self.scheduler.step(val_metrics["val_mse"])
                metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
                history.append(metrics)
                if val_metrics["val_mse"] < best_val:
                    best_val = val_metrics["val_mse"]
                print(metrics)
        return TrainerResult(best_val_metric=best_val, history=history)


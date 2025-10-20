from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Optional

import torch
from torch.optim import Optimizer
from ..utils.config import ExperimentConfig, flatten_mapping, merge_params
from ..utils.mlflow_utils import start_run
from ..utils.registry import get_model, get_model_entry
from ..data.registry import get_dataset
from ..utils.logging import log_metrics_to_mlflow
from ..train.metrics import compute_metrics
from ..train.serialization import save_artifact_bundle, create_artifact_path
from ..visualization.plots import create_evaluation_plots


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
    best_model_state: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" for minimization/maximization
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save state if improved
            
        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience


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
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch(batch)
                outputs, traj = self.model(batch, t_eval=None)
                loss = self.model.compute_loss(batch, outputs, traj)
                total_loss += float(loss.detach().item())
                
                # Collect predictions and targets for metrics
                pred = outputs.get("soh_r", outputs.get("soh"))
                target = batch.labels.get("soh_r", batch.labels.get("soh"))
                
                if pred is not None and target is not None:
                    all_predictions.append(pred.cpu())
                    all_targets.append(target.cpu())
        
        metrics = {"val_loss": total_loss / max(1, len(self.val_loader))}
        
        # Compute additional metrics if we have predictions
        if all_predictions and all_targets:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Compute comprehensive metrics
            eval_metrics = compute_metrics(predictions, targets)
            metrics.update({f"val_{k}": v for k, v in eval_metrics.items()})
        
        return metrics

    def fit(self) -> TrainerResult:
        """Fit the model with early stopping and artifact saving."""
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
            mode=self.config.early_stopping.mode,
        )
        
        history: list[Dict[str, float]] = []
        self.model.to(self.device)
        
        # Create artifact directory
        artifact_dir = create_artifact_path(
            self.config.run.save_dir,
            self.config.model.name,
            self.config.dataset.name,
        )
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            train_loss = self._train_one_epoch()
            val_metrics = self._evaluate()
            
            # Step scheduler
            if self.scheduler:
                monitor_metric = val_metrics.get(self.config.early_stopping.monitor, "val_loss")
                self.scheduler.step(monitor_metric)
            
            # Prepare metrics
            metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(metrics)
            
            # Log metrics to MLflow
            log_metrics_to_mlflow(metrics, step=epoch)
            
            # Check early stopping
            monitor_metric = val_metrics.get(self.config.early_stopping.monitor, "val_loss")
            if early_stopping(monitor_metric, self.model):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            print(metrics)
        
        # Save best model state
        if early_stopping.best_model_state:
            self.model.load_state_dict(early_stopping.best_model_state)
        
        # Save artifacts
        # Serialize config to plain dict for saving
        try:
            from dataclasses import asdict
            config_dict = asdict(self.config)
        except Exception:
            config_dict = {
                "model": dict(name=self.config.model.name, params=self.config.model.params),
                "dataset": dict(name=self.config.dataset.name, params=self.config.dataset.params),
            }

        artifact_path = save_artifact_bundle(
            artifact_dir,
            self.model,
            config_dict,
            metadata={"feature_names": [], "target_name": "soh", "num_samples": len(self.train_loader.dataset)},
            metrics={"best_val_metric": early_stopping.best_score, "history": history},
        )
        
        # Create evaluation plots
        try:
            # Get final predictions for plotting
            self.model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = self._move_batch(batch)
                    outputs, _ = self.model(batch, t_eval=None)
                    
                    pred = outputs.get("soh_r", outputs.get("soh"))
                    target = batch.labels.get("soh_r", batch.labels.get("soh"))
                    
                    if pred is not None and target is not None:
                        all_predictions.append(pred.cpu())
                        all_targets.append(target.cpu())
            
            if all_predictions and all_targets:
                predictions = torch.cat(all_predictions, dim=0)
                targets = torch.cat(all_targets, dim=0)
                
                create_evaluation_plots(
                    predictions,
                    targets,
                    artifact_dir,
                    history,
                    self.config.model.name,
                    self.config.dataset.name,
                )
        except Exception as e:
            print(f"Warning: Could not create evaluation plots: {e}")
        
        return TrainerResult(
            best_val_metric=early_stopping.best_score,
            history=history,
            best_model_state=early_stopping.best_model_state,
            artifact_path=artifact_path,
        )


# mlbench Baseline Architecture Notes

## Registries and Discovery
- `mlbench/src/mlbench/utils/registry.py` is a simple string-to-constructor map used primarily for models.
- `mlbench/src/mlbench/data/registry.py` mirrors the pattern for dataset builders.
- Registration happens eagerly via imports (e.g. `mlbench/src/mlbench/models/__init__.py` imports every model module so decorators execute at import time).

## Dataset Builders
- `mlbench/src/mlbench/data/dummy.py` wraps a synthetic dataset and exposes `build_dummy_loaders` that returns `(train_loader, val_loader)`.
- `mlbench/src/mlbench/data/nasa.py` provides NASA battery loaders, including CC variants. Builders accept `batch_size`, `val_split`, `seed`, plus domain-specific args.
- Builders perform data splitting internally and produce `torch.utils.data.DataLoader` instances with a custom `Batch` collate.

## Model Implementations
- Each model module decorates a subclass of `ForwardModel` with `@register("model_name")`, so registries store classes.
- `ForwardModel` defines `forward(batch, t_eval)` and `compute_loss(batch, outputs, traj)`, leaving optimizer and training policy outside the model.
- Example: `mlbench/src/mlbench/models/lstm/model.py` encodes sequences with an LSTM and emits `{"soh_r": pred}`; NODE-based implementations integrate ODE dynamics via torchdiffeq.

## Training Orchestration
- Entry point `mlbench/src/mlbench/train/run.py` offers both CLI and importable `run()`.
- The script pulls model classes from the registry, builds datasets via `get_dataset`, and instantiates `AdamW` plus `ReduceLROnPlateau`.
- Training loop functions `train_one_epoch` and `evaluate` operate purely on loaders and model; gradient clipping is performed manually at 1.0 norm.
- MLflow logging (handled via `mlbench/src/mlbench/utils/mlflow_utils.py`) starts a run, records params/metrics, and saves checkpoints under `artifacts/<model>.pt`.

## Experiment Entrypoints
- CLI arguments map one-to-one with `run()` parameters (model selection, data windowing, hyperparameters).
- Higher-level automation scripts (`mlbench/src/mlbench/experiments/sweep.py`, `.../hpo.py`) repeatedly call `train.run` and rely on the same global registries.

## Observed Pain Points (Pre-Refactor)
- Registries expose raw callables without metadata (defaults, schemas), requiring manual kwarg handling per script.
- Dataset builders return loaders directly, limiting configurability and reuse across sweeps/HPO.
- Training orchestration logic mixes configuration parsing, device selection, optimization setup, and logging in a single module.


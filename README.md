## mlbench: Neural ODE and Sequence Models for Battery SoH/RUL

Lightweight research framework to train and tune sequence models (LSTM, NODE, ANODE, PCRNN, UDE-CHARM) for battery state-of-health (SoH) style regression on dummy and NASA PCoE data. Includes first-class experiment tracking via MLflow and built-in hyperparameter optimization (Optuna or Ax/Botorch).

### Features
- Train via simple CLI: `python -m mlbench.train.run`
- Models: `lstm`, `pcrnn`, `node`, `anode`, `ude_charm`
- Datasets: `dummy` (synthetic), `nasa` (PCoE .mat files)
- HPO backends: Optuna or Ax
- MLflow autologging to local `mlbench/mlruns/`

### Installation
1) Create a Python 3.11+ virtual environment.
2) Install the package in editable mode:

```bash
pip install -r mlbench/requirements.txt --index-url https://download.pytorch.org/whl/cu121
pip install -e mlbench
```

This installs dependencies including PyTorch, torchdiffeq, MLflow, Optuna, Ax, BoTorch.

**Note**: All commands below assume you're in the project root directory (`H:\NODE`) and have activated your virtual environment.

### Data
- Dummy dataset requires no files.
- NASA PCoE: place `.mat` files under one of:
  - Default: `mlbench/data/nasa/raw/` (already present with sample files), or
  - Custom folder and pass `--data_dir <path>`.

Each `.mat` should be one battery file (e.g., `B0005.mat`). The loader extracts discharge cycles, summarizes features, creates sliding windows, and predicts last-window SoH-like target.

### Training
Run training for a chosen model and dataset:

```bash
# General form
python -m mlbench.train.run --model lstm --epochs 10 --lr 1e-3 --batch_size 32 --seed 0 --data_name dummy --val_split 0.2

# Example: NASA dataset (windowed sequences)
python -m mlbench.train.run --model node --epochs 20 --lr 1e-3 --batch_size 32 --data_name nasa --data_dir mlbench/data/NASA --window 20 --stride 5 --val_split 0.2
```

**Windows PowerShell**: If you get module not found errors, activate your virtual environment first:
```powershell
.\mlbench_env\Scripts\Activate.ps1
```

Arguments
- `--model`: one of `lstm`, `pcrnn`, `node`, `anode`, `ude_charm`
- `--epochs`, `--lr`, `--batch_size`, `--seed`
- `--data_name`: `dummy` or `nasa`
- `--data_dir`: path with `.mat` files (used when `data_name=nasa`)
- `--window`, `--stride`: NASA sequence windowing
- `--val_split`: train/val split ratio

Outputs
- Checkpoints in `artifacts/<model>.pt`
- MLflow run with params/metrics and artifact logged to `mlbench/mlruns/`

### Train all models (quick sweep)

```bash
python -m mlbench.experiments.sweep --epochs 3 --batch_size 32
```

This sequentially runs `lstm`, `pcrnn`, `node`, `anode`, `ude_charm` with the given training length and batch size.

### Hyperparameter Optimization (HPO)
Use the built-in HPO entrypoint to search model and training hyperparameters defined in `mlbench/configs/sweep/hpo_spaces.yaml`.

Config structure (example excerpt):

```yaml
lstm:
  model:
    hidden_dim: {type: choice, values: [32, 64, 128]}
    num_layers: {type: int, low: 1, high: 3}
  train:
    lr: {type: loguniform, low: 1.0e-4, high: 1.0e-2}
    batch_size: {type: choice, values: [16, 32, 64]}
```

Run HPO (Optuna by default):

```bash
# Default (Optuna) on NASA data for LSTM
python -m mlbench.experiments.hpo --backend optuna --model lstm --space mlbench/configs/sweep/hpo_spaces.yaml --n-trials 20 --epochs 10 --seed 0 --data_name nasa --data_dir mlbench/data/NASA --window 20 --stride 5 --val_split 0.2

# Ax backend on NASA data with windowing
python -m mlbench.experiments.hpo --backend ax --model node --space mlbench/configs/sweep/hpo_spaces.yaml --n-trials 15 --epochs 15 --seed 42 --data_name nasa --data_dir mlbench/data/NASA --window 20 --stride 5 --val_split 0.2
```

**Windows PowerShell**: Make sure to activate your virtual environment first:
```powershell
.\mlbench_env\Scripts\Activate.ps1
```

HPO arguments
- `--backend`: `optuna` or `ax`
- `--model`: target model key in the YAML
- `--space`: path to the YAML search space
- `--n-trials`: number of trials
- `--epochs`, `--seed`, and dataset args as in training

Behavior
- Each trial trains via the same training loop and logs to MLflow as a nested run.
- At the end, the script prints a JSON with `best_params` and `best_val_mse`.

### Viewing Results
Launch MLflow UI to inspect runs and artifacts:

```bash
mlflow ui --backend-store-uri file:///$(pwd | sed 's/\\/\//g')/mlbench/mlruns
```

On Windows PowerShell, you can also open `mlbench/mlruns` directly in a file browser; MLflow URIs are set automatically by the package.

### Extending
- Add a new model: implement a subclass of `mlbench.models.base.ForwardModel`, register with `@register("your_model")`, and ensure it returns `{"soh_r": tensor}` and a scalar MSE loss via `compute_loss`.
- Add a new dataset: register a builder in `mlbench.data.registry` that returns `(train_loader, val_loader)` given `batch_size`, `val_split`, and `seed` (plus dataset-specific args).
- Add HPO space: extend `mlbench/configs/sweep/hpo_spaces.yaml` under your model key with `model` and `train` namespaces.

### Troubleshooting
- CUDA/GPU: training automatically uses GPU if available; otherwise CPU.
- NASA data not found: pass `--data_dir` to your `.mat` folder or place files under `mlbench/data/nasa/raw/`.
- Ax/Botorch GPU installs may require matching CUDA versions; if issues arise, stick with Optuna.
- **PyTorch/torchvision compatibility**: If you get `RuntimeError: operator torchvision::nms does not exist`, reinstall compatible versions:
  ```powershell
  pip uninstall torch torchvision torchaudio
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  ```
- **Module not found**: Ensure you're in the project root (`H:\NODE`) and have activated the virtual environment:
  ```powershell
  .\mlbench_env\Scripts\Activate.ps1
  ```

### License
For research use. See source headers for details.



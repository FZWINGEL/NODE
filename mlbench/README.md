# MLBench: Machine Learning Benchmark for Battery State-of-Health Modeling

A clean, extensible Python package for battery State-of-Health (SoH) modeling with first-class experiment tracking, easy model/dataset extensibility, and reproducible runs.

## Features

- **Multiple Models**: LSTM, PCRNN, Neural ODEs (NODE/ANODE), UDE-CHARM, ACLA
- **NASA Dataset**: Preprocessed NASA PCoE battery data with proper scaling and splits
- **Experiment Tracking**: MLflow integration with comprehensive logging
- **Hyperparameter Optimization**: Optuna and Ax backends
- **Reproducible**: Seeded runs, deterministic behavior, artifact bundles
- **Extensible**: Easy to add new models and datasets via registry pattern
- **CLI Tools**: Command-line interfaces for training, evaluation, and HPO

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlbench

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

## Quick Start

### 1. Train a Model

```bash
# Train LSTM on NASA data
soh-train --config src/mlbench/configs/train/default.yaml --override model.name=lstm dataset.name=nasa dataset.params.data_dir=data/nasa/raw

# Train with custom parameters
soh-train --config src/mlbench/configs/train/default.yaml \
  --override model.name=node \
  dataset.name=nasa \
  dataset.params.data_dir=data/nasa/raw \
  dataset.params.window=20 \
  dataset.params.stride=5 \
  training.epochs=50 \
  optimizer.params.lr=1e-3
```

### 2. Evaluate a Model

```bash
# Evaluate a saved checkpoint
soh-eval --artifact artifacts/lstm/nasa/20241221_143022/model.pt \
  --data-name nasa \
  --data-dir data/nasa/raw

# Evaluate with MLflow run ID
soh-eval --artifact mlflow://run_id \
  --data-name nasa \
  --data-dir data/nasa/raw
```

### 3. Hyperparameter Optimization

```bash
# Run Optuna HPO
soh-hpo --config src/mlbench/configs/train/default.yaml \
  --hpo-config src/mlbench/configs/sweep/hpo_spaces.yaml \
  --backend optuna \
  --override dataset.name=nasa dataset.params.data_dir=data/nasa/raw

# Run Ax HPO
soh-hpo --config src/mlbench/configs/train/default.yaml \
  --hpo-config src/mlbench/configs/sweep/hpo_spaces.yaml \
  --backend ax \
  --override dataset.name=nasa dataset.params.data_dir=data/nasa/raw
```

### 4. Model Sweep

```bash
# Run sequential sweep over multiple models
soh-sweep --config src/mlbench/configs/train/default.yaml \
  --sweep-config src/mlbench/configs/sweep/baselines.yaml \
  --override dataset.name=nasa dataset.params.data_dir=data/nasa/raw
```

## Configuration

MLBench uses YAML configuration files with hierarchical structure:

```yaml
# Example config
model:
  name: "lstm"
  params:
    hidden_size: 64
    num_layers: 2

dataset:
  name: "nasa"
  params:
    batch_size: 32
    val_split: 0.2
    window: 20
    stride: 5
    data_dir: "data/nasa/raw"

training:
  epochs: 100
  device: "auto"
  gradient_clip_norm: 1.0

optimizer:
  name: "adamw"
  params:
    lr: 1e-3
    weight_decay: 1e-4

scheduler:
  name: "reduce_on_plateau"
  params:
    mode: "min"
    factor: 0.5
    patience: 5

early_stopping:
  monitor: "val_loss"
  patience: 10
  min_delta: 0.001
  mode: "min"
```

## Adding New Models

1. Create a new model class inheriting from `ForwardModel`:

```python
from mlbench.models.base import ForwardModel
from mlbench.models.registry import register_model

@register_model("my_model")
class MyModel(ForwardModel):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=16, hidden_size=hidden_size)
        self.output = torch.nn.Linear(hidden_size, 2)
    
    def forward(self, batch, t_eval=None):
        x = batch.x_seq
        output, _ = self.lstm(x)
        soh_r = self.output(output[:, -1])
        return {"soh_r": soh_r}, {}
    
    def compute_loss(self, batch, outputs, traj):
        pred = outputs["soh_r"]
        target = batch.labels["soh_r"]
        return torch.nn.functional.mse_loss(pred, target)
```

2. Add default configuration in `src/mlbench/configs/model/my_model.yaml`

3. Train with: `soh-train --override model.name=my_model`

## Adding New Datasets

1. Create a dataset builder function:

```python
from mlbench.data.registry import register_dataset
from mlbench.data.base import BuildResult

@register_dataset("my_dataset")
def build_my_dataset(batch_size=32, val_split=0.2, seed=0, **kwargs) -> BuildResult:
    # Load and preprocess data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    metadata = {
        "feature_names": ["feature_1", "feature_2"],
        "target_name": "soh",
        "num_samples": len(train_dataset),
    }
    
    return train_loader, val_loader, metadata
```

2. Add default configuration in `src/mlbench/configs/data/my_dataset.yaml`

3. Train with: `soh-train --override dataset.name=my_dataset`

## MLflow Integration

MLBench automatically logs experiments to MLflow with:

- **Parameters**: All configuration values (flattened)
- **Metrics**: Training/validation metrics per epoch
- **Tags**: Git SHA, model name, dataset name, device, etc.
- **Artifacts**: Model checkpoints, configs, scalers, plots

View results:
```bash
mlflow ui
```

## Dataset Structure

### NASA Dataset

Place NASA PCoE battery data in `data/nasa/raw/`:
```
data/nasa/raw/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── ...
```

The dataset builder will:
- Extract discharge cycles from .mat files
- Create sequence windows with configurable window/stride
- Apply feature scaling (StandardScaler by default)
- Split data by battery ID to prevent leakage
- Return DataLoaders with proper collation

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_models_forward.py -v
pytest tests/test_data_nasa.py -v
pytest tests/test_trainer_smoke.py -v
pytest tests/test_cli.py -v
```

### Code Quality

```bash
# Format code
black src/mlbench tests/
isort src/mlbench tests/

# Lint
flake8 src/mlbench tests/
mypy src/mlbench
```

### CI/CD

GitHub Actions automatically runs:
- Linting (flake8, black, isort, mypy)
- Unit tests with coverage
- Cross-platform testing (Ubuntu, Windows)

## Architecture

```
src/mlbench/
├── configs/           # Configuration files
├── data/             # Dataset loaders and transforms
├── experiments/      # CLI entry points
├── models/           # Model implementations
├── train/            # Training utilities
├── utils/            # Utility functions
└── visualization/    # Plotting utilities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use MLBench in your research, please cite:

```bibtex
@software{mlbench2024,
  title={MLBench: Machine Learning Benchmark for Battery State-of-Health Modeling},
  author={MLBench Team},
  year={2024},
  url={https://github.com/your-org/mlbench}
}
```

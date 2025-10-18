### HPO usage examples

Run Optuna on NODE for 30 trials, 10 epochs each:

```bash
python -m mlbench.experiments.hpo --backend optuna --model node --n-trials 30 --epochs 10 --data_name dummy
```

Run Ax on LSTM for 25 trials:

```bash
python -m mlbench.experiments.hpo --backend ax --model lstm --n-trials 25 --epochs 10 --data_name dummy
```

Specify a custom search space file:

```bash
python -m mlbench.experiments.hpo --backend optuna --model anode --space mlbench/configs/sweep/hpo_spaces.yaml --n-trials 20
```



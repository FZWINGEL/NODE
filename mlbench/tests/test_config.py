from __future__ import annotations

from pathlib import Path

import pytest

from mlbench.utils.config import experiment_from_mapping


def test_experiment_from_mapping_basic():
    data = {
        "model": {"name": "lstm", "params": {"hidden_dim": 32}},
        "dataset": {"name": "dummy", "params": {"batch_size": 16}},
        "optimizer": {"name": "adamw", "params": {"lr": 1e-3}},
        "scheduler": {"name": "reduce_on_plateau", "params": {"factor": 0.5}},
        "training": {"epochs": 5, "seed": 42, "gradient_clip_norm": 1.0},
        "tags": {"exp": "unit"},
    }
    cfg = experiment_from_mapping(data)

    assert cfg.model.name == "lstm"
    assert cfg.model.params["hidden_dim"] == 32
    assert cfg.dataset.name == "dummy"
    assert cfg.dataset.params["batch_size"] == 16
    assert cfg.training.epochs == 5
    assert cfg.tags["exp"] == "unit"


def test_experiment_from_mapping_missing_section():
    with pytest.raises(ValueError):
        experiment_from_mapping({"model": {"name": "lstm"}})


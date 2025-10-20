from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


@dataclass(slots=True)
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerConfig:
    name: str | None = "reduce_on_plateau"
    params: Dict[str, Any] = field(default_factory=lambda: {"mode": "min", "factor": 0.5, "patience": 3})


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 10
    seed: int = 0
    device: str | None = None
    gradient_clip_norm: float | None = 1.0


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tags: Dict[str, Any] = field(default_factory=dict)


def merge_params(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: MutableMapping[str, Any] = dict(base)
    for key, value in overrides.items():
        merged[key] = value
    return dict(merged)


def _section_params(section: Mapping[str, Any]) -> Dict[str, Any]:
    params = section.get("params") if isinstance(section, Mapping) else None
    if not isinstance(params, Mapping):
        return {}
    return dict(params)


def _section_name(section: Mapping[str, Any], key: str) -> str:
    name = section.get("name") if isinstance(section, Mapping) else None
    if not isinstance(name, str):
        raise ConfigError(f"'{key}.name' must be provided in experiment config")
    return name


def experiment_from_mapping(data: Mapping[str, Any]) -> ExperimentConfig:
    if "model" not in data or "dataset" not in data:
        raise ConfigError("Experiment config must include 'model' and 'dataset' sections")

    model_section = data["model"]
    dataset_section = data["dataset"]

    model_cfg = ModelConfig(
        name=_section_name(model_section, "model"),
        params=_section_params(model_section),
    )
    dataset_cfg = DatasetConfig(
        name=_section_name(dataset_section, "dataset"),
        params=_section_params(dataset_section),
    )

    optimizer_section = data.get("optimizer", {})
    scheduler_section = data.get("scheduler", {})
    training_section = data.get("training", {})

    optimizer_cfg = OptimizerConfig(
        name=str(optimizer_section.get("name", "adamw")),
        params=_section_params(optimizer_section),
    )
    scheduler_name = scheduler_section.get("name", "reduce_on_plateau")
    scheduler_cfg = SchedulerConfig(
        name=None if scheduler_name in (None, "none", "") else str(scheduler_name),
        params=_section_params(scheduler_section),
    )

    training_cfg = TrainingConfig(
        epochs=int(training_section.get("epochs", 10)),
        seed=int(training_section.get("seed", 0)),
        device=training_section.get("device"),
        gradient_clip_norm=training_section.get("gradient_clip_norm", 1.0),
    )

    tags_section = data.get("tags", {})
    tags = dict(tags_section) if isinstance(tags_section, Mapping) else {}

    return ExperimentConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training=training_cfg,
        tags=tags,
    )


def flatten_mapping(mapping: Mapping[str, Any], prefix: str | None = None) -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_mapping(value, name))
        else:
            items[name] = value
    return items


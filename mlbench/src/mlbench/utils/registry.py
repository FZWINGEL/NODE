from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class RegistryEntry:
    name: str
    constructor: Callable[..., Any]
    category: str
    description: str | None = None
    default_config: Mapping[str, Any] | None = None
    tags: Iterable[str] = field(default_factory=tuple)


# Generic registry storage
_REGISTRIES: Dict[str, Dict[str, RegistryEntry]] = {}


def register(namespace: str, name: str):
    """Generic registry decorator for any namespace."""
    def deco(obj: Callable[..., T]) -> Callable[..., T]:
        entry = RegistryEntry(
            name=name,
            constructor=obj,
            category=namespace,
            description=getattr(obj, "__doc__", None),
            default_config=getattr(obj, "default_config", None),
            tags=getattr(obj, "tags", ()),
        )
        _REGISTRIES.setdefault(namespace, {})[name] = entry
        return obj
    return deco


def get(namespace: str, name: str) -> Callable[..., Any]:
    """Get a registered object from a namespace."""
    try:
        return _REGISTRIES[namespace][name].constructor
    except KeyError as exc:
        raise KeyError(f"Unknown {namespace} registry key: {name}") from exc


def get_entry(namespace: str, name: str) -> RegistryEntry:
    """Get a registry entry from a namespace."""
    try:
        return _REGISTRIES[namespace][name]
    except KeyError as exc:
        raise KeyError(f"Unknown {namespace} registry key: {name}") from exc


def available(namespace: str) -> Dict[str, RegistryEntry]:
    """Get all available entries in a namespace."""
    return dict(_REGISTRIES.get(namespace, {}))


def reset(namespace: str) -> None:
    """Reset all entries in a namespace."""
    if namespace in _REGISTRIES:
        _REGISTRIES[namespace].clear()


# Convenience functions for specific namespaces
def register_model(name: str, *, description: str | None = None, default_config: Mapping[str, Any] | None = None, tags: Iterable[str] | None = None):
    """Register a model class."""
    def wrapper(cls: Callable[..., T]) -> Callable[..., T]:
        entry = RegistryEntry(
            name=name,
            constructor=cls,
            category="model",
            description=description or cls.__doc__,
            default_config=default_config,
            tags=tuple(tags or ()),
        )
        _REGISTRIES.setdefault("model", {})[name] = entry
        return cls
    return wrapper


def register_dataset(name: str, *, description: str | None = None, default_config: Mapping[str, Any] | None = None, tags: Iterable[str] | None = None):
    """Register a dataset builder function."""
    def wrapper(func: Callable[..., T]) -> Callable[..., T]:
        entry = RegistryEntry(
            name=name,
            constructor=func,
            category="dataset",
            description=description or func.__doc__,
            default_config=default_config,
            tags=tuple(tags or ()),
        )
        _REGISTRIES.setdefault("dataset", {})[name] = entry
        return func
    return wrapper


# Backwards compatibility exports
def get_model(name: str) -> Callable[..., Any]:
    return get("model", name)


def get_dataset(name: str) -> Callable[..., Any]:
    return get("dataset", name)


def get_model_entry(name: str) -> RegistryEntry:
    return get_entry("model", name)


def get_dataset_entry(name: str) -> RegistryEntry:
    return get_entry("dataset", name)


def available_models() -> Dict[str, RegistryEntry]:
    return available("model")


def available_datasets() -> Dict[str, RegistryEntry]:
    return available("dataset")


def reset_model_registry() -> None:
    reset("model")


def reset_dataset_registry() -> None:
    reset("dataset")

# Back-compat aliases expected by older imports
def model_entry(name: str) -> RegistryEntry:
    return get_entry("model", name)


def dataset_entry(name: str) -> RegistryEntry:
    return get_entry("dataset", name)
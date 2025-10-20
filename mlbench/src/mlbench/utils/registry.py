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


class TypedRegistry:
    def __init__(self, category: str):
        self._category = category
        self._entries: Dict[str, RegistryEntry] = {}

    def register(
        self,
        name: str,
        constructor: Callable[..., T],
        *,
        description: str | None = None,
        default_config: Mapping[str, Any] | None = None,
        tags: Iterable[str] | None = None,
    ) -> Callable[..., T]:
        entry = RegistryEntry(
            name=name,
            constructor=constructor,
            category=self._category,
            description=description,
            default_config=default_config,
            tags=tuple(tags or ()),
        )
        self._entries[name] = entry
        return constructor

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._entries[name].constructor
        except KeyError as exc:
            raise KeyError(f"Unknown {self._category} registry key: {name}") from exc

    def get_entry(self, name: str) -> RegistryEntry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"Unknown {self._category} registry key: {name}") from exc

    def available(self) -> Dict[str, RegistryEntry]:
        return dict(self._entries)


_MODEL_REGISTRY = TypedRegistry("model")


def reset_model_registry() -> None:
    _MODEL_REGISTRY._entries.clear()


def register_model(
    name: str,
    *,
    description: str | None = None,
    default_config: Mapping[str, Any] | None = None,
    tags: Iterable[str] | None = None,
):
    def wrapper(cls: Callable[..., T]) -> Callable[..., T]:
        _MODEL_REGISTRY.register(
            name,
            cls,
            description=description,
            default_config=default_config,
            tags=tags,
        )
        return cls

    return wrapper


def model_entry(name: str) -> RegistryEntry:
    return get_model_entry(name)


def get_model(name: str) -> Callable[..., Any]:
    return _MODEL_REGISTRY.get(name)


def get_model_entry(name: str) -> RegistryEntry:
    return _MODEL_REGISTRY.get_entry(name)


def available_models() -> Dict[str, RegistryEntry]:
    return _MODEL_REGISTRY.available()


# Backwards compatibility exports
register = register_model
get = get_model
available = available_models
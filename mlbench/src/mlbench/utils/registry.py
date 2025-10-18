from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}

def register(name: str):
	def deco(fn: Callable):
		_REGISTRY[name] = fn
		return fn
	return deco

def get(name: str) -> Callable:
	return _REGISTRY[name]

def available() -> Dict[str, Callable]:
	return dict(_REGISTRY)
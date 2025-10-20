"""Configuration loading utilities with legacy fallback."""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from importlib.resources import files


def load_config(config_name: str, config_type: str = "defaults") -> Dict[str, Any]:
    """Load configuration file with legacy fallback.
    
    Args:
        config_name: Name of the config file (without extension)
        config_type: Type of config (defaults, model, data, train, etc.)
        
    Returns:
        Configuration dictionary
    """
    # Try new location first
    try:
        config_path = files("mlbench.configs").joinpath(f"{config_type}/{config_name}.yaml")
        if config_path.exists():
            with config_path.open("r") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    
    # Fallback to legacy location
    legacy_paths = [
        Path("configs") / config_type / f"{config_name}.yaml",
        Path("mlbench/configs") / config_type / f"{config_name}.yaml",
        Path("src/mlbench/configs") / config_type / f"{config_name}.yaml",
    ]
    
    for path in legacy_paths:
        if path.exists():
            with path.open("r") as f:
                return yaml.safe_load(f)
    
    # If not found, return empty dict
    return {}


def get_config_path(config_name: str, config_type: str = "defaults") -> Optional[Path]:
    """Get the path to a configuration file.
    
    Args:
        config_name: Name of the config file (without extension)
        config_type: Type of config (defaults, model, data, train, etc.)
        
    Returns:
        Path to config file or None if not found
    """
    # Try new location first
    try:
        config_path = files("mlbench.configs").joinpath(f"{config_type}/{config_name}.yaml")
        if config_path.exists():
            return Path(config_path)
    except Exception:
        pass
    
    # Fallback to legacy location
    legacy_paths = [
        Path("configs") / config_type / f"{config_name}.yaml",
        Path("mlbench/configs") / config_type / f"{config_name}.yaml",
        Path("src/mlbench/configs") / config_type / f"{config_name}.yaml",
    ]
    
    for path in legacy_paths:
        if path.exists():
            return path
    
    return None


def list_available_configs(config_type: str = "defaults") -> list[str]:
    """List available configuration files.
    
    Args:
        config_type: Type of config (defaults, model, data, train, etc.)
        
    Returns:
        List of available config names
    """
    configs = []
    
    # Try new location first
    try:
        config_dir = files("mlbench.configs").joinpath(config_type)
        if config_dir.exists():
            for file_path in config_dir.iterdir():
                if file_path.suffix == ".yaml":
                    configs.append(file_path.stem)
    except Exception:
        pass
    
    # Fallback to legacy location
    legacy_paths = [
        Path("configs") / config_type,
        Path("mlbench/configs") / config_type,
        Path("src/mlbench/configs") / config_type,
    ]
    
    for path in legacy_paths:
        if path.exists():
            for file_path in path.glob("*.yaml"):
                if file_path.stem not in configs:
                    configs.append(file_path.stem)
    
    return sorted(configs)

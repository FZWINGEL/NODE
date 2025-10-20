from __future__ import annotations

import subprocess
import os
from pathlib import Path
from typing import Optional


def get_git_sha() -> Optional[str]:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_package_version() -> str:
    """Get the package version."""
    try:
        from importlib.metadata import version
        return version("mlbench")
    except Exception:
        return "0.1.0"


def get_version_info() -> dict[str, str]:
    """Get comprehensive version information."""
    return {
        "package_version": get_package_version(),
        "git_sha": get_git_sha() or "unknown",
    }

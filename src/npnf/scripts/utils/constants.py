"""Shared constants for conditioning configuration analysis."""

from pathlib import Path

CONFIG_NAMES = {
    "noenv_nogeno": "Baseline",
    "env_nogeno": "Temperature",
    "noenv_geno": "Markers",
    "env_geno": "Full",
}

CONFIG_ORDER = ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]

METHODS = ["half", "no_context"]


def find_conditioning_folders(base_path: Path, method: str) -> dict[str, Path]:
    """Find conditioning configuration folders under base_path for a method."""
    configs = {}
    for config_name in CONFIG_ORDER:
        config_path = base_path / config_name / method
        if config_path.exists():
            configs[config_name] = config_path
    return configs

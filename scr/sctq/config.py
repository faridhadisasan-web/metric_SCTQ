import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from sctq.exceptions import ConfigError
from sctq.types import ExperimentConfig, RunConfig


class ConfigLoader:
    """Helper class to load and merge configuration files."""

    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        path = Path(file_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {path}: {e}")

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a JSON configuration file."""
        path = Path(file_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except json.JSONDecodeError as e:
            raise ConfigError(f"Error parsing JSON file {path}: {e}")

    @staticmethod
    def save_yaml(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save a dictionary to a YAML file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)

    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        merged = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = ConfigLoader.merge_configs(merged[k], v)
            else:
                merged[k] = v
        return merged



def resolve_config_path(default_cfg: Dict[str, Any], default_key: str, fallback_filename: str) -> Path:
    """Resolve a sub-config path from default.yaml.

    Priority:
    1. defaults.<default_key> if provided
    2. paths.configs_dir / <fallback_filename>

    Relative paths from defaults.* are first checked as-is; if they do not exist,
    they are resolved relative to paths.configs_dir.
    """
    paths_cfg = default_cfg.get("paths", {})
    configs_dir = Path(paths_cfg.get("configs_dir", "configs"))
    defaults_cfg = default_cfg.get("defaults", {})
    raw_path = defaults_cfg.get(default_key)

    if raw_path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate

        # Prefer resolving relative paths against configs_dir so an external
        # config bundle can redirect all sub-configs by changing paths.configs_dir.
        configs_candidate = configs_dir / (candidate.name if len(candidate.parts) > 1 else candidate)
        if configs_candidate.exists() or configs_dir != Path("configs"):
            return configs_candidate

        if candidate.exists():
            return candidate

    return configs_dir / fallback_filename


def get_outputs_dir(default_cfg: Dict[str, Any]) -> Path:
    """Return the configured output root directory."""
    return Path(default_cfg.get("paths", {}).get("outputs_dir", "data/outputs"))


def get_processed_dir(default_cfg: Dict[str, Any]) -> Path:
    """Return the configured processed-data directory."""
    return Path(default_cfg.get("paths", {}).get("processed_dir", "data/processed"))

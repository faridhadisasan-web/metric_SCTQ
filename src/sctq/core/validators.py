from typing import Any, Dict, List

from sctq.exceptions import ConfigError


class ConfigValidator:
    """Validates configuration dictionaries."""

    @staticmethod
    def validate_metrics_config(config: Dict[str, Any]) -> None:
        if "sctq_weights" not in config:
            raise ConfigError("Missing 'sctq_weights' in metrics config.")

        weights = config["sctq_weights"]
        required_weights = ["w_p", "w_d", "w_f", "w_c"]
        for w in required_weights:
            if w not in weights:
                raise ConfigError(f"Missing weight '{w}' in sctq_weights.")
            if weights[w] < 0:
                raise ConfigError(f"Weight '{w}' must be non-negative.")

        total_weight = sum(weights[w] for w in required_weights)
        if abs(total_weight - 1.0) > 1e-6:
            raise ConfigError("SCTQ component weights must sum to 1.0.")

    @staticmethod
    def validate_trackers_config(config: List[Dict[str, Any]]) -> None:
        if not isinstance(config, list):
            raise ConfigError("Trackers config must be a list.")

        valid_types = ["motrackers", "ultralytics", "fallback", "internal"]
        for t in config:
            if "name" not in t or "type" not in t:
                raise ConfigError(f"Tracker missing 'name' or 'type': {t}")
            if t["type"] not in valid_types:
                raise ConfigError(f"Unknown tracker type: {t['type']} for tracker {t['name']}")

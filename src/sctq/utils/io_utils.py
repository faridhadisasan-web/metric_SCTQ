import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union


def _json_safe(value: Any) -> Any:
    try:
        import dataclasses
        import numpy as np
    except Exception:  # pragma: no cover
        dataclasses = None
        np = None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.integer):
        return int(value)
    if np is not None and isinstance(value, np.floating):
        return float(value)
    if dataclasses is not None and dataclasses.is_dataclass(value):
        return {k: _json_safe(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return {str(k): _json_safe(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return str(value)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dir(path: Union[str, Path], clean: bool = False) -> Path:
    path = Path(path)
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from sctq.utils.io_utils import ensure_dir


class JSONReporter:
    """Saves complete structured records for reproducibility."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = ensure_dir(output_dir)
        self.per_track_dir = ensure_dir(self.output_dir / "per_track")
        self.per_run_dir = ensure_dir(self.output_dir / "per_run")
        self.per_tracker_dir = ensure_dir(self.output_dir / "per_tracker")
        self.validation_dir = ensure_dir(self.output_dir / "validation")

    def _dump(self, path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def save_per_track(self, filename: str, metrics: List[Dict[str, Any]]) -> None:
        if metrics:
            self._dump(self.per_track_dir / f"{filename}.json", metrics)

    def save_per_run(self, filename: str, summary: Dict[str, Any]) -> None:
        if summary:
            self._dump(self.per_run_dir / f"{filename}.json", summary)

    def save_aggregated(self, filename: str, summaries: List[Dict[str, Any]]) -> None:
        if summaries:
            self._dump(self.per_tracker_dir / f"{filename}.json", summaries)

    def save_validation(self, filename: str, summaries: List[Dict[str, Any]]) -> None:
        if summaries:
            self._dump(self.validation_dir / f"{filename}.json", summaries)

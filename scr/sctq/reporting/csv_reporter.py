from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from sctq.types import TrackedObject
from sctq.utils.io_utils import ensure_dir
from sctq.utils.tabular import write_csv_rows


class CSVReporter:
    """Saves results to structured CSV files without requiring pandas."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = ensure_dir(output_dir)
        self.per_track_dir = ensure_dir(self.output_dir / "per_track")
        self.per_run_dir = ensure_dir(self.output_dir / "per_run")
        self.per_tracker_dir = ensure_dir(self.output_dir / "per_tracker")
        self.validation_dir = ensure_dir(self.output_dir / "validation")
        self.raw_tracks_dir = ensure_dir(self.output_dir / "raw_tracks")

    def save_rows(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        write_csv_rows(path, list(rows))

    def save_per_track(self, filename: str, metrics: List[Dict[str, Any]]) -> None:
        if metrics:
            self.save_rows(self.per_track_dir / f"{filename}.csv", metrics)

    def save_per_run(self, filename: str, summary: Dict[str, Any]) -> None:
        if summary:
            self.save_rows(self.per_run_dir / f"{filename}.csv", [summary])

    def save_aggregated(self, filename: str, summaries: List[Dict[str, Any]]) -> None:
        if summaries:
            self.save_rows(self.per_tracker_dir / f"{filename}.csv", summaries)

    def save_validation(self, filename: str, summaries: List[Dict[str, Any]]) -> None:
        if summaries:
            self.save_rows(self.validation_dir / f"{filename}.csv", summaries)

    def save_raw_tracks(self, filename: str, tracks: List[TrackedObject]) -> None:
        if not tracks:
            return

        rows = []
        for track in tracks:
            for point in track.points:
                row = {
                    "track_id": track.track_id,
                    "frame_idx": point.frame_idx,
                    "cx": point.cx,
                    "cy": point.cy,
                    "w": point.w,
                    "h": point.h,
                    "conf": point.conf if point.conf is not None else "",
                    "class_id": point.class_id if point.class_id is not None else "",
                }
                rows.append(row)

        if rows:
            self.save_rows(self.raw_tracks_dir / f"{filename}.csv", rows)

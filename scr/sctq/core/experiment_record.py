import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """Record of an experiment run, including metadata and results."""

    experiment_id: str
    run_id: str
    tracker_name: str
    video_id: str
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # Results
    per_track_metrics: List[Dict[str, Any]] = field(default_factory=list)
    run_summary: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        """Mark the experiment as finished."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get total execution time."""
        return self.end_time - self.start_time

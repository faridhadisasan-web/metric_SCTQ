from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Generic Types
ArrayLike = Union[List[float], np.ndarray]
BBoxType = Tuple[float, float, float, float]  # (cx, cy, w, h)
PointType = Tuple[float, float]  # (x, y)


@dataclass
class DetectionRaw:
    """A single detection as parsed from a file or network output."""

    frame_idx: int
    det_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    class_id: Optional[int] = None
    visibility: Optional[float] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class TrackPoint:
    """A single point in a tracked object's trajectory."""

    frame_idx: int
    cx: float
    cy: float
    w: float
    h: float
    conf: Optional[float] = None
    class_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class TrackedObject:
    """A single object tracked over multiple frames."""

    track_id: int
    points: List[TrackPoint]

    @property
    def start_frame(self) -> int:
        return self.points[0].frame_idx if self.points else -1

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame_idx if self.points else -1

    @property
    def length(self) -> int:
        return len(self.points)


@dataclass
class FrameDetections:
    """Detections for a single frame."""

    frame_idx: int
    detections: List[DetectionRaw]


@dataclass
class TrackSet:
    """A collection of tracked objects from a single run."""

    run_id: str
    tracker_name: str
    video_id: str
    tracks: Dict[int, TrackedObject]
    total_frames: int


@dataclass
class RunConfig:
    """Configuration for a single evaluation run."""

    run_id: str
    tracker_name: str
    video_path: Optional[str] = None
    detections_path: Optional[str] = None
    output_dir: str = ""
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment (multiple runs/trackers/videos)."""

    experiment_id: str
    trackers: List[str]
    videos: List[str]
    metrics_config: Dict[str, Any]
    output_dir: str
    num_runs: int = 1
    seeds: List[int] = None

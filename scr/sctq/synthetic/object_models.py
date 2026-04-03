from dataclasses import dataclass
from typing import List, Tuple

from sctq.types import TrackPoint


@dataclass
class SyntheticObject:
    """Represents an object in a synthetic scene."""

    object_id: int
    class_id: int
    trajectory: List[TrackPoint]

    @property
    def start_frame(self) -> int:
        return self.trajectory[0].frame_idx if self.trajectory else -1

    @property
    def end_frame(self) -> int:
        return self.trajectory[-1].frame_idx if self.trajectory else -1

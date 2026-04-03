from abc import ABC, abstractmethod
from typing import Any, Dict, List

from sctq.types import FrameDetections, TrackedObject


class BaseTrackerAdapter(ABC):
    """Abstract base class for all tracker adapters."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self._name = name
        self.params = params

    @property
    def name(self) -> str:
        """Name of the tracker."""
        return self._name

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state for a new run."""
        pass

    @abstractmethod
    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        """Update tracker state with new detections and return updated tracks."""
        pass

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from sctq.types import FrameDetections


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self._name = name
        self.params = params

    @property
    def name(self) -> str:
        """Name of the detector."""
        return self._name

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run detection on a single frame."""
        pass

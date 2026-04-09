from typing import Any, Dict, List

import cv2
import numpy as np

from sctq.corruptions.corruption_registry import CorruptionRegistry
from sctq.types import FrameDetections


class CorruptionRunner:
    """Runner for applying corruptions to data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def apply_image_corruptions(
        self, frame: np.ndarray, corruptions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Apply a sequence of image corruptions to a frame."""
        for c in corruptions:
            name = c.get("name")
            severity = c.get("severity", 1)
            func = CorruptionRegistry.get_image_corruption(name)
            if func:
                frame = func(frame, severity)
        return frame

    def apply_detection_corruptions(
        self, frame_dets: FrameDetections, corruptions: List[Dict[str, Any]]
    ) -> FrameDetections:
        """Apply a sequence of detection corruptions to a set of frame detections."""
        for c in corruptions:
            name = c.get("name")
            severity = c.get("severity", 1)
            func = CorruptionRegistry.get_detection_corruption(name)
            if func:
                frame_dets = func(frame_dets, severity)
        return frame_dets

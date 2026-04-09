from typing import List

from sctq.types import DetectionRaw, FrameDetections


class FrameDetectionsManager:
    """Manager for frame-level detections."""

    @staticmethod
    def create(frame_idx: int, detections: List[DetectionRaw]) -> FrameDetections:
        """Create a FrameDetections object."""
        return FrameDetections(frame_idx=frame_idx, detections=detections)

    @staticmethod
    def filter_by_confidence(fd: FrameDetections, conf_thresh: float) -> FrameDetections:
        """Return a new FrameDetections with low confidence detections filtered out."""
        filtered = [d for d in fd.detections if d.conf is None or d.conf >= conf_thresh]
        return FrameDetectionsManager.create(fd.frame_idx, filtered)

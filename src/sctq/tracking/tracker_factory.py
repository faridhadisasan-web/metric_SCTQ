from typing import Dict, Any, List
from sctq.types import FrameDetections, TrackedObject, TrackPoint
from sctq.tracking.tracker_base import BaseTrackerAdapter

class TrackerFactory:
    """Factory for creating tracker adapters."""

    @staticmethod
    def create(name: str, config: Dict[str, Any], strict_mode: bool = True) -> BaseTrackerAdapter:
        """Create a tracker adapter based on configuration."""
        tracker_type = config.get("type", "")
        params = config.get("params", {})

        if tracker_type == "internal":
            return TrackerFactory._create_internal(name, params)
        elif tracker_type == "ultralytics":
            return TrackerFactory._create_ultralytics(name, params, strict_mode)
        elif tracker_type == "fallback":
            if strict_mode:
                raise ValueError(f"Strict mode enabled: Cannot use fallback tracker for {name}")
            return TrackerFactory._create_fallback(name, params)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

    @staticmethod
    def _create_internal(name: str, params: Dict[str, Any]) -> BaseTrackerAdapter:
        """Create a custom internal tracker adapter."""
        if name == "sort":
            from sctq.tracking.internal_sort import InternalSORTTracker
            return InternalSORTTracker(name, params)
        elif name == "iou":
            from sctq.tracking.internal_iou import InternalIOUTracker
            return InternalIOUTracker(name, params)
        elif name == "centroid":
            from sctq.tracking.internal_centroid import InternalCentroidTracker
            return InternalCentroidTracker(name, params)
        elif name == "centroidkf":
            from sctq.tracking.internal_centroid import InternalCentroidKFTracker
            return InternalCentroidKFTracker(name, params)
        else:
            raise ValueError(f"Unknown internal tracker: {name}")

    @staticmethod
    def _create_ultralytics(name: str, params: Dict[str, Any], strict_mode: bool) -> BaseTrackerAdapter:
        """Create an ultralytics adapter."""
        from sctq.tracking.ultralytics_tracker_adapter import UltralyticsAdapter
        try:
            import ultralytics
            if name == "bytetrack":
                from ultralytics.trackers.byte_tracker import BYTETracker
                return UltralyticsAdapter(name, params, BYTETracker)
            elif name == "botsort":
                from ultralytics.trackers.bot_sort import BOTSORT
                return UltralyticsAdapter(name, params, BOTSORT)
            else:
                raise ValueError(f"Unknown ultralytics tracker: {name}")
        except ImportError as e:
            if strict_mode:
                raise ValueError(f"Strict mode enabled: Ultralytics installation required for {name}. Error: {e}")
            import logging
            logging.warning("ultralytics not installed or import failed, falling back if possible.")
            return TrackerFactory._create_fallback(name, params)

    @staticmethod
    def _create_fallback(name: str, params: Dict[str, Any]) -> BaseTrackerAdapter:
         """Create a minimal fallback tracker adapter."""
         class DummyAdapter(BaseTrackerAdapter):
             def reset(self):
                 pass
             def update(self, frame_detections):
                 # Dummy tracking: assume det_id is track_id
                 tracks = []
                 for det in frame_detections.detections:
                     track_id = det.det_id if det.det_id is not None else hash(str(det.cx) + str(det.cy)) % 10000
                     tp = TrackPoint(
                         frame_idx=frame_detections.frame_idx,
                         cx=det.cx, cy=det.cy, w=det.w, h=det.h,
                         conf=det.conf, class_id=det.class_id
                     )
                     tracks.append(TrackedObject(track_id=track_id, points=[tp]))
                 return tracks
         return DummyAdapter(name, params)

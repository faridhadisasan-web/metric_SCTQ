from typing import Any, Dict, List

from sctq.core.track_history import TrackHistory
from sctq.core.trackset import TrackSetManager
from sctq.metrics.sctq import SCTQEngine
from sctq.tracking.tracker_factory import TrackerFactory
from sctq.types import FrameDetections


class Pipeline:
    """Core evaluation pipeline."""

    def __init__(self, metrics_config: Dict[str, Any]):
        self.sctq_engine = SCTQEngine(metrics_config)

    def run_single(
        self,
        tracker_name: str,
        tracker_config: Dict[str, Any],
        detections: List[FrameDetections],
        run_id: str,
        video_id: str,
        total_frames: int,
    ) -> Dict[str, Any]:
        """Run a single tracker and return the resulting trackset and metrics."""
        try:
            tracker = TrackerFactory.create(tracker_name, tracker_config)
        except Exception as e:
            print(f"Failed to create tracker {tracker_name}: {e}")
            return None

        tracker.reset()
        track_history = TrackHistory()

        for fd in detections:
            updated_tracks = tracker.update(fd)
            track_history.update_from_tracker(updated_tracks)

        finalized_tracks = track_history.finalize()

        trackset = TrackSetManager.create_track_set(
            run_id=run_id,
            tracker_name=tracker_name,
            video_id=video_id,
            tracks=finalized_tracks,
            total_frames=total_frames,
        )

        sctq_summary = self.sctq_engine.compute_sctq_core(trackset)

        return {
            "trackset": trackset,
            "sctq_summary": sctq_summary,
            "run_summary": {
                "tracker_name": tracker_name,
                "sctq_core": sctq_summary["sctq_core"],
                "persistence_aggregate": sctq_summary["persistence_aggregate"],
                "dynamic_aggregate": sctq_summary["dynamic_aggregate"],
                "fragmentation_aggregate": sctq_summary["fragmentation_aggregate"],
                "consistency_aggregate": sctq_summary["consistency_aggregate"],
                "number_of_tracks": len(trackset.tracks),
            },
        }

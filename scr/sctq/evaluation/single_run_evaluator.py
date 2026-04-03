from typing import Dict, Any, List
from sctq.types import FrameDetections
from sctq.tracking.tracker_factory import TrackerFactory
from sctq.metrics.sctq import SCTQEngine
from sctq.core.trackset import TrackSetManager
from sctq.core.track_history import TrackHistory

class SingleRunEvaluator:
    """Evaluates a single tracker on a single sequence of detections."""

    def __init__(self, metrics_config: Dict[str, Any], strict_mode: bool = True):
        self.sctq_engine = SCTQEngine(metrics_config)
        self.strict_mode = strict_mode

    def evaluate(self, tracker_name: str, tracker_config: Dict[str, Any], detections: List[FrameDetections], run_id: str, video_id: str, total_frames: int) -> Dict[str, Any]:
        """Run tracker on detections and compute metrics."""

        try:
            tracker = TrackerFactory.create(tracker_name, tracker_config, self.strict_mode)
        except Exception as e:
            if self.strict_mode:
                raise RuntimeError(f"Strict Mode Abort: Failed to create tracker {tracker_name}: {e}")
            else:
                import logging
                logging.error(f"Failed to create tracker {tracker_name}: {e}")
                return {}

        tracker.reset()
        track_history = TrackHistory()

        for fd in detections:
            try:
                updated_tracks = tracker.update(fd)
                track_history.update_from_tracker(updated_tracks)
            except Exception as e:
                if self.strict_mode:
                    raise RuntimeError(f"Strict Mode Abort: Tracker {tracker_name} crashed during update: {e}")
                else:
                    import logging
                    logging.error(f"Tracker {tracker_name} crashed during update: {e}")
                    return {}

        finalized_tracks = track_history.finalize()

        trackset = TrackSetManager.create_track_set(
            run_id=run_id,
            tracker_name=tracker_name,
            video_id=video_id,
            tracks=finalized_tracks,
            total_frames=total_frames
        )

        sctq_summary = self.sctq_engine.compute_sctq_core(trackset)

        return {
            "trackset": trackset,
            "sctq_summary": sctq_summary,
            "run_summary": {
                "tracker_name": tracker_name,
                "sctq_core": sctq_summary['sctq_core'],
                "persistence_aggregate": sctq_summary['persistence_aggregate'],
                "dynamic_aggregate": sctq_summary['dynamic_aggregate'],
                "fragmentation_aggregate": sctq_summary['fragmentation_aggregate'],
                "consistency_aggregate": sctq_summary['consistency_aggregate'],
                "number_of_tracks": len(trackset.tracks)
            }
        }

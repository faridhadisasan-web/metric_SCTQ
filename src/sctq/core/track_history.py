from typing import Dict, List

from sctq.core.data_models import detection_to_trackpoint
from sctq.types import DetectionRaw, TrackedObject, TrackPoint


class TrackHistory:
    """Maintains the active track histories during a tracking run."""

    def __init__(self):
        self.active_tracks: Dict[int, List[TrackPoint]] = {}

    def update_from_tracker(self, frame_tracks: List[TrackedObject]) -> None:
        """Update track histories with tracked objects returned from a tracker."""
        for track in frame_tracks:
            track_id = track.track_id
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = []

            # Since trackers generally return the point for the current frame
            # we just append it
            for pt in track.points:
                self.active_tracks[track_id].append(pt)

    def finalize(self) -> Dict[int, TrackedObject]:
        """Finalize and return all tracks."""
        return {
            tid: TrackedObject(track_id=tid, points=pts) for tid, pts in self.active_tracks.items()
        }

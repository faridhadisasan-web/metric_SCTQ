from typing import Dict, List

from sctq.types import TrackedObject, TrackSet


class TrackSetManager:
    """Helper for managing collections of tracked objects."""

    @staticmethod
    def create_track_set(
        run_id: str,
        tracker_name: str,
        video_id: str,
        tracks: Dict[int, TrackedObject],
        total_frames: int,
    ) -> TrackSet:
        """Create a track set from finalized tracks."""
        return TrackSet(
            run_id=run_id,
            tracker_name=tracker_name,
            video_id=video_id,
            tracks=tracks,
            total_frames=total_frames,
        )

    @staticmethod
    def get_track(ts: TrackSet, track_id: int) -> TrackedObject:
        """Retrieve a specific track from a track set."""
        return ts.tracks.get(track_id)

    @staticmethod
    def get_all_tracks(ts: TrackSet) -> List[TrackedObject]:
        """Get all tracks as a list."""
        return list(ts.tracks.values())

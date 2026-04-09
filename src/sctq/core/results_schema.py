from typing import Any, Dict, List


class ResultsSchema:
    """Schema helper for structured output data."""

    @staticmethod
    def get_per_track_schema(
        experiment_id: str,
        run_id: str,
        tracker_name: str,
        video_id: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
        length: int,
        relative_length: float,
        mean_speed: float,
        std_speed: float,
        mean_turn: float,
        max_turn: float,
        mean_acceleration: float,
        bbox_size_std: float,
        persistence_score: float,
        dynamic_score: float,
        consistency_score: float,
        raw_point_count: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Schema for a single track's results."""
        schema = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "tracker_name": tracker_name,
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "length": length,
            "relative_length": relative_length,
            "mean_speed": mean_speed,
            "std_speed": std_speed,
            "mean_turn": mean_turn,
            "max_turn": max_turn,
            "mean_acceleration": mean_acceleration,
            "bbox_size_std": bbox_size_std,
            "persistence_score": persistence_score,
            "dynamic_score": dynamic_score,
            "consistency_score": consistency_score,
            "raw_point_count": raw_point_count,
            **kwargs,
        }
        return schema

    @staticmethod
    def get_per_run_schema(
        number_of_tracks: int,
        mean_track_length: float,
        median_track_length: float,
        std_track_length: float,
        short_track_count: int,
        medium_track_count: int,
        long_track_count: int,
        persistence_aggregate: float,
        dynamic_aggregate: float,
        fragmentation_aggregate: float,
        consistency_aggregate: float,
        sctq_core: float,
        sctq_final: float,
        corruption_type: str,
        corruption_severity: int,
        detector_info: Dict[str, Any],
        tracker_info: Dict[str, Any],
        runtime_stats: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Schema for run-level results."""
        schema = {
            "number_of_tracks": number_of_tracks,
            "mean_track_length": mean_track_length,
            "median_track_length": median_track_length,
            "std_track_length": std_track_length,
            "short_track_count": short_track_count,
            "medium_track_count": medium_track_count,
            "long_track_count": long_track_count,
            "persistence_aggregate": persistence_aggregate,
            "dynamic_aggregate": dynamic_aggregate,
            "fragmentation_aggregate": fragmentation_aggregate,
            "consistency_aggregate": consistency_aggregate,
            "sctq_core": sctq_core,
            "sctq_final": sctq_final,
            "corruption_type": corruption_type,
            "corruption_severity": corruption_severity,
            "detector_info": detector_info,
            "tracker_info": tracker_info,
            "runtime_stats": runtime_stats,
            **kwargs,
        }
        return schema

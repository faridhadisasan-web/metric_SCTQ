import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sctq.config import ConfigLoader, get_outputs_dir, get_processed_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.detection.detection_cache import DetectionCache
from sctq.detection.yolo_detector import YOLODetector
from sctq.metrics.ranking import compute_ranking
from sctq.metrics.sctq import SCTQEngine
from sctq.reporting.csv_reporter import CSVReporter
from sctq.reporting.json_reporter import JSONReporter
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.tracking.tracker_factory import TrackerFactory
from sctq.core.track_history import TrackHistory
from sctq.core.trackset import TrackSetManager
from sctq.utils.io_utils import prepare_output_dir, save_json
from sctq.utils.tabular import write_csv_rows
from sctq.utils.video_utils import read_video_frames
from sctq.visualization.metric_plots import PlottingManager


def _build_clean_cache_suffix(video_path: str, detector_cfg: Dict[str, Any], cache_suffix: str = "clean") -> str:
    """
    Build a cache suffix that is unique per clip / frame range.

    This prevents different clips of the same video from silently sharing
    the same clean detection cache.
    """
    clip_id = str(detector_cfg.get("clip_id", "") or "").strip()
    start_frame = detector_cfg.get("start_frame", None)
    end_frame = detector_cfg.get("end_frame", None)

    parts = [cache_suffix]
    if clip_id:
        parts.append(clip_id)
    if start_frame is not None:
        parts.append(f"s{int(start_frame)}")
    if end_frame is not None:
        parts.append(f"e{int(end_frame)}")
    return "_".join(parts)


def load_or_detect_video(
    video_path: str,
    detector_cfg: Dict[str, Any],
    default_cfg: Dict[str, Any],
    cache_suffix: str = "clean",
) -> Tuple[List[Any], int, bool]:
    """
    Load clean detections from cache or run detection on the selected clip.

    The cache key must depend on clip_id/start_frame/end_frame, otherwise
    multiple clips from the same video will incorrectly reuse one cache.
    """
    clean_suffix = _build_clean_cache_suffix(video_path, detector_cfg, cache_suffix=cache_suffix)
    video_id = f"{Path(video_path).stem}_{clean_suffix}"

    cache_dir = get_processed_dir(default_cfg) / "detections"
    cache = DetectionCache(cache_dir)
    detector_name = Path(detector_cfg.get("model", "yolo11n.pt")).stem

    cached = cache.load(video_id, detector_name, detector_cfg)
    if cached:
        return cached, len(cached), True

    detector = YOLODetector("yolo", detector_cfg)
    all_detections = []
    total_frames = 0

    max_frames = detector_cfg.get("max_frames")
    sample_mode = detector_cfg.get("frame_sampling", "head")
    frame_stride = int(detector_cfg.get("frame_stride", 1))
    start_frame = detector_cfg.get("start_frame", None)
    end_frame = detector_cfg.get("end_frame", None)

    for frame_idx, frame in read_video_frames(
        video_path,
        max_frames=max_frames,
        sample_mode=sample_mode,
        frame_stride=frame_stride,
        start_frame=start_frame,
        end_frame=end_frame,
    ):
        all_detections.append(detector.detect(frame, frame_idx))
        total_frames += 1

    cache.save(video_id, detector_name, all_detections, detector_cfg)
    return all_detections, total_frames, False


def run_real_video(
    config: Dict[str, Any],
    video_path: str,
    output_name: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    print(f"Running real video evaluation on {video_path}")

    strict_mode = config.get("default", {}).get("execution", {}).get("strict_mode", True)
    clean_outputs = bool(config.get("default", {}).get("execution", {}).get("clean_output_dirs", False))

    cfg_real = config.get("real_video", {})
    cfg_metrics = config.get("metrics", {})
    cfg_trackers = config.get("trackers", [])

    detector_cfg = dict(cfg_real.get("detector", {}))
    detector_cfg.setdefault("max_frames", cfg_real.get("max_frames", 50))
    detector_cfg.setdefault("frame_sampling", cfg_real.get("frame_sampling", "head"))
    detector_cfg.setdefault("frame_stride", cfg_real.get("frame_stride", 1))

    # IMPORTANT: propagate clip boundaries and identity into detector_cfg
    # so that detection caching and frame reading both use the clip selection.
    if cfg_real.get("start_frame") is not None:
        detector_cfg["start_frame"] = int(cfg_real.get("start_frame"))
    if cfg_real.get("end_frame") is not None:
        detector_cfg["end_frame"] = int(cfg_real.get("end_frame"))
    if cfg_real.get("clip_id") is not None:
        detector_cfg["clip_id"] = str(cfg_real.get("clip_id"))

    configured_output_dir = cfg_real.get("output_dir")
    output_base_root = Path(output_root) if output_root else (
        Path(configured_output_dir) if configured_output_dir else get_outputs_dir(config.get("default", {}))
    )
    output_base_dir = prepare_output_dir(output_base_root / (output_name or Path(video_path).stem), clean=clean_outputs)

    csv_reporter = CSVReporter(output_base_dir)
    json_reporter = JSONReporter(output_base_dir)
    plotter = PlottingManager(output_base_dir)

    all_detections, total_frames, from_cache = load_or_detect_video(
        video_path,
        detector_cfg,
        config.get("default", {}),
        cache_suffix="clean",
    )

    print("Loaded detections from cache." if from_cache else "Detected clean frames and cached them.")

    sctq_engine = SCTQEngine(cfg_metrics)
    trackers_to_run = cfg_real.get("trackers_to_run", [])
    active_trackers = [t for t in cfg_trackers if t["name"] in trackers_to_run]

    clip_id = str(cfg_real.get("clip_id", "clip0"))
    scene_type = str(cfg_real.get("scene_type", Path(video_path).stem))
    start_frame = int(detector_cfg.get("start_frame", 0) or 0)
    end_frame = int(detector_cfg.get("end_frame", 0) or 0)

    all_run_summaries: List[Dict[str, Any]] = []

    for tracker_cfg in active_trackers:
        t_name = tracker_cfg["name"]
        t_type = tracker_cfg.get("type", "unknown")
        print(f"Running tracker: {t_name} ({t_type})")

        tracker = TrackerFactory.create(t_name, tracker_cfg, strict_mode)
        tracker.reset()

        track_history = TrackHistory()
        for fd in all_detections:
            track_history.update_from_tracker(tracker.update(fd))

        finalized_tracks = track_history.finalize()
        trackset = TrackSetManager.create_track_set(
            run_id=f"real_{Path(video_path).stem}_{clip_id}_{t_name}",
            tracker_name=t_name,
            video_id=Path(video_path).stem,
            tracks=finalized_tracks,
            total_frames=total_frames,
        )

        sctq_summary = sctq_engine.compute_sctq_core(trackset)

        run_summary = {
            "tracker_name": t_name,
            "tracker_backend": t_type,
            "video_id": Path(video_path).stem,
            "clip_id": clip_id,
            "scene_type": scene_type,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "sctq_core": sctq_summary["sctq_core"],
            "persistence_aggregate": sctq_summary["persistence_aggregate"],
            "dynamic_aggregate": sctq_summary["dynamic_aggregate"],
            "fragmentation_aggregate": sctq_summary["fragmentation_aggregate"],
            "consistency_aggregate": sctq_summary["consistency_aggregate"],
            "consistency_effective": sctq_summary["consistency_effective"],
            "number_of_tracks": len(trackset.tracks),
            "num_frames_evaluated": total_frames,
        }

        all_run_summaries.append(run_summary)
        csv_reporter.save_per_run(f"real_{Path(video_path).stem}_{clip_id}_{t_name}", run_summary)
        json_reporter.save_per_run(f"real_{Path(video_path).stem}_{clip_id}_{t_name}", run_summary)
        csv_reporter.save_per_track(f"{clip_id}_{t_name}_per_track", sctq_summary["per_track_metrics"])
        json_reporter.save_per_track(f"{clip_id}_{t_name}_per_track", sctq_summary["per_track_metrics"])
        csv_reporter.save_raw_tracks(f"{clip_id}_{t_name}_raw_tracks", list(trackset.tracks.values()))

    ranked = compute_ranking(all_run_summaries, "sctq_core")
    csv_reporter.save_aggregated("real_video_summary", ranked)
    json_reporter.save_aggregated("real_video_summary", ranked)
    write_csv_rows(output_base_dir / "all_runs_flat.csv", all_run_summaries)

    plotter.plot_ranking_chart(ranked, "sctq_core", "ranking_chart", f"Real Clean Ranking: {Path(video_path).stem} ({clip_id})")
    plotter.plot_component_comparison(ranked, "component_comparison", f"Real Clean Components: {Path(video_path).stem} ({clip_id})")

    save_json(config, output_base_dir / "config_snapshot.json")

    md_reporter = MarkdownReporter(output_base_dir / "reports")
    md_reporter.generate_report(
        f"Real Video Benchmark: {Path(video_path).stem} ({clip_id})",
        ranked,
        "real_video_report.md",
        narrative=[
            f"This clean benchmark evaluates clip {clip_id} from frames {start_frame} to {end_frame}.",
            "All trackers use the same clean detections for the selected clip.",
        ],
    )

    print(json.dumps(ranked, indent=2))
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Run Real Video Evaluation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to default config.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of sampled frames.")
    parser.add_argument("--start-frame", type=int, default=None, help="Optional inclusive start frame.")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional inclusive end frame.")
    parser.add_argument("--clip-id", type=str, default="clip0", help="Optional clip identifier.")
    parser.add_argument("--scene-type", type=str, default=None, help="Optional scene label.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} does not exist.")
        sys.exit(1)

    config = ConfigLoader.load_yaml(args.config)
    real_config = ConfigLoader.load_yaml(resolve_config_path(config, "real_video_config", "real_video.yaml"))
    trk_config = ConfigLoader.load_yaml(resolve_config_path(config, "trackers_config", "trackers.yaml"))
    met_config = ConfigLoader.load_yaml(resolve_config_path(config, "metrics_config", "metrics.yaml"))

    if args.max_frames is not None:
        real_config.setdefault("real_video", {}).setdefault("detector", {})["max_frames"] = args.max_frames
        real_config["real_video"]["max_frames"] = args.max_frames
    if args.start_frame is not None:
        real_config.setdefault("real_video", {})["start_frame"] = args.start_frame
    if args.end_frame is not None:
        real_config.setdefault("real_video", {})["end_frame"] = args.end_frame
    if args.clip_id is not None:
        real_config.setdefault("real_video", {})["clip_id"] = args.clip_id
    if args.scene_type is not None:
        real_config.setdefault("real_video", {})["scene_type"] = args.scene_type

    full_config = {
        "default": config,
        "real_video": real_config.get("real_video", {}),
        "trackers": trk_config.get("trackers", []),
        "metrics": met_config.get("metrics", {}),
    }
    run_real_video(full_config, args.video)


if __name__ == "__main__":
    main()

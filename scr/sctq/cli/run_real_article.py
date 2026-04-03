import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sctq.cli.run_corruption_suite import run_corruption_benchmark
from sctq.cli.run_real_video import run_real_video
from sctq.config import ConfigLoader, get_outputs_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.reporting.csv_reporter import CSVReporter
from sctq.reporting.json_reporter import JSONReporter
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.utils.io_utils import prepare_output_dir, save_json
from sctq.utils.tabular import aggregate_numeric, write_csv_rows
from sctq.utils.video_utils import get_video_frame_count
from sctq.visualization.metric_plots import PlottingManager


VIDEO_SPECS = [
    {"name": "sparse", "folder": "video1", "filename": "video1.mp4", "groundtruth": None},
    {"name": "medium", "folder": "video2", "filename": "video2.mp4", "groundtruth": "video2-groundtruth.top", "calibration": "video2-calibration.ci"},
    {"name": "crowded", "folder": "video3", "filename": "video3.mp4", "groundtruth": None},
]


def _discover_videos(dataset_root: Path) -> List[Dict[str, Any]]:
    entries = []
    for spec in VIDEO_SPECS:
        video_path = dataset_root / spec["folder"] / spec["filename"]
        if not video_path.exists():
            raise FileNotFoundError(f"Expected video not found: {video_path}")
        entry = {"scene_type": spec["name"], "video_path": video_path, "video_id": video_path.stem}
        if spec.get("groundtruth"):
            gt_path = dataset_root / spec["folder"] / spec["groundtruth"]
            if gt_path.exists():
                entry["groundtruth_path"] = gt_path
        if spec.get("calibration"):
            cal_path = dataset_root / spec["folder"] / spec["calibration"]
            if cal_path.exists():
                entry["calibration_path"] = cal_path
        entries.append(entry)
    return entries


def _clip_ranges(total_frames: int, clip_length: int, clips_per_video: int) -> List[Dict[str, int]]:
    if total_frames <= 0:
        return [{"clip_id": "clip0", "start_frame": 0, "end_frame": 0}]
    if clip_length <= 0 or clip_length >= total_frames or clips_per_video <= 1:
        return [{"clip_id": "clip0", "start_frame": 0, "end_frame": total_frames - 1}]
    max_start = max(0, total_frames - clip_length)
    starts = sorted({int(round(i * max_start / max(1, clips_per_video - 1))) for i in range(clips_per_video)})
    clips = []
    for idx, start in enumerate(starts):
        end = min(total_frames - 1, start + clip_length - 1)
        clips.append({"clip_id": f"clip{idx}", "start_frame": start, "end_frame": end})
    return clips


def _attach_context(rows: Iterable[Dict[str, Any]], **context) -> List[Dict[str, Any]]:
    return [{**row, **context} for row in rows]


def _sort_rows(rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=reverse)


def _aggregate_by_tracker(rows: List[Dict[str, Any]], *, excluded: Iterable[str]) -> List[Dict[str, Any]]:
    aggregated = aggregate_numeric(rows, "tracker_name", excluded=set(excluded))
    sort_key = "sctq_final" if any("sctq_final" in row for row in aggregated) else "sctq_core"
    return _sort_rows(aggregated, sort_key, reverse=True)


def _aggregate_severity_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    for row in rows:
        prepared.append({**row, "group_key": f"{row.get('tracker_name')}||{row.get('corruption')}||{row.get('severity')}"})
    aggregated = aggregate_numeric(
        prepared,
        "group_key",
        excluded={
            "video_id",
            "video_path",
            "scene_type",
            "clip_id",
            "start_frame",
            "end_frame",
            "groundtruth_available",
            "tracker_backend",
            "corruption",
            "severity",
            "tracker_name",
        },
    )
    out = []
    for row in aggregated:
        tracker_name, corruption, severity = str(row["group_key"]).split("||")
        row["tracker_name"] = tracker_name
        row["corruption"] = corruption
        row["severity"] = int(float(severity))
        del row["group_key"]
        out.append(row)
    return sorted(out, key=lambda r: (str(r["corruption"]), int(r["severity"]), str(r["tracker_name"])))


def _aggregate_corruption_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    for row in rows:
        prepared.append({**row, "group_key": f"{row.get('tracker_name')}||{row.get('corruption')}"})
    aggregated = aggregate_numeric(
        prepared,
        "group_key",
        excluded={
            "video_id",
            "video_path",
            "scene_type",
            "clip_id",
            "start_frame",
            "end_frame",
            "groundtruth_available",
            "tracker_backend",
            "corruption",
            "severity",
            "tracker_name",
        },
    )
    out = []
    for row in aggregated:
        tracker_name, corruption = str(row["group_key"]).split("||")
        row["tracker_name"] = tracker_name
        row["corruption"] = corruption
        del row["group_key"]
        out.append(row)
    return sorted(out, key=lambda r: (str(r["corruption"]), str(r["tracker_name"])))


def _aggregate_video_rows(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    prepared = []
    for row in rows:
        prepared.append({**row, "group_key": f"{row.get('tracker_name')}||{row.get('video_id')}"})
    aggregated = aggregate_numeric(
        prepared,
        "group_key",
        excluded={"video_path", "scene_type", "clip_id", "start_frame", "end_frame", "groundtruth_available", "tracker_backend", "tracker_name", "video_id", "corruption", "severity"},
    )
    out: List[Dict[str, Any]] = []
    for row in aggregated:
        tracker_name, video_id = str(row["group_key"]).split("||")
        row["tracker_name"] = tracker_name
        row["video_id"] = video_id
        del row["group_key"]
        out.append(row)
    return _sort_rows(out, metric, reverse=True)


def run_article_protocol(config: Dict[str, Any], dataset_root: str, max_frames: int = 400, run_corruptions: bool = True, clips_per_video: int = 3) -> None:
    dataset_root_path = Path(dataset_root)
    videos = _discover_videos(dataset_root_path)
    default_cfg = config.get("default", {})
    output_root = prepare_output_dir(get_outputs_dir(default_cfg) / "real_article", clean=bool(default_cfg.get("execution", {}).get("clean_output_dirs", True)))
    csv_reporter = CSVReporter(output_root)
    json_reporter = JSONReporter(output_root)
    plotter = PlottingManager(output_root)

    clean_rows: List[Dict[str, Any]] = []
    corruption_flat_rows: List[Dict[str, Any]] = []
    robustness_severity_rows: List[Dict[str, Any]] = []
    robustness_corruption_rows: List[Dict[str, Any]] = []
    robustness_overall_rows: List[Dict[str, Any]] = []
    protocol_videos: List[Dict[str, Any]] = []

    cfg_real = dict(config.setdefault("real_video", {}))
    cfg_real.setdefault("detector", {})["max_frames"] = max_frames
    cfg_real["max_frames"] = max_frames

    for entry in videos:
        video_path = Path(entry["video_path"])
        total_frames = get_video_frame_count(str(video_path))
        clips = _clip_ranges(total_frames, max_frames, clips_per_video)
        protocol_videos.append(
            {
                "scene_type": entry["scene_type"],
                "video_id": entry["video_id"],
                "video_path": str(video_path),
                "total_frames": total_frames,
                "clips": clips,
                "groundtruth_available": "groundtruth_path" in entry,
            }
        )
        for clip in clips:
            clip_cfg = dict(cfg_real)
            clip_cfg["detector"] = dict(cfg_real.get("detector", {}))
            clip_cfg["max_frames"] = max_frames
            clip_cfg["start_frame"] = clip["start_frame"]
            clip_cfg["end_frame"] = clip["end_frame"]
            clip_cfg["clip_id"] = clip["clip_id"]
            clip_cfg["scene_type"] = entry["scene_type"]
            local_config = {**config, "real_video": clip_cfg}
            suffix = f"{entry['scene_type']}_{video_path.stem}_{clip['clip_id']}"
            context = {
                "scene_type": entry["scene_type"],
                "clip_id": clip["clip_id"],
                "video_id": entry["video_id"],
                "video_path": str(video_path),
                "groundtruth_available": ("groundtruth_path" in entry),
                "start_frame": clip["start_frame"],
                "end_frame": clip["end_frame"],
                "corruption": "clean",
                "corruption_type": "clean",
                "severity": 0,
                "run_index": 0,
            }
            clean_summary = run_real_video(local_config, str(video_path), output_name=f"clean_{suffix}", output_root=output_root / "per_video")
            clean_rows.extend(_attach_context(clean_summary, **context))
            if run_corruptions:
                details = run_corruption_benchmark(local_config, str(video_path), output_name=f"corrupted_{suffix}", output_root=output_root / "per_video", return_details=True)
                corruption_flat_rows.extend(_attach_context(details["all_runs_flat"], **{k: v for k, v in context.items() if k not in {"corruption", "corruption_type", "severity", "run_index"}}))
                robustness_severity_rows.extend(_attach_context(details["robustness_by_severity"], **{k: v for k, v in context.items() if k not in {"corruption", "corruption_type", "severity", "run_index"}}))
                robustness_corruption_rows.extend(_attach_context(details["robustness_by_corruption"], **{k: v for k, v in context.items() if k not in {"corruption", "corruption_type", "severity", "run_index"}}))
                robustness_overall_rows.extend(_attach_context(details["robustness_overall"], **{k: v for k, v in context.items() if k not in {"corruption", "corruption_type", "severity", "run_index"}}))

    clean_agg = _aggregate_by_tracker(clean_rows, excluded={"scene_type", "video_id", "video_path", "groundtruth_available", "tracker_backend", "clip_id", "start_frame", "end_frame", "corruption", "corruption_type", "severity", "run_index", "total_video_frames"})
    clean_by_video = _aggregate_video_rows(clean_rows, "sctq_core")
    csv_reporter.save_aggregated("clean_summary_across_videos", clean_agg)
    json_reporter.save_aggregated("clean_summary_across_videos", clean_agg)
    write_csv_rows(output_root / "clean_all_runs_flat.csv", clean_rows)
    write_csv_rows(output_root / "clean_by_video.csv", clean_by_video)
    plotter.plot_ranking_chart(clean_agg, "sctq_core", "clean_ranking_chart", "Real Clean Benchmark Across Videos and Clips")
    plotter.plot_component_comparison(clean_agg, "clean_component_comparison", "Real Clean Components Across Videos and Clips")

    narrative = [
        f"This article protocol uses three real pedestrian videos (sparse, medium, crowded) and evaluates {clips_per_video} contiguous clips per video.",
        f"Each clip contains up to {max_frames} frames and all trackers see the same clip boundaries and clean detections.",
    ]
    extra_sections = [
        {
            "title": "Clean Benchmark Outputs",
            "body": "The folder contains clean_all_runs_flat.csv with per-clip clean outputs and clean_by_video.csv with tracker means aggregated within each video before the cross-video summary.",
        }
    ]

    if corruption_flat_rows:
        severity_agg = _aggregate_severity_rows(robustness_severity_rows)
        corruption_agg = _aggregate_corruption_rows(robustness_corruption_rows)
        overall_agg = _aggregate_by_tracker(robustness_overall_rows, excluded={"scene_type", "video_id", "video_path", "groundtruth_available", "tracker_backend", "clip_id", "start_frame", "end_frame", "corruption"})
        corruption_by_video = _aggregate_video_rows(robustness_overall_rows, "sctq_final")

        csv_reporter.save_aggregated("corruption_summary_across_videos", overall_agg)
        json_reporter.save_aggregated("corruption_summary_across_videos", overall_agg)
        write_csv_rows(output_root / "corruption_all_runs_flat.csv", corruption_flat_rows)
        write_csv_rows(output_root / "robustness_by_severity.csv", severity_agg)
        write_csv_rows(output_root / "robustness_by_corruption.csv", corruption_agg)
        write_csv_rows(output_root / "robustness_overall.csv", overall_agg)
        write_csv_rows(output_root / "robustness_by_video.csv", corruption_by_video)
        plotter.plot_ranking_chart(overall_agg, "sctq_final", "corruption_ranking_chart", "Real Corruption Benchmark Across Videos and Clips")
        for corruption_name in sorted({row["corruption"] for row in severity_agg}):
            subset = [row for row in severity_agg if row["corruption"] == corruption_name]
            plotter.plot_robustness_curves(subset, "sctq_core_mean_drop", f"degradation_sctq_{corruption_name}", f"SCTQ Degradation: {corruption_name}")
            plotter.plot_robustness_curves(subset, "persistence_mean_drop", f"degradation_persistence_{corruption_name}", f"Persistence Degradation: {corruption_name}")
            plotter.plot_robustness_curves(subset, "dynamics_mean_drop", f"degradation_dynamics_{corruption_name}", f"Dynamics Degradation: {corruption_name}")
            plotter.plot_robustness_curves(subset, "fragmentation_mean_drop", f"degradation_fragmentation_{corruption_name}", f"Fragmentation Degradation: {corruption_name}")
            plotter.plot_robustness_curves(subset, "consistency_mean_drop", f"degradation_consistency_{corruption_name}", f"Consistency Degradation: {corruption_name}")
        extra_sections.append(
            {
                "title": "Robustness Outputs",
                "body": "The multi-video corruption benchmark preserves per-run rows in corruption_all_runs_flat.csv and exports aggregated summaries in robustness_by_severity.csv, robustness_by_corruption.csv, robustness_by_video.csv, and robustness_overall.csv.",
            }
        )

    save_json({"dataset_root": str(dataset_root_path), "videos": protocol_videos, "max_frames": max_frames, "clips_per_video": clips_per_video}, output_root / "protocol_snapshot.json")
    md = MarkdownReporter(output_root / "reports")
    md.generate_report("Real Article Benchmark", clean_agg, "real_article_report.md", narrative=narrative, extra_sections=extra_sections)


def main():
    parser = argparse.ArgumentParser(description="Run the final three-video real benchmark protocol")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset-root", type=str, required=True, help="Directory containing video1/, video2/, video3/")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--clips-per-video", type=int, default=3)
    parser.add_argument("--skip-corruptions", action="store_true")
    args = parser.parse_args()
    config = ConfigLoader.load_yaml(args.config)
    real_config = ConfigLoader.load_yaml(resolve_config_path(config, "real_video_config", "real_video.yaml"))
    trk_config = ConfigLoader.load_yaml(resolve_config_path(config, "trackers_config", "trackers.yaml"))
    met_config = ConfigLoader.load_yaml(resolve_config_path(config, "metrics_config", "metrics.yaml"))
    cor_config = ConfigLoader.load_yaml(resolve_config_path(config, "corruptions_config", "corruptions.yaml"))
    full_config = {
        "default": config,
        "real_video": real_config.get("real_video", {}),
        "trackers": trk_config.get("trackers", []),
        "metrics": met_config.get("metrics", {}),
        "corruptions": cor_config,
    }
    run_article_protocol(full_config, args.dataset_root, max_frames=args.max_frames, run_corruptions=not args.skip_corruptions, clips_per_video=args.clips_per_video)


if __name__ == "__main__":
    main()

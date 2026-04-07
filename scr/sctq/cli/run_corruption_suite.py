import argparse
import json
import os
import sys
from copy import deepcopy
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sctq.config import ConfigLoader, get_outputs_dir, get_processed_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.core.track_history import TrackHistory
from sctq.core.trackset import TrackSetManager
from sctq.corruptions.corruption_runner import CorruptionRunner
from sctq.detection.detection_cache import DetectionCache
from sctq.detection.yolo_detector import YOLODetector
from sctq.metrics.ranking import compute_ranking
from sctq.metrics.sctq import SCTQEngine
from sctq.metrics.stability import compute_run_summary_vector, compute_sctq_final, compute_stability_score
from sctq.reporting.csv_reporter import CSVReporter
from sctq.reporting.json_reporter import JSONReporter
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.tracking.tracker_factory import TrackerFactory
from sctq.utils.io_utils import prepare_output_dir, save_json
from sctq.utils.random_utils import set_seed
from sctq.utils.tabular import aggregate_numeric, linear_regression_slope, trapezoid_area, write_csv_rows
from sctq.utils.video_utils import read_video_frames
from sctq.visualization.metric_plots import PlottingManager


def normalize_corruption_config(root_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize both supported config shapes:

    A) {
         "corruptions": {
             "image_level": [...],
             "detection_level": [...]
         },
         "runner": {...}
       }

    B) {
         "image_level": [...],
         "detection_level": [...],
         "runner": {...}
       }

    The previous implementation lost the top-level runner block when the
    nested "corruptions" block existed, which silently reduced
    runs_per_severity to 1.
    """
    if not isinstance(root_config, dict):
        return {"image_level": [], "detection_level": [], "runner": {}}

    nested = root_config.get("corruptions", None)
    if isinstance(nested, dict):
        image_level = nested.get("image_level", root_config.get("image_level", []))
        detection_level = nested.get("detection_level", root_config.get("detection_level", []))
        runner_cfg = root_config.get("runner", nested.get("runner", {}))
    else:
        image_level = root_config.get("image_level", [])
        detection_level = root_config.get("detection_level", [])
        runner_cfg = root_config.get("runner", {})

    return {
        "image_level": [dict(x) for x in image_level],
        "detection_level": [dict(x) for x in detection_level],
        "runner": dict(runner_cfg or {}),
    }


def try_remove_bad_cache(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return False
    except (JSONDecodeError, OSError, ValueError):
        if path.exists():
            path.unlink()
        return True


def validate_corruption_outputs(all_runs_flat: List[Dict[str, Any]], normalized_cfg: Dict[str, Any]) -> None:
    noisy_rows = [row for row in all_runs_flat if row.get("corruption_type") in {"image", "detection"}]
    if not noisy_rows:
        raise RuntimeError("No corrupted runs were produced. Refusing to compute stability from clean-only output.")

    names = {row.get("corruption") for row in noisy_rows if row.get("corruption")}
    if not names:
        raise RuntimeError("No corruption names were found in the per-run outputs.")

    configured_severities = set()
    configured_expected_runs = 0
    for key in ("image_level", "detection_level"):
        for cfg in normalized_cfg.get(key, []):
            if cfg.get("enabled", True):
                configured_severities.update(cfg.get("severities", []))
                configured_expected_runs = max(configured_expected_runs, int(cfg.get("runs_per_severity", 1)))

    seen_severities = {int(row.get("severity", 0)) for row in noisy_rows if row.get("severity") is not None}
    if len(configured_severities) >= 2 and len(seen_severities) < 2:
        raise RuntimeError("Configured multiple severities but did not observe at least two in the corruption outputs.")

    if configured_expected_runs >= 2:
        seen_runs = {int(row.get("run_index", 0)) for row in noisy_rows}
        if len(seen_runs) < 2:
            raise RuntimeError(
                "The corruption config requested repeated runs per severity, "
                "but the outputs contain only one run index. "
                "This usually means the runner configuration was not applied correctly."
            )


def _run_tracker_on_detections(
    tracker_cfg: Dict[str, Any],
    detections,
    strict_mode: bool,
    video_id: str,
    total_frames: int,
    metrics_cfg: Dict[str, Any],
    run_id: str,
):
    tracker = TrackerFactory.create(tracker_cfg["name"], tracker_cfg, strict_mode)
    tracker.reset()

    history = TrackHistory()
    for fd in detections:
        history.update_from_tracker(tracker.update(fd))

    trackset = TrackSetManager.create_track_set(
        run_id=run_id,
        tracker_name=tracker_cfg["name"],
        video_id=video_id,
        tracks=history.finalize(),
        total_frames=total_frames,
    )

    summary = SCTQEngine(metrics_cfg).compute_sctq_core(trackset)
    return trackset, summary


def _noise_seed(base_seed: int, corruption_name: str, severity: int, run_idx: int) -> int:
    return abs(hash((base_seed, corruption_name, severity, run_idx))) % (2**31 - 1)


def _sorted_rows(rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=reverse)


def _precompute_detection_runs(clean_detections, det_cfgs, runner: CorruptionRunner, base_seed: int):
    runs: List[Tuple[Dict[str, Any], List[Any]]] = []

    for corruption_cfg in det_cfgs:
        if not corruption_cfg.get("enabled", True):
            continue

        name = corruption_cfg["name"]
        runs_per_severity = int(corruption_cfg.get("runs_per_severity", 1))

        for severity in corruption_cfg.get("severities", [1]):
            for run_idx in range(runs_per_severity):
                set_seed(_noise_seed(base_seed, name, severity, run_idx))
                print(f"[detection] corruption={name} severity={severity} run={run_idx}")
                noisy = [
                    runner.apply_detection_corruptions(fd, [{"name": name, "severity": severity}])
                    for fd in clean_detections
                ]
                runs.append(
                    (
                        {
                            "corruption": name,
                            "corruption_type": "detection",
                            "severity": severity,
                            "run_index": run_idx,
                        },
                        noisy,
                    )
                )
    return runs


def _precompute_image_runs(
    video_path: str,
    sampled_frames,
    detector_cfg: Dict[str, Any],
    img_cfgs,
    runner: CorruptionRunner,
    default_cfg: Dict[str, Any],
    base_seed: int,
    clip_tag: str = "",
):
    runs: List[Tuple[Dict[str, Any], List[Any]]] = []

    if not any(cfg.get("enabled", True) for cfg in img_cfgs):
        return runs

    print("Initializing YOLO detector for image corruptions...")
    cache = DetectionCache(get_processed_dir(default_cfg) / "detections")
    detector_name = Path(detector_cfg.get("model", "yolo11n.pt")).stem
    noisy_detector = YOLODetector("yolo", detector_cfg)

    for corruption_cfg in img_cfgs:
        if not corruption_cfg.get("enabled", True):
            continue

        name = corruption_cfg["name"]
        runs_per_severity = int(corruption_cfg.get("runs_per_severity", 1))

        for severity in corruption_cfg.get("severities", [1]):
            for run_idx in range(runs_per_severity):
                print(f"[image] corruption={name} severity={severity} run={run_idx}")

                cache_cfg = dict(detector_cfg)
                cache_cfg.update(
                    {
                        "corruption": name,
                        "severity": severity,
                        "run_index": run_idx,
                        "max_frames": detector_cfg.get("max_frames"),
                        "frame_sampling": detector_cfg.get("frame_sampling"),
                        "frame_stride": detector_cfg.get("frame_stride"),
                        "start_frame": detector_cfg.get("start_frame"),
                        "end_frame": detector_cfg.get("end_frame"),
                        "clip_id": detector_cfg.get("clip_id"),
                    }
                )

                video_id = (
                    f"{Path(video_path).stem}_{clip_tag}_img_{name}_s{severity}_r{run_idx}"
                    if clip_tag
                    else f"{Path(video_path).stem}_img_{name}_s{severity}_r{run_idx}"
                )

                try:
                    cached = cache.load(video_id, detector_name, cache_cfg)
                except Exception:
                    path = cache._get_cache_path(video_id, detector_name, cache_cfg)
                    try_remove_bad_cache(path)
                    cached = None

                if cached:
                    noisy_dets = cached
                else:
                    set_seed(_noise_seed(base_seed, name, severity, run_idx))
                    noisy_dets = []
                    for frame_idx, frame in sampled_frames:
                        corrupted = runner.apply_image_corruptions(frame.copy(), [{"name": name, "severity": severity}])
                        noisy_dets.append(noisy_detector.detect(corrupted, frame_idx))
                    cache.save(video_id, detector_name, noisy_dets, cache_cfg)

                runs.append(
                    (
                        {
                            "corruption": name,
                            "corruption_type": "image",
                            "severity": severity,
                            "run_index": run_idx,
                        },
                        noisy_dets,
                    )
                )

    return runs


METRIC_PAIRS = [
    ("sctq_core", "sctq_core"),
    ("persistence_aggregate", "persistence"),
    ("dynamic_aggregate", "dynamics"),
    ("fragmentation_aggregate", "fragmentation"),
    ("consistency_aggregate", "consistency"),
]


def _summarize_severity_curves(all_runs_flat: List[Dict[str, Any]], clean_by_tracker: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}

    for row in all_runs_flat:
        if row.get("corruption_type") in {"image", "detection"}:
            grouped.setdefault((row["tracker_name"], row["corruption"], int(row.get("severity", 0))), []).append(row)

    out: List[Dict[str, Any]] = []
    for (tracker_name, corruption, severity), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        summary = {"tracker_name": tracker_name, "corruption": corruption, "severity": severity}
        clean = clean_by_tracker[tracker_name]

        for clean_key, out_prefix in METRIC_PAIRS:
            vals = [float(r.get(clean_key, 0.0) or 0.0) for r in rows]
            mean_noisy = sum(vals) / max(1, len(vals)) if vals else 0.0
            clean_value = float(clean.get(clean_key, 0.0) or 0.0)

            summary[f"{out_prefix}_clean"] = clean_value
            summary[f"{out_prefix}_mean_noisy"] = mean_noisy
            summary[f"{out_prefix}_mean_drop"] = max(0.0, clean_value - mean_noisy)

        out.append(summary)

    return out


def _summarize_robustness(all_runs_flat: List[Dict[str, Any]], clean_by_tracker: Dict[str, Dict[str, float]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in all_runs_flat:
        if row.get("corruption_type") in {"image", "detection"}:
            grouped.setdefault((row["tracker_name"], row["corruption"]), []).append(row)

    per_corruption: List[Dict[str, Any]] = []
    for (tracker_name, corruption), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: (int(r.get("severity", 0)), int(r.get("run_index", 0))))
        summary = {"tracker_name": tracker_name, "corruption": corruption}
        clean = clean_by_tracker[tracker_name]
        severities = sorted({int(r.get("severity", 0)) for r in rows_sorted})

        for clean_key, out_prefix in METRIC_PAIRS:
            mean_drop_by_severity = []
            for severity in severities:
                sev_rows = [r for r in rows_sorted if int(r.get("severity", 0)) == severity]
                sev_values = [float(r.get(clean_key, 0.0) or 0.0) for r in sev_rows]
                noisy_mean = sum(sev_values) / max(1, len(sev_values)) if sev_values else 0.0
                mean_drop_by_severity.append(max(0.0, float(clean.get(clean_key, 0.0) or 0.0) - noisy_mean))

            summary[f"{out_prefix}_mean_drop"] = sum(mean_drop_by_severity) / max(1, len(mean_drop_by_severity)) if mean_drop_by_severity else 0.0
            summary[f"{out_prefix}_slope"] = linear_regression_slope([float(s) for s in severities], mean_drop_by_severity) if len(severities) >= 2 else 0.0
            summary[f"{out_prefix}_audc"] = trapezoid_area([float(s) for s in severities], mean_drop_by_severity) if len(severities) >= 2 else 0.0

        per_corruption.append(summary)

    overall = aggregate_numeric(per_corruption, "tracker_name", excluded={"corruption"})
    overall = _sorted_rows(overall, "sctq_core_mean_drop", reverse=False)
    return per_corruption, overall


def run_corruption_benchmark(
    config: Dict[str, Any],
    video_path: str,
    output_name: Optional[str] = None,
    output_root: Optional[Path] = None,
    return_details: bool = False,
):
    print(f"Running corruption benchmark on {video_path}")

    strict_mode = config.get("default", {}).get("execution", {}).get("strict_mode", True)
    clean_outputs = bool(config.get("default", {}).get("execution", {}).get("clean_output_dirs", False))

    cfg_real = config.get("real_video", {})
    cfg_metrics = config.get("metrics", {})
    cfg_trackers = config.get("trackers", [])

    corruption_cfg = normalize_corruption_config(config.get("corruptions", {}))

    detector_cfg = dict(cfg_real.get("detector", {}))
    detector_cfg.setdefault("max_frames", cfg_real.get("max_frames", 50))
    detector_cfg.setdefault("frame_sampling", cfg_real.get("frame_sampling", "head"))
    detector_cfg.setdefault("frame_stride", cfg_real.get("frame_stride", 1))

    if cfg_real.get("start_frame") is not None:
        detector_cfg["start_frame"] = int(cfg_real.get("start_frame"))
    if cfg_real.get("end_frame") is not None:
        detector_cfg["end_frame"] = int(cfg_real.get("end_frame"))
    if cfg_real.get("clip_id") is not None:
        detector_cfg["clip_id"] = str(cfg_real.get("clip_id"))

    clip_tag = str(cfg_real.get("clip_id", "")).strip()

    runner_cfg = dict(corruption_cfg.get("runner", {}))
    runs_per_severity = int(runner_cfg.get("runs_per_severity", 1))
    base_seed = int(runner_cfg.get("seed", 42))

    # Propagate the global runner repetition count into each corruption config.
    for c in corruption_cfg.get("image_level", []):
        c["runs_per_severity"] = int(c.get("runs_per_severity", runs_per_severity))
    for c in corruption_cfg.get("detection_level", []):
        c["runs_per_severity"] = int(c.get("runs_per_severity", runs_per_severity))

    configured_output_dir = cfg_real.get("output_dir")
    output_base_root = Path(output_root) if output_root else (
        Path(configured_output_dir) if configured_output_dir else get_outputs_dir(config.get("default", {}))
    )
    output_base_dir = prepare_output_dir(output_base_root / (output_name or f"{Path(video_path).stem}_corrupted"), clean=clean_outputs)

    csv_reporter = CSVReporter(output_base_dir)
    json_reporter = JSONReporter(output_base_dir)
    plotter = PlottingManager(output_base_dir)

    print("Running YOLO detector on clean video...")
    cache = DetectionCache(get_processed_dir(config.get("default", {})) / "detections")
    detector_name = Path(detector_cfg.get("model", "yolo11n.pt")).stem
    clean_video_id = f"{Path(video_path).stem}_{clip_tag}_clean" if clip_tag else f"{Path(video_path).stem}_clean"

    clean_detections = cache.load(clean_video_id, detector_name, detector_cfg)

    sampled_frames = list(
        read_video_frames(
            video_path,
            max_frames=detector_cfg.get("max_frames"),
            sample_mode=detector_cfg.get("frame_sampling", "head"),
            frame_stride=int(detector_cfg.get("frame_stride", 1)),
            start_frame=detector_cfg.get("start_frame"),
            end_frame=detector_cfg.get("end_frame"),
        )
    )

    if not clean_detections:
        detector = YOLODetector("yolo", detector_cfg)
        clean_detections = [detector.detect(frame, frame_idx) for frame_idx, frame in sampled_frames]
        cache.save(clean_video_id, detector_name, clean_detections, detector_cfg)

    total_frames = len(clean_detections)
    runner = CorruptionRunner(corruption_cfg)

    detection_runs = _precompute_detection_runs(
        clean_detections,
        corruption_cfg.get("detection_level", []),
        runner,
        base_seed,
    )

    image_runs = _precompute_image_runs(
        video_path,
        sampled_frames,
        detector_cfg,
        corruption_cfg.get("image_level", []),
        runner,
        config.get("default", {}),
        base_seed,
        clip_tag=clip_tag,
    )

    noisy_runs = detection_runs + image_runs

    all_run_summaries: List[Dict[str, Any]] = []
    all_runs_flat: List[Dict[str, Any]] = []

    trackers_to_run = cfg_real.get("trackers_to_run", [])
    active_trackers = [t for t in cfg_trackers if t["name"] in trackers_to_run]

    clean_vectors: Dict[str, Any] = {}
    clean_by_tracker: Dict[str, Dict[str, float]] = {}

    lambda_stab = float(cfg_metrics.get("sctq_weights", {}).get("lambda_stab", 0.20))
    clip_id = cfg_real.get("clip_id", "clip0")
    scene_type = cfg_real.get("scene_type", Path(video_path).stem)
    start_frame = int(detector_cfg.get("start_frame", 0) or 0)
    end_frame = int(detector_cfg.get("end_frame", (sampled_frames[-1][0] if sampled_frames else 0)) or 0)

    for tracker_cfg in active_trackers:
        t_name = tracker_cfg["name"]
        t_type = tracker_cfg.get("type", "unknown")
        print(f"--- Testing Tracker: {t_name} ({t_type}) ---")

        clean_trackset, clean_summary = _run_tracker_on_detections(
            tracker_cfg,
            clean_detections,
            strict_mode,
            Path(video_path).stem,
            total_frames,
            cfg_metrics,
            f"clean_{Path(video_path).stem}_{clip_id}_{t_name}",
        )

        clean_vectors[t_name] = compute_run_summary_vector(clean_summary, clean_trackset)
        clean_by_tracker[t_name] = clean_summary
        clean_final = compute_sctq_final(clean_summary["sctq_core"], 1.0, lambda_stab=lambda_stab)

        clean_row = {
            "tracker_name": t_name,
            "tracker_backend": t_type,
            "video_id": Path(video_path).stem,
            "clip_id": clip_id,
            "scene_type": scene_type,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "corruption": "clean",
            "corruption_type": "clean",
            "severity": 0,
            "run_index": 0,
            "sctq_core": clean_summary["sctq_core"],
            "persistence_aggregate": clean_summary["persistence_aggregate"],
            "dynamic_aggregate": clean_summary["dynamic_aggregate"],
            "fragmentation_aggregate": clean_summary["fragmentation_aggregate"],
            "consistency_aggregate": clean_summary["consistency_aggregate"],
            "consistency_effective": clean_summary["consistency_effective"],
            "stability_score": 1.0,
            "sctq_final": clean_final,
            "number_of_tracks": len(clean_trackset.tracks),
        }
        all_runs_flat.append(clean_row)
        csv_reporter.save_raw_tracks(f"{clip_id}_{t_name}_clean_raw_tracks", list(clean_trackset.tracks.values()))

        noisy_vectors = []
        for meta, noisy_dets in noisy_runs:
            noisy_trackset, summary_noisy = _run_tracker_on_detections(
                tracker_cfg,
                noisy_dets,
                strict_mode,
                Path(video_path).stem,
                total_frames,
                cfg_metrics,
                f"{meta['corruption_type']}_{meta['corruption']}_s{meta['severity']}_r{meta['run_index']}_{clip_id}_{t_name}",
            )

            vec_noisy = compute_run_summary_vector(summary_noisy, noisy_trackset)
            noisy_vectors.append(vec_noisy)

            run_output = {
                **clean_row,
                **meta,
                "sctq_core": summary_noisy["sctq_core"],
                "persistence_aggregate": summary_noisy["persistence_aggregate"],
                "dynamic_aggregate": summary_noisy["dynamic_aggregate"],
                "fragmentation_aggregate": summary_noisy["fragmentation_aggregate"],
                "consistency_aggregate": summary_noisy["consistency_aggregate"],
                "consistency_effective": summary_noisy["consistency_effective"],
                "sctq_core_drop": max(0.0, clean_summary["sctq_core"] - summary_noisy["sctq_core"]),
                "persistence_drop": max(0.0, clean_summary["persistence_aggregate"] - summary_noisy["persistence_aggregate"]),
                "dynamics_drop": max(0.0, clean_summary["dynamic_aggregate"] - summary_noisy["dynamic_aggregate"]),
                "fragmentation_drop": max(0.0, clean_summary["fragmentation_aggregate"] - summary_noisy["fragmentation_aggregate"]),
                "consistency_drop": max(0.0, clean_summary["consistency_aggregate"] - summary_noisy["consistency_aggregate"]),
                "number_of_tracks": len(noisy_trackset.tracks),
            }

            all_runs_flat.append(run_output)
            csv_reporter.save_per_run(
                f"{clip_id}_{t_name}_{meta['corruption_type']}_{meta['corruption']}_s{meta['severity']}_r{meta['run_index']}",
                run_output,
            )
            json_reporter.save_per_run(
                f"{clip_id}_{t_name}_{meta['corruption_type']}_{meta['corruption']}_s{meta['severity']}_r{meta['run_index']}",
                run_output,
            )
            csv_reporter.save_raw_tracks(
                f"{clip_id}_{t_name}_{meta['corruption_type']}_{meta['corruption']}_s{meta['severity']}_r{meta['run_index']}_raw_tracks",
                list(noisy_trackset.tracks.values()),
            )

        stability = compute_stability_score(clean_vectors[t_name], noisy_vectors)
        final_score = compute_sctq_final(clean_summary["sctq_core"], stability, lambda_stab=lambda_stab)

        tracker_summary = {
            "tracker_name": t_name,
            "tracker_backend": t_type,
            "video_id": Path(video_path).stem,
            "clip_id": clip_id,
            "scene_type": scene_type,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "sctq_core": clean_summary["sctq_core"],
            "persistence_aggregate": clean_summary["persistence_aggregate"],
            "dynamic_aggregate": clean_summary["dynamic_aggregate"],
            "fragmentation_aggregate": clean_summary["fragmentation_aggregate"],
            "consistency_aggregate": clean_summary["consistency_aggregate"],
            "consistency_effective": clean_summary["consistency_effective"],
            "stability_score": stability,
            "sctq_final": final_score,
            "number_of_clean_tracks": len(clean_trackset.tracks),
            "num_noisy_runs": len(noisy_runs),
        }
        all_run_summaries.append(tracker_summary)

    validate_corruption_outputs(all_runs_flat, corruption_cfg)

    severity_curves = _summarize_severity_curves(all_runs_flat, clean_by_tracker)
    per_corruption_summary, overall_robustness = _summarize_robustness(all_runs_flat, clean_by_tracker)

    ranked = compute_ranking(all_run_summaries, "sctq_final")

    csv_reporter.save_aggregated("corruption_summary", ranked)
    json_reporter.save_aggregated("corruption_summary", ranked)

    write_csv_rows(output_base_dir / "all_runs_flat.csv", all_runs_flat)
    write_csv_rows(output_base_dir / "robustness_by_severity.csv", severity_curves)
    write_csv_rows(output_base_dir / "robustness_by_corruption.csv", per_corruption_summary)
    write_csv_rows(output_base_dir / "robustness_overall.csv", overall_robustness)

    save_json(
        {
            "ranked_summary": ranked,
            "robustness_by_severity": severity_curves,
            "robustness_by_corruption": per_corruption_summary,
            "robustness_overall": overall_robustness,
        },
        output_base_dir / "corruption_summary.json",
    )

    plotter.plot_ranking_chart(ranked, "sctq_final", "ranking_chart", f"Corruption Ranking: {Path(video_path).stem} ({clip_id})")
    plotter.plot_component_comparison(ranked, "component_comparison", f"Clean Components: {Path(video_path).stem} ({clip_id})")

    for corruption_name in sorted({row["corruption"] for row in severity_curves}):
        subset = [row for row in severity_curves if row["corruption"] == corruption_name]
        plotter.plot_robustness_curves(subset, "sctq_core_mean_drop", f"degradation_sctq_{corruption_name}", f"SCTQ Degradation: {corruption_name}")
        plotter.plot_robustness_curves(subset, "persistence_mean_drop", f"degradation_persistence_{corruption_name}", f"Persistence Degradation: {corruption_name}")
        plotter.plot_robustness_curves(subset, "dynamics_mean_drop", f"degradation_dynamics_{corruption_name}", f"Dynamics Degradation: {corruption_name}")
        plotter.plot_robustness_curves(subset, "fragmentation_mean_drop", f"degradation_fragmentation_{corruption_name}", f"Fragmentation Degradation: {corruption_name}")
        plotter.plot_robustness_curves(subset, "consistency_mean_drop", f"degradation_consistency_{corruption_name}", f"Consistency Degradation: {corruption_name}")

    narrative = [
        f"This corruption benchmark evaluates clip {clip_id} from frames {start_frame} to {end_frame}.",
        "The benchmark fails loudly when no noisy runs are produced and uses shared noisy detections across trackers for fair comparisons.",
        "Image-level corruptions rerun YOLO once per corruption, severity, and run index, then cache the resulting detections for reuse across trackers.",
        "Robustness summaries include mean clean-to-noisy drop, degradation slope, and area under the degradation curve for SCTQ and each component.",
    ]
    extra_sections = [
        {
            "title": "Artifacts",
            "body": (
                "This folder includes all_runs_flat.csv with per-run outputs, "
                "robustness_by_severity.csv with severity-level means, "
                "robustness_by_corruption.csv with per-corruption summaries, "
                "and robustness_overall.csv with tracker-level cross-corruption aggregation."
            ),
        }
    ]

    md_reporter = MarkdownReporter(output_base_dir / "reports")
    md_reporter.generate_report(
        f"Corruption Benchmark: {Path(video_path).stem} ({clip_id})",
        ranked,
        "corruption_report.md",
        narrative=narrative,
        extra_sections=extra_sections,
    )

    save_json(config, output_base_dir / "config_snapshot.json")
    print(json.dumps(ranked, indent=2))

    if return_details:
        return {
            "ranked_summary": ranked,
            "all_runs_flat": all_runs_flat,
            "robustness_by_severity": severity_curves,
            "robustness_by_corruption": per_corruption_summary,
            "robustness_overall": overall_robustness,
        }

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Run corruption robustness suite")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to default.yaml")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of sampled frames.")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--clip-id", type=str, default="clip0")
    parser.add_argument("--scene-type", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} does not exist.")
        sys.exit(1)

    config = ConfigLoader.load_yaml(args.config)
    real_config = ConfigLoader.load_yaml(resolve_config_path(config, "real_video_config", "real_video.yaml"))
    trk_config = ConfigLoader.load_yaml(resolve_config_path(config, "trackers_config", "trackers.yaml"))
    met_config = ConfigLoader.load_yaml(resolve_config_path(config, "metrics_config", "metrics.yaml"))
    cor_config = ConfigLoader.load_yaml(resolve_config_path(config, "corruptions_config", "corruptions.yaml"))

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
        "corruptions": cor_config,
    }

    run_corruption_benchmark(full_config, args.video)


if __name__ == "__main__":
    main()

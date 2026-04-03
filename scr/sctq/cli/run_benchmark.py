import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

from sctq.config import ConfigLoader, get_outputs_dir, get_processed_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.evaluation.multi_run_evaluator import MultiRunEvaluator
from sctq.utils.mot_reader import MOTReader
from sctq.reporting.csv_reporter import CSVReporter
from sctq.reporting.json_reporter import JSONReporter
from sctq.visualization.metric_plots import PlottingManager
from sctq.metrics.ranking import compute_ranking
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.utils.io_utils import save_json

def run_mot_benchmark(config: Dict[str, Any], det_file: str):
    print(f"Running MOT benchmark using detections from: {det_file}")
    strict_mode = config.get("default", {}).get("execution", {}).get("strict_mode", True)

    cfg_metrics = config.get('metrics', {})
    cfg_trackers = config.get('trackers', [])

    output_base_dir = get_outputs_dir(config.get("default", {})) / f"mot_benchmark_{Path(det_file).stem}"
    csv_reporter = CSVReporter(output_base_dir)
    json_reporter = JSONReporter(output_base_dir)
    plotter = PlottingManager(output_base_dir)

    try:
        detections = MOTReader.read_detections(det_file)
    except Exception as e:
        print(f"Error reading MOT detections: {e}")
        return

    if not detections:
        print("No detections found.")
        return

    total_frames = max([fd.frame_idx for fd in detections]) if detections else 0
    print(f"Loaded {len(detections)} frames of detections. Max frame index: {total_frames}")

    evaluator = MultiRunEvaluator(cfg_metrics, strict_mode=strict_mode)

    # Run Evaluation
    results = evaluator.evaluate(
        cfg_trackers,
        detections,
        base_run_id="mot",
        video_id=Path(det_file).stem,
        total_frames=total_frames
    )

    all_run_summaries = []

    for res in results:
        trackset = res['trackset']
        sctq_summary = res['sctq_summary']
        run_summary = res['run_summary']
        t_name = run_summary['tracker_name']

        all_run_summaries.append(run_summary)

        # Save individual run summaries
        csv_reporter.save_per_run(f"mot_{t_name}_summary", run_summary)
        json_reporter.save_per_run(f"mot_{t_name}_summary", run_summary)

        # Reporting
        csv_reporter.save_per_track(f"{t_name}_per_track", sctq_summary['per_track_metrics'])
        json_reporter.save_per_track(f"{t_name}_per_track", sctq_summary['per_track_metrics'])

        lengths = [t.length for t in trackset.tracks.values()]
        plotter.plot_track_length_histogram(lengths, f"{t_name}_lengths", title=f"{t_name} Track Lengths")
        plotter.plot_sctq_components(sctq_summary, f"{t_name}_sctq_comp", title=f"{t_name} SCTQ Components")

    csv_reporter.save_aggregated("mot_benchmark_summary", all_run_summaries)
    json_reporter.save_aggregated("mot_benchmark_summary", all_run_summaries)

    ranked = compute_ranking(all_run_summaries, "sctq_core")
    print("\n--- MOT Benchmark Ranking by SCTQ Core ---")
    print(json.dumps(ranked, indent=2))

    save_json(config, output_base_dir / "config_snapshot.json")

    md_reporter = MarkdownReporter(output_base_dir / "reports")
    md_reporter.generate_report(f"MOT Benchmark: {Path(det_file).stem}", all_run_summaries, "mot_report.md")

    print("MOT benchmark complete.")

def main():
    parser = argparse.ArgumentParser(description="Run MOT Benchmark")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to default.yaml")
    parser.add_argument("--det_file", type=str, required=True, help="Path to MOT format detection txt file.")
    args = parser.parse_args()

    config = ConfigLoader.load_yaml(args.config)
    trk_config = ConfigLoader.load_yaml(resolve_config_path(config, "trackers_config", "trackers.yaml"))
    met_config = ConfigLoader.load_yaml(resolve_config_path(config, "metrics_config", "metrics.yaml"))

    full_config = {
        "default": config,
        "trackers": trk_config.get("trackers", []),
        "metrics": met_config.get("metrics", {})
    }

    run_mot_benchmark(full_config, args.det_file)

if __name__ == "__main__":
    main()

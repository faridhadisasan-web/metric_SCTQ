import argparse
import json
from typing import Any, Dict, Iterable, List, Tuple

from sctq.config import ConfigLoader, get_outputs_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.evaluation.synthetic_validator import SyntheticValidator
from sctq.metrics.ranking import compute_correlation, compute_ranking
from sctq.reporting.csv_reporter import CSVReporter
from sctq.reporting.json_reporter import JSONReporter
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.synthetic.perturbations import apply_synthetic_perturbations
from sctq.synthetic.scene_generator import SyntheticSceneGenerator
from sctq.utils.io_utils import prepare_output_dir, save_json
from sctq.utils.random_utils import set_seed
from sctq.utils.tabular import aggregate_numeric, write_csv_rows
from sctq.visualization.metric_plots import PlottingManager
from sctq.visualization.overlays import render_synthetic_video


def _iter_scene_seed_runs(cfg_syn: Dict[str, Any]) -> Iterable[Tuple[int, int, int]]:
    gen_cfg = cfg_syn.get("generation", {})
    num_scenes = int(gen_cfg.get("num_scenes", 1))
    seeds_per_scene = max(1, int(gen_cfg.get("seeds_per_scene", 1)))
    base_seed = int(gen_cfg.get("base_seed", 42))
    explicit_seeds = gen_cfg.get("scene_seeds")
    for scene_idx in range(num_scenes):
        if explicit_seeds:
            scene_seeds = explicit_seeds[scene_idx] if scene_idx < len(explicit_seeds) else []
            if not scene_seeds:
                scene_seeds = [base_seed + scene_idx]
        else:
            start = base_seed + scene_idx * seeds_per_scene
            scene_seeds = [start + offset for offset in range(seeds_per_scene)]
        for seed_run_idx, seed in enumerate(scene_seeds):
            yield scene_idx, seed_run_idx, int(seed)


def run_synthetic_benchmark(config: Dict[str, Any]) -> None:
    strict_mode = config.get("default", {}).get("execution", {}).get("strict_mode", True)
    clean_outputs = bool(config.get("default", {}).get("execution", {}).get("clean_output_dirs", True))

    cfg_syn = config.get("synthetic", {})
    cfg_metrics = config.get("metrics", {})
    cfg_trackers = config.get("trackers", [])

    render_cfg = cfg_syn.get("rendering", {})
    render_vid = render_cfg.get("render_video", False)
    render_first_seed_only = render_cfg.get("render_first_seed_only", True)
    width = cfg_syn.get("scene_width", 1920)
    height = cfg_syn.get("scene_height", 1080)
    fps = cfg_syn.get("fps", 30)

    output_base_dir = prepare_output_dir(get_outputs_dir(config.get("default", {})) / "synthetic", clean=clean_outputs)
    validator = SyntheticValidator(cfg_metrics, strict_mode=strict_mode)
    csv_reporter = CSVReporter(output_base_dir)
    json_reporter = JSONReporter(output_base_dir)
    plotter = PlottingManager(output_base_dir)

    all_run_summaries: List[Dict[str, Any]] = []
    all_run_outputs_flat: List[Dict[str, Any]] = []

    for scene_idx, seed_run_idx, seed in _iter_scene_seed_runs(cfg_syn):
        set_seed(seed)
        print(f"\n--- Generating Synthetic Scene {scene_idx} | Seed {seed} ---")
        scene_gen = SyntheticSceneGenerator(cfg_syn)
        scene_gen.generate()
        ideal_dets = scene_gen.get_all_detections()
        gt_objects = scene_gen.objects
        perturbed_dets = apply_synthetic_perturbations(ideal_dets, cfg_syn.get("noise", {}))
        results = validator.evaluate(
            cfg_trackers,
            perturbed_dets,
            gt_objects,
            base_run_id=f"syn_scene_{scene_idx}_seed_{seed}",
            video_id=f"synthetic_scene_{scene_idx}",
            total_frames=scene_gen.total_frames,
        )

        for res in results:
            trackset = res["trackset"]
            sctq_summary = res["sctq_summary"]
            run_summary = dict(res["run_summary"])
            t_name = run_summary["tracker_name"]
            run_summary.update({"scene_id": scene_idx, "seed": seed, "seed_run_idx": seed_run_idx})
            all_run_summaries.append(run_summary)

            run_output = {
                "tracker": t_name,
                "scene_id": scene_idx,
                "seed": seed,
                "seed_run_idx": seed_run_idx,
                "sctq_core": sctq_summary["sctq_core"],
                "persistence": sctq_summary["persistence_aggregate"],
                "dynamics": sctq_summary["dynamic_aggregate"],
                "fragmentation": sctq_summary["fragmentation_aggregate"],
                "consistency": sctq_summary["consistency_aggregate"],
                "consistency_effective": sctq_summary["consistency_effective"],
                "tracks": len(trackset.tracks),
                "idp": run_summary.get("idp", 0.0),
                "idr": run_summary.get("idr", 0.0),
                "idf1": run_summary.get("idf1", 0.0),
            }
            all_run_outputs_flat.append(run_output)
            csv_reporter.save_per_run(f"scene_{scene_idx}_seed_{seed}_{t_name}", run_output)
            json_reporter.save_per_run(f"scene_{scene_idx}_seed_{seed}_{t_name}", run_output)
            csv_reporter.save_per_track(f"{t_name}_scene_{scene_idx}_seed_{seed}_per_track", sctq_summary["per_track_metrics"])
            json_reporter.save_per_track(f"{t_name}_scene_{scene_idx}_seed_{seed}_per_track", sctq_summary["per_track_metrics"])

            should_render = render_vid and (not render_first_seed_only or seed_run_idx == 0)
            if should_render:
                vid_path = output_base_dir / "videos" / f"scene_{scene_idx}_seed_{seed}_{t_name}.mp4"
                vid_path.parent.mkdir(parents=True, exist_ok=True)
                render_synthetic_video(trackset, vid_path, width, height, fps, scene_gen.total_frames)
            if seed_run_idx == 0:
                lengths = [t.length for t in trackset.tracks.values()]
                plotter.plot_track_length_histogram(lengths, f"{t_name}_scene_{scene_idx}_lengths", title=f"{t_name} Scene {scene_idx} Track Lengths")

    agg_summaries = aggregate_numeric(all_run_summaries, "tracker_name", excluded={"scene_id", "seed", "seed_run_idx"})
    ranked = compute_ranking(agg_summaries, "sctq_core")
    csv_reporter.save_aggregated("synthetic_summary", agg_summaries)
    json_reporter.save_aggregated("synthetic_summary", agg_summaries)

    keep_keys = {
        "tracker_name", "sctq_core", "sctq_core_std", "gt_track_count", "gt_track_count_std",
        "pred_track_count", "pred_track_count_std", "gt_fragmentation", "gt_fragmentation_std",
        "actual_assignment_purity", "actual_assignment_purity_std", "idp", "idp_std", "idr", "idr_std",
        "idf1", "idf1_std", "num_runs", "persistence_aggregate", "dynamic_aggregate",
        "fragmentation_aggregate", "consistency_aggregate", "consistency_effective"
    }
    val_summaries = [{k: v for k, v in row.items() if k in keep_keys} for row in agg_summaries]
    csv_reporter.save_validation("validation_summary", val_summaries)
    json_reporter.save_validation("validation_summary", val_summaries)

    write_csv_rows(output_base_dir / "all_runs_flat.csv", all_run_outputs_flat)

    correlations = {
        "tracker_mean_idf1": compute_correlation([r.get("sctq_core", 0.0) for r in agg_summaries], [r.get("idf1", 0.0) for r in agg_summaries]),
        "tracker_mean_idp": compute_correlation([r.get("sctq_core", 0.0) for r in agg_summaries], [r.get("idp", 0.0) for r in agg_summaries]),
        "tracker_mean_idr": compute_correlation([r.get("sctq_core", 0.0) for r in agg_summaries], [r.get("idr", 0.0) for r in agg_summaries]),
        "run_level_idf1": compute_correlation([r.get("sctq_core", 0.0) for r in all_run_summaries], [r.get("idf1", 0.0) for r in all_run_summaries]),
    }
    save_json(correlations, output_base_dir / "correlations.json")

    plotter.plot_ranking_chart(ranked, "sctq_core", "ranking_chart", "Synthetic Tracker Ranking (SCTQ)")
    plotter.plot_component_comparison(agg_summaries, "component_comparison", "Synthetic Component Comparison")
    plotter.plot_scatter([r.get("sctq_core", 0.0) for r in agg_summaries], [r.get("idf1", 0.0) for r in agg_summaries], "SCTQ", "IDF1", "sctq_vs_idf1", "SCTQ vs IDF1")
    plotter.plot_scatter([r.get("sctq_core", 0.0) for r in agg_summaries], [r.get("idp", 0.0) for r in agg_summaries], "SCTQ", "IDP", "sctq_vs_idp", "SCTQ vs IDP")
    plotter.plot_scatter([r.get("sctq_core", 0.0) for r in agg_summaries], [r.get("idr", 0.0) for r in agg_summaries], "SCTQ", "IDR", "sctq_vs_idr", "SCTQ vs IDR")

    narrative = [
        "Synthetic validation uses multiple scenes and seeds so tracker rankings are not driven by a single synthetic setup.",
        f"Tracker-mean correlation with IDF1: Pearson={correlations['tracker_mean_idf1']['pearson']:.3f}, Spearman={correlations['tracker_mean_idf1']['spearman']:.3f}, Kendall={correlations['tracker_mean_idf1']['kendall']:.3f}.",
        "The effective consistency term uses gated consistency, so smooth but fragmented trackers cannot recover purely through local consistency.",
    ]
    md_reporter = MarkdownReporter(output_base_dir / "reports")
    md_reporter.generate_report("Synthetic Benchmark Report", ranked, "synthetic_report.md", narrative=narrative)
    save_json(config, output_base_dir / "config_snapshot.json")
    print(json.dumps(ranked, indent=2))
    print("Synthetic benchmark complete.")


def main():
    parser = argparse.ArgumentParser(description="Run Synthetic Benchmark")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to default.yaml")
    args = parser.parse_args()

    config = ConfigLoader.load_yaml(args.config)
    syn_config = ConfigLoader.load_yaml(resolve_config_path(config, "synthetic_config", "synthetic.yaml"))
    trk_config = ConfigLoader.load_yaml(resolve_config_path(config, "trackers_config", "trackers.yaml"))
    met_config = ConfigLoader.load_yaml(resolve_config_path(config, "metrics_config", "metrics.yaml"))
    full_config = {
        "default": config,
        "synthetic": syn_config.get("synthetic_benchmark", {}),
        "trackers": trk_config.get("trackers", []),
        "metrics": met_config.get("metrics", {}),
    }
    run_synthetic_benchmark(full_config)


if __name__ == "__main__":
    main()

import argparse
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from sctq.config import ConfigLoader, get_outputs_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.evaluation.synthetic_validator import SyntheticValidator
from sctq.reporting.csv_reporter import CSVReporter
from sctq.reporting.json_reporter import JSONReporter
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.synthetic.perturbations import apply_synthetic_perturbations
from sctq.synthetic.scene_generator import SyntheticSceneGenerator
from sctq.utils.io_utils import prepare_output_dir, save_json
from sctq.utils.random_utils import set_seed
from sctq.utils.tabular import aggregate_numeric, pivot_mean, write_csv_rows
from sctq.visualization.metric_plots import PlottingManager


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


def run_ablation_study(config: Dict[str, Any]) -> None:
    print("Starting SCTQ Ablation and Baseline Study...")
    strict_mode = config.get("default", {}).get("execution", {}).get("strict_mode", True)
    clean_outputs = bool(config.get("default", {}).get("execution", {}).get("clean_output_dirs", True))
    cfg_syn = config.get("synthetic", {})
    cfg_metrics = config.get("metrics", {})
    cfg_trackers = config.get("trackers", [])

    ablations = [
        {"name": "Full_SCTQ", "weights": {"w_p": 0.10, "w_d": 0.10, "w_f": 0.50, "w_c": 0.30, "use_gated_consistency": True}},
        {"name": "No_Fragmentation", "weights": {"w_p": 0.35, "w_d": 0.25, "w_f": 0.0, "w_c": 0.40, "use_gated_consistency": True}},
        {"name": "No_Dynamics", "weights": {"w_p": 0.20, "w_d": 0.0, "w_f": 0.50, "w_c": 0.30, "use_gated_consistency": True}},
        {"name": "Persistence_Only", "weights": {"w_p": 1.0, "w_d": 0.0, "w_f": 0.0, "w_c": 0.0, "use_gated_consistency": True}},
    ]

    output_base_dir = prepare_output_dir(get_outputs_dir(config.get("default", {})) / "ablation_study", clean=clean_outputs)
    csv_reporter = CSVReporter(output_base_dir)
    json_reporter = JSONReporter(output_base_dir)
    plotter = PlottingManager(output_base_dir)

    all_ablation_results: List[Dict[str, Any]] = []
    baseline_results: List[Dict[str, Any]] = []

    for ab in ablations:
        ab_name = ab["name"]
        print(f"\n--- Running Ablation: {ab_name} ---")
        ab_metrics_cfg = dict(cfg_metrics)
        ab_metrics_cfg["sctq_weights"] = ab["weights"]
        validator = SyntheticValidator(ab_metrics_cfg, strict_mode=strict_mode)

        for scene_idx, seed_run_idx, seed in _iter_scene_seed_runs(cfg_syn):
            set_seed(seed)
            scene_gen = SyntheticSceneGenerator(cfg_syn)
            scene_gen.generate()
            ideal_dets = scene_gen.get_all_detections()
            gt_objects = scene_gen.objects
            perturbed_dets = apply_synthetic_perturbations(ideal_dets, cfg_syn.get("noise", {}))
            results = validator.evaluate(
                cfg_trackers,
                perturbed_dets,
                gt_objects,
                base_run_id=f"ablation_{ab_name}_scene_{scene_idx}_seed_{seed}",
                video_id=f"synthetic_scene_{scene_idx}",
                total_frames=scene_gen.total_frames,
            )
            for res in results:
                run_summary = dict(res["run_summary"])
                t_name = run_summary["tracker_name"]
                run_summary.update({"ablation_config": ab_name, "scene_id": scene_idx, "seed": seed})
                all_ablation_results.append(run_summary)
                if ab_name == "Full_SCTQ":
                    trackset = res["trackset"]
                    lengths = [t.length for t in trackset.tracks.values()]
                    avg_length = float(np.mean(lengths)) if lengths else 0.0
                    baseline_results.append(
                        {
                            "tracker_name": t_name,
                            "scene_id": scene_idx,
                            "seed": seed,
                            "avg_track_length": avg_length,
                            "track_count": len(lengths),
                            "actual_assignment_purity": run_summary.get("actual_assignment_purity", 0.0),
                        }
                    )

    write_csv_rows(output_base_dir / "ablation_results.csv", all_ablation_results)
    write_csv_rows(output_base_dir / "heuristic_baselines.csv", baseline_results)

    ablation_summary = aggregate_numeric(all_ablation_results, "ablation_config", excluded={"scene_id", "seed"})
    by_tracker_summary = aggregate_numeric(all_ablation_results, "tracker_name", excluded={"scene_id", "seed"})
    pivot_rows = pivot_mean(all_ablation_results, "tracker_name", "ablation_config", "sctq_core")
    write_csv_rows(output_base_dir / "ablation_pivot.csv", pivot_rows)
    csv_reporter.save_aggregated("ablation_summary", ablation_summary)
    json_reporter.save_aggregated("ablation_summary", ablation_summary)
    save_json({"baseline_summary": aggregate_numeric(baseline_results, "tracker_name", excluded={"scene_id", "seed"})}, output_base_dir / "baseline_summary.json")

    plotter.plot_ablation(pivot_rows, "ablation_comparison", "Ablation Comparison")

    narrative = [
        "The ablation study evaluates whether each SCTQ component contributes to a tracker ranking that remains aligned with ground-truth-aware validation.",
        "The final configuration keeps all four components and uses gated consistency so consistency cannot rescue fragmented trackers.",
    ]
    md_reporter = MarkdownReporter(output_base_dir / "reports")
    md_reporter.generate_report("SCTQ Ablation Study", by_tracker_summary, "ablation_report.md", narrative=narrative)
    print(f"Ablation study complete. Results saved to {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Ablation and Baseline Study")
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
    run_ablation_study(full_config)


if __name__ == "__main__":
    main()

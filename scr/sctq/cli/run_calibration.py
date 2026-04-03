import argparse
import json
from typing import Any, Dict, Iterable, List, Tuple

from sctq.config import ConfigLoader, get_outputs_dir, resolve_config_path
from sctq.constants import DEFAULT_CONFIG_PATH
from sctq.evaluation.synthetic_validator import SyntheticValidator
from sctq.metrics.ranking import compute_correlation
from sctq.reporting.markdown_reporter import MarkdownReporter
from sctq.synthetic.perturbations import apply_synthetic_perturbations
from sctq.synthetic.scene_generator import SyntheticSceneGenerator
from sctq.utils.io_utils import prepare_output_dir, save_json
from sctq.utils.random_utils import set_seed
from sctq.utils.tabular import aggregate_numeric, sort_rows, write_csv_rows


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


def _select_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    for row in rows:
        balance_penalty = abs(row["w_p"] - 0.1) + abs(row["w_d"] - 0.1) + abs(row["w_f"] - 0.5) + abs(row["w_c"] - 0.3)
        composite = 1.0 * row.get("spearman", 0.0) + 0.35 * row.get("kendall", 0.0) + 0.15 * row.get("pearson", 0.0) - 0.05 * balance_penalty
        enriched = dict(row)
        enriched["selection_score"] = composite
        scored.append(enriched)
    scored = sort_rows(scored, "selection_score")
    return scored[0] if scored else {}


def run_weight_calibration(config: Dict[str, Any]) -> None:
    print("Starting SCTQ Weight Calibration...")
    strict_mode = config.get("default", {}).get("execution", {}).get("strict_mode", True)
    clean_outputs = bool(config.get("default", {}).get("execution", {}).get("clean_output_dirs", True))
    cfg_syn = config.get("synthetic", {})
    cfg_metrics = config.get("metrics", {})
    cfg_trackers = config.get("trackers", [])
    output_base_dir = prepare_output_dir(get_outputs_dir(config.get("default", {})) / "calibration", clean=clean_outputs)

    steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    weight_combinations = []
    for w_p in steps:
        for w_d in steps:
            for w_f in steps:
                w_c = round(1.0 - (w_p + w_d + w_f), 2)
                if 0.1 <= w_c <= 0.6:
                    weight_combinations.append({"w_p": w_p, "w_d": w_d, "w_f": w_f, "w_c": w_c, "use_gated_consistency": True})

    results: List[Dict[str, Any]] = []
    for idx, weights in enumerate(weight_combinations):
        print(f"[{idx + 1}/{len(weight_combinations)}] Testing weights: {weights}")
        ab_metrics_cfg = dict(cfg_metrics)
        ab_metrics_cfg["sctq_weights"] = weights
        validator = SyntheticValidator(ab_metrics_cfg, strict_mode=strict_mode)

        all_runs: List[Dict[str, Any]] = []
        for scene_idx, seed_run_idx, seed in _iter_scene_seed_runs(cfg_syn):
            set_seed(seed)
            scene_gen = SyntheticSceneGenerator(cfg_syn)
            scene_gen.generate()
            ideal_dets = scene_gen.get_all_detections()
            gt_objects = scene_gen.objects
            perturbed_dets = apply_synthetic_perturbations(ideal_dets, cfg_syn.get("noise", {}))
            run_results = validator.evaluate(
                cfg_trackers,
                perturbed_dets,
                gt_objects,
                base_run_id=f"calibration_{idx}_scene_{scene_idx}_seed_{seed}",
                video_id=f"synthetic_scene_{scene_idx}",
                total_frames=scene_gen.total_frames,
            )
            for r in run_results:
                rs = dict(r["run_summary"])
                rs.update({"scene_id": scene_idx, "seed": seed})
                all_runs.append(rs)

        agg = aggregate_numeric(all_runs, "tracker_name", excluded={"scene_id", "seed"})
        sctq_scores = [row.get("sctq_core", 0.0) for row in agg]
        idf1_scores = [row.get("idf1", 0.0) for row in agg]
        correlations = compute_correlation(sctq_scores, idf1_scores)
        results.append({**weights, **correlations})

    write_csv_rows(output_base_dir / "calibration_results.csv", results)
    best_weights = _select_best(results)
    highest_pearson = sort_rows(results, "pearson")[0] if results else {}
    save_json(best_weights, output_base_dir / "best_weights.json")
    save_json(highest_pearson, output_base_dir / "highest_pearson_weights.json")

    rationale = [
        "Calibration selects weights using a multi-criterion objective: Spearman is primary because the paper needs ranking agreement with IDF1, Kendall supports order stability, Pearson preserves monotonic alignment, and a light balance penalty discourages degenerate single-component solutions.",
        f"Selected weights: w_p={best_weights.get('w_p', 0.0)}, w_d={best_weights.get('w_d', 0.0)}, w_f={best_weights.get('w_f', 0.0)}, w_c={best_weights.get('w_c', 0.0)}.",
        f"Highest Pearson-only candidate: w_p={highest_pearson.get('w_p', 0.0)}, w_d={highest_pearson.get('w_d', 0.0)}, w_f={highest_pearson.get('w_f', 0.0)}, w_c={highest_pearson.get('w_c', 0.0)}.",
        "The final choice is preferred when it preserves strong IDF1 rank agreement while keeping fragmentation and gated consistency active in the final metric, which directly addresses the SORT-vs-Centroid inversion problem.",
    ]
    md = MarkdownReporter(output_base_dir / "reports")
    md.generate_report("Calibration Report", [best_weights], "calibration_report.md", narrative=rationale)
    print(json.dumps(best_weights, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run Weight Calibration Grid Search")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
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
    run_weight_calibration(full_config)


if __name__ == "__main__":
    main()

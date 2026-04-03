from typing import Any, Dict, List

from sctq.corruptions.corruption_runner import CorruptionRunner
from sctq.evaluation.single_run_evaluator import SingleRunEvaluator
from sctq.metrics.stability import (
    compute_run_summary_vector,
    compute_sctq_final,
    compute_stability_score,
)
from sctq.types import FrameDetections


class RobustnessEvaluator:
    """Evaluates trackers across clean and corrupted runs to compute stability and final SCTQ."""

    def __init__(self, metrics_config: Dict[str, Any], corruptions_config: Dict[str, Any]):
        self.metrics_config = metrics_config
        self.corruptions_config = corruptions_config
        self.single_eval = SingleRunEvaluator(metrics_config)
        self.runner = CorruptionRunner(corruptions_config)

    def evaluate(
        self,
        trackers_config: List[Dict[str, Any]],
        clean_detections: List[FrameDetections],
        base_run_id: str,
        video_id: str,
        total_frames: int,
    ) -> List[Dict[str, Any]]:
        """Run multiple trackers on clean and corrupted detections."""

        final_results = []
        det_corrupts = self.corruptions_config.get("corruptions", {}).get("detection_level", [])

        for t_config in trackers_config:
            t_name = t_config.get("name")
            print(f"\n--- Testing Tracker Robustness: {t_name} ---")

            # 1. Clean Run
            clean_run_id = f"clean_{base_run_id}_{t_name}"
            clean_result = self.single_eval.evaluate(
                t_name, t_config, clean_detections, clean_run_id, video_id, total_frames
            )

            if not clean_result:
                print(f"Skipping {t_name} due to failure in clean run.")
                continue

            trackset_clean = clean_result["trackset"]
            summary_clean = clean_result["sctq_summary"]
            vec_clean = compute_run_summary_vector(summary_clean, trackset_clean)

            # 2. Corrupted Runs
            noisy_vectors = []
            for c_cfg in det_corrupts:
                if not c_cfg.get("enabled", False):
                    continue

                c_name = c_cfg["name"]
                severities = c_cfg.get("severities", [1])

                print(f"  Applying {c_name}...")

                for sev in severities:
                    # Apply corruption
                    corrupted_dets = []
                    for fd in clean_detections:
                        corr_fd = self.runner.apply_detection_corruptions(
                            fd, [{"name": c_name, "severity": sev}]
                        )
                        corrupted_dets.append(corr_fd)

                    noisy_run_id = f"{c_name}_{sev}_{base_run_id}_{t_name}"
                    noisy_result = self.single_eval.evaluate(
                        t_name, t_config, corrupted_dets, noisy_run_id, video_id, total_frames
                    )

                    if noisy_result:
                        trackset_noisy = noisy_result["trackset"]
                        summary_noisy = noisy_result["sctq_summary"]
                        vec_noisy = compute_run_summary_vector(summary_noisy, trackset_noisy)
                        noisy_vectors.append(vec_noisy)

            # 3. Calculate Stability and Final SCTQ
            stability_score = compute_stability_score(vec_clean, noisy_vectors)
            lambda_stab = self.metrics_config.get("sctq_weights", {}).get("lambda_stab", 0.20)
            sctq_final = compute_sctq_final(
                summary_clean["sctq_core"], stability_score, lambda_stab
            )

            final_summary = {
                "tracker_name": t_name,
                "sctq_core": summary_clean["sctq_core"],
                "stability_score": stability_score,
                "sctq_final": sctq_final,
                "number_of_clean_tracks": len(trackset_clean.tracks),
                "persistence_clean": summary_clean["persistence_aggregate"],
                "dynamic_clean": summary_clean["dynamic_aggregate"],
                "fragmentation_clean": summary_clean["fragmentation_aggregate"],
                "consistency_clean": summary_clean["consistency_aggregate"],
            }

            final_results.append(final_summary)
            print(f"  -> SCTQ Core: {summary_clean['sctq_core']:.4f}")
            print(f"  -> Stability: {stability_score:.4f}")
            print(f"  -> SCTQ Final: {sctq_final:.4f}")

        return final_results

from typing import Dict, Any, List
from sctq.types import FrameDetections
from sctq.synthetic.object_models import SyntheticObject
from sctq.evaluation.single_run_evaluator import SingleRunEvaluator
from sctq.metrics.validation_metrics import compute_synthetic_validation

class SyntheticValidator:
    """Evaluates trackers on synthetic data and computes GT validation metrics."""

    def __init__(self, metrics_config: Dict[str, Any], strict_mode: bool = True):
        self.single_eval = SingleRunEvaluator(metrics_config, strict_mode=strict_mode)

    def evaluate(self, trackers_config: List[Dict[str, Any]], detections: List[FrameDetections], gt_objects: List[SyntheticObject], base_run_id: str, video_id: str, total_frames: int) -> List[Dict[str, Any]]:
        """Run multiple trackers and validate against GT."""

        results = []
        for t_config in trackers_config:
            t_name = t_config.get("name")
            print(f"Running tracker (with validation): {t_name}")

            run_id = f"{base_run_id}_{t_name}"
            result = self.single_eval.evaluate(t_name, t_config, detections, run_id, video_id, total_frames)

            if result:
                trackset = result['trackset']
                val_metrics = compute_synthetic_validation(trackset, gt_objects)

                # Merge validation into run summary
                summary = result['run_summary']
                summary.update({
                    "gt_track_count": val_metrics.get("gt_track_count", 0),
                    "pred_track_count": val_metrics.get("pred_track_count", 0),
                    "gt_fragmentation": val_metrics.get("gt_fragmentation", 0),
                    "idp": val_metrics.get("idp", 0.0),
                    "idr": val_metrics.get("idr", 0.0),
                    "idf1": val_metrics.get("idf1", 0.0),
                    "actual_assignment_purity": val_metrics.get("actual_assignment_purity", 0.0)
                })

                results.append(result)

        return results

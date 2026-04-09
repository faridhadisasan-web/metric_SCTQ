from typing import Dict, Any, List
from sctq.types import FrameDetections
from sctq.evaluation.single_run_evaluator import SingleRunEvaluator

class MultiRunEvaluator:
    """Evaluates multiple trackers on the same sequence of detections."""

    def __init__(self, metrics_config: Dict[str, Any], strict_mode: bool = True):
        self.single_eval = SingleRunEvaluator(metrics_config, strict_mode)

    def evaluate(self, trackers_config: List[Dict[str, Any]], detections: List[FrameDetections], base_run_id: str, video_id: str, total_frames: int) -> List[Dict[str, Any]]:
        """Run multiple trackers on detections and compute metrics."""

        results = []
        for t_config in trackers_config:
            t_name = t_config.get("name")
            print(f"Running tracker: {t_name}")

            run_id = f"{base_run_id}_{t_name}"
            result = self.single_eval.evaluate(t_name, t_config, detections, run_id, video_id, total_frames)

            if result:
                results.append(result)

        return results

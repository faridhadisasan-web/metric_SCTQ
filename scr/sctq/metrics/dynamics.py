import math
from typing import Any, Dict, List, Tuple

import numpy as np

from sctq.types import TrackedObject
from sctq.utils.math_utils import (
    compute_displacement,
    compute_heading,
    compute_speed,
    unwrap_angles,
)


def compute_track_dynamics(track: TrackedObject) -> Dict[str, Any]:
    """Compute dynamic features for a single track."""
    if track.length < 2:
        return {
            "mean_speed": 0.0,
            "median_speed": 0.0,
            "std_speed": 0.0,
            "mean_turn": 0.0,
            "median_turn": 0.0,
            "max_turn": 0.0,
            "mean_acceleration": 0.0,
            "median_acceleration": 0.0,
            "max_jerk": 0.0,
            "raw_point_count": track.length,
        }

    pts = track.points
    speeds: List[float] = []
    headings: List[float] = []

    for i in range(len(pts) - 1):
        dx, dy = compute_displacement((pts[i].cx, pts[i].cy), (pts[i + 1].cx, pts[i + 1].cy))
        speed = compute_speed(dx, dy)
        heading = compute_heading(dx, dy)
        speeds.append(speed)
        headings.append(heading)

    headings = unwrap_angles(headings)

    turns: List[float] = []
    if len(headings) > 1:
        for i in range(len(headings) - 1):
            turns.append(abs(headings[i + 1] - headings[i]))

    accelerations: List[float] = []
    if len(speeds) > 1:
        for i in range(len(speeds) - 1):
            accelerations.append(abs(speeds[i + 1] - speeds[i]))

    jerks: List[float] = []
    if len(accelerations) > 1:
        for i in range(len(accelerations) - 1):
            jerks.append(abs(accelerations[i + 1] - accelerations[i]))

    return {
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "median_speed": float(np.median(speeds)) if speeds else 0.0,
        "std_speed": float(np.std(speeds)) if speeds else 0.0,
        "mean_turn": float(np.mean(turns)) if turns else 0.0,
        "median_turn": float(np.median(turns)) if turns else 0.0,
        "max_turn": float(np.max(turns)) if turns else 0.0,
        "mean_acceleration": float(np.mean(accelerations)) if accelerations else 0.0,
        "median_acceleration": float(np.median(accelerations)) if accelerations else 0.0,
        "max_jerk": float(np.max(jerks)) if jerks else 0.0,
        "raw_point_count": track.length,
    }


def score_track_dynamics(
    features: Dict[str, Any],
    beta_1: float,
    beta_2: float,
    tau_phi: float,
    tau_a: float,
    use_robust_stats: bool = True,
) -> float:
    """Compute dynamic plausibility score for a single track."""
    if features["raw_point_count"] < 3:
        return 1.0

    if use_robust_stats:
        turn_val = features.get("median_turn", 0.0)
        accel_val = features.get("median_acceleration", 0.0)
    else:
        turn_val = features.get("mean_turn", 0.0)
        accel_val = features.get("mean_acceleration", 0.0)

    turn_penalty = min(turn_val / tau_phi, 1.0) if tau_phi > 0 else 1.0
    accel_penalty = min(accel_val / tau_a, 1.0) if tau_a > 0 else 1.0

    jerk_val = features.get("max_jerk", 0.0)
    jerk_penalty = 0.0
    if tau_a > 0 and jerk_val > (2.0 * tau_a):
        jerk_penalty = min(0.5, (jerk_val - 2.0 * tau_a) / tau_a) * 0.5

    dynamic_penalty = beta_1 * turn_penalty + beta_2 * accel_penalty + jerk_penalty
    return max(0.0, 1.0 - dynamic_penalty)


def _compute_track_weight(length: int, min_track_length_for_eval: int) -> float:
    """Effective aggregation weight for a track.

    Short tracklets are the main failure mode that can artificially inflate motion-based
    metrics. We therefore suppress their contribution in component aggregation while still
    keeping their individual scores available for diagnostics.
    """
    if length < min_track_length_for_eval:
        return 0.0
    return float(length - min_track_length_for_eval + 1)


def aggregate_dynamics(
    tracks: List[TrackedObject],
    beta_1: float,
    beta_2: float,
    tau_phi: float,
    tau_a: float,
    use_robust_stats: bool = True,
    min_track_length_for_eval: int = 5,
    aggregation: str = "length_weighted",
) -> Tuple[float, List[float], List[Dict[str, Any]]]:
    """Compute aggregate dynamics score for a set of tracks.

    Returns (aggregate_score, list_of_track_scores, list_of_track_features).
    """
    if not tracks:
        return 0.0, [], []

    scores: List[float] = []
    features_list: List[Dict[str, Any]] = []
    weights: List[float] = []

    for t in tracks:
        feats = compute_track_dynamics(t)
        score = score_track_dynamics(feats, beta_1, beta_2, tau_phi, tau_a, use_robust_stats)
        scores.append(score)
        features_list.append(feats)
        if aggregation == "length_weighted":
            weights.append(_compute_track_weight(t.length, min_track_length_for_eval))
        else:
            weights.append(1.0)

    weights_arr = np.asarray(weights, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)

    if np.any(weights_arr > 0):
        agg = float(np.average(scores_arr, weights=weights_arr))
    else:
        agg = float(np.mean(scores_arr))

    return agg, scores, features_list

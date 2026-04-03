import math
from typing import Any, Dict, List, Tuple

import numpy as np

from sctq.types import TrackedObject


def compute_track_persistence(
    track: TrackedObject, total_frames: int, alpha_p: float = 0.20
) -> float:
    """
    Compute persistence score for a single track.

    Formula: P_i = 1 - exp(-(L_i / T) / alpha_p)
    """
    length = track.length
    if total_frames == 0 or alpha_p == 0:
        return 0.0
    ratio = length / total_frames
    return 1.0 - math.exp(-ratio / alpha_p)


def aggregate_persistence(
    tracks: List[TrackedObject], total_frames: int, alpha_p: float = 0.20
) -> Tuple[float, List[float]]:
    """
    Compute aggregate persistence score over a set of tracks.

    Formula: S_pers = (1 / N) * sum(P_i)
    Returns: (aggregate_score, list_of_track_scores)
    """
    if not tracks:
        return 0.0, []

    scores = [compute_track_persistence(t, total_frames, alpha_p) for t in tracks]
    return float(np.mean(scores)), scores

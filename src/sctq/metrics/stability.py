from typing import Any, Dict, List, Tuple

import numpy as np

from sctq.types import TrackSet


def compute_run_summary_vector(sctq_summary: Dict[str, Any], trackset: TrackSet) -> np.ndarray:
    """
    Extract the summary feature vector m(R) for a run.
    m(R) = [mean_track_length, num_tracks, mean_turn, SCTQ_core]
    """
    tracks = list(trackset.tracks.values())
    num_tracks = len(tracks)
    mean_length = np.mean([t.length for t in tracks]) if num_tracks > 0 else 0.0

    mean_turns = []
    # Try to get mean turn from per_track_metrics if available, otherwise 0
    if "per_track_metrics" in sctq_summary:
        for ptm in sctq_summary["per_track_metrics"]:
            mean_turns.append(ptm.get("mean_turn", 0.0))

    mean_turn = np.mean(mean_turns) if mean_turns else 0.0
    sctq_core = sctq_summary.get("sctq_core", 0.0)

    return np.array([mean_length, num_tracks, mean_turn, sctq_core])


def compute_stability_score(
    clean_vector: np.ndarray, noisy_vectors: List[np.ndarray], epsilon: float = 1e-6
) -> float:
    """
    Compute overall stability score S_stab across multiple corruptions/severities.

    delta_k = (1/J) * sum(|m_j(R_0) - m_j(R_k)| / (|m_j(R_0)| + epsilon))
    S_stab^(k) = max(0, 1 - delta_k)
    S_stab = (1/K) * sum(S_stab^(k))
    """
    if not noisy_vectors:
        return 1.0  # Perfect stability if no noisy runs

    stability_scores = []
    J = len(clean_vector)

    for noisy_vec in noisy_vectors:
        # Compute normalized instability distance delta_k
        diffs = np.abs(clean_vector - noisy_vec)
        norms = np.abs(clean_vector) + epsilon
        delta_k = np.sum(diffs / norms) / J

        # Stability score for severity k
        s_stab_k = max(0.0, 1.0 - delta_k)
        stability_scores.append(s_stab_k)

    return float(np.mean(stability_scores))


def compute_sctq_final(
    sctq_core: float, stability_score: float, lambda_stab: float = 0.20
) -> float:
    """
    Compute final SCTQ score incorporating stability.
    SCTQ = (1 - lambda) * SCTQ_core + lambda * S_stab
    """
    return (1.0 - lambda_stab) * sctq_core + lambda_stab * stability_score

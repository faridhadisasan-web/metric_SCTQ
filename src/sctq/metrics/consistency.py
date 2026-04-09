from typing import Any, Dict, List, Tuple

import numpy as np

from sctq.types import TrackedObject


def compute_track_consistency(track: TrackedObject) -> Dict[str, Any]:
    """Compute intra-track consistency features."""
    if track.length < 2:
        return {
            "bbox_size_std": 0.0,
            "conf_std": 0.0,
            "has_conf": False,
            "appearance_variance": 0.0,
            "has_appearance": False,
        }

    widths = [p.w for p in track.points]
    heights = [p.h for p in track.points]

    std_w = np.std(widths) if widths else 0.0
    std_h = np.std(heights) if heights else 0.0
    size_std = (std_w + std_h) / 2.0

    confs = [p.conf for p in track.points if p.conf is not None]
    conf_std = np.std(confs) if confs else 0.0
    has_conf = len(confs) > 1

    embeddings = [p.embedding for p in track.points if p.embedding is not None]
    appearance_variance = 0.0
    has_appearance = len(embeddings) > 1
    if has_appearance:
        distances: List[float] = []
        n_emb = len(embeddings)
        for i in range(n_emb):
            emb1 = embeddings[i]
            norm1 = np.linalg.norm(emb1)
            if norm1 == 0:
                continue
            emb1_norm = emb1 / norm1
            for j in range(i + 1, n_emb):
                emb2 = embeddings[j]
                norm2 = np.linalg.norm(emb2)
                if norm2 == 0:
                    distances.append(1.0)
                else:
                    cos_sim = np.dot(emb1_norm, emb2 / norm2)
                    distances.append(1.0 - cos_sim)
        if distances:
            appearance_variance = float(np.mean(distances))

    return {
        "bbox_size_std": float(size_std),
        "conf_std": float(conf_std),
        "has_conf": has_conf,
        "appearance_variance": appearance_variance,
        "has_appearance": has_appearance,
    }


def score_track_consistency(
    features: Dict[str, Any],
    tau_size: float,
    tau_conf: float,
    gamma_1: float,
    gamma_2: float,
    tau_app: float = 0.5,
    gamma_app: float = 0.5,
) -> float:
    """Compute consistency score for a single track."""
    size_std = features.get("bbox_size_std", 0.0)
    conf_std = features.get("conf_std", 0.0)
    has_conf = features.get("has_conf", False)
    app_var = features.get("appearance_variance", 0.0)
    has_app = features.get("has_appearance", False)

    size_penalty = min(size_std / tau_size, 1.0) if tau_size > 0 else 1.0
    total_penalty = gamma_1 * size_penalty

    if has_conf:
        conf_penalty = min(conf_std / tau_conf, 1.0) if tau_conf > 0 else 1.0
        total_penalty += gamma_2 * conf_penalty

    if has_app:
        app_penalty = min(app_var / tau_app, 1.0) if tau_app > 0 else 1.0
        total_penalty += gamma_app * app_penalty

    return max(0.0, 1.0 - total_penalty)


def _compute_track_weight(length: int, min_track_length_for_eval: int) -> float:
    if length < min_track_length_for_eval:
        return 0.0
    return float(length - min_track_length_for_eval + 1)


def aggregate_consistency(
    tracks: List[TrackedObject],
    tau_size: float,
    tau_conf: float,
    gamma_1: float,
    gamma_2: float,
    tau_app: float = 0.5,
    gamma_app: float = 0.5,
    min_track_length_for_eval: int = 5,
    aggregation: str = "length_weighted",
) -> Tuple[float, List[float], List[Dict[str, Any]]]:
    """Compute aggregate consistency score for a set of tracks."""
    if not tracks:
        return 0.0, [], []

    scores: List[float] = []
    features_list: List[Dict[str, Any]] = []
    weights: List[float] = []

    for t in tracks:
        feats = compute_track_consistency(t)
        score = score_track_consistency(feats, tau_size, tau_conf, gamma_1, gamma_2, tau_app, gamma_app)
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

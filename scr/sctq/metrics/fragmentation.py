import math
from typing import List

import numpy as np

from sctq.metrics.geometry import distance_between_points
from sctq.types import TrackedObject
from sctq.utils.math_utils import angular_difference, compute_heading


def compute_simple_fragmentation(
    tracks: List[TrackedObject], total_frames: int, tau_short: float = 0.10
) -> float:
    """Level A: simple short-track proxy."""
    if not tracks:
        return 0.0

    n_short = 0
    for t in tracks:
        r_i = t.length / total_frames if total_frames > 0 else 0
        if r_i < tau_short:
            n_short += 1

    return max(0.0, 1.0 - (n_short / len(tracks)))


def compute_continuous_bridgeable_fragmentation(
    tracks: List[TrackedObject],
    tau_t: int = 5,
    tau_s: float = 50.0,
    tau_theta: float = 0.5,
    w_t: float = 0.3,
    w_s: float = 0.4,
    w_theta: float = 0.3,
) -> float:
    """Level B: bridgeable-tracklet continuous penalty."""
    if len(tracks) < 2:
        return 1.0

    bridgeable_scores = []
    sorted_tracks = sorted(tracks, key=lambda x: x.start_frame)

    for i in range(len(sorted_tracks)):
        t1 = sorted_tracks[i]
        if t1.length < 2:
            continue

        p_last = t1.points[-1]
        p_prev = t1.points[-2]
        v_x = p_last.cx - p_prev.cx
        v_y = p_last.cy - p_prev.cy
        theta1 = compute_heading(v_x, v_y)

        for j in range(i + 1, len(sorted_tracks)):
            t2 = sorted_tracks[j]
            if t2.length < 2:
                continue

            dt = t2.start_frame - t1.end_frame
            if dt < 0:
                continue
            if dt > tau_t:
                break

            p_first = t2.points[0]
            p_next = t2.points[1]
            v_x2 = p_next.cx - p_first.cx
            v_y2 = p_next.cy - p_first.cy
            theta2 = compute_heading(v_x2, v_y2)

            d_theta = angular_difference(theta1, theta2)
            if d_theta > tau_theta:
                continue

            extrap_x = p_last.cx + v_x * dt
            extrap_y = p_last.cy + v_y * dt
            d_space = distance_between_points((extrap_x, extrap_y), (p_first.cx, p_first.cy))

            avg_speed_t1 = math.sqrt(v_x ** 2 + v_y ** 2)
            dynamic_tau_s = max(tau_s, avg_speed_t1 * dt * 1.5)
            if d_space > dynamic_tau_s:
                continue

            p_t = min(dt / tau_t, 1.0) if tau_t > 0 else 1.0
            p_s = min(d_space / dynamic_tau_s, 1.0) if dynamic_tau_s > 0 else 1.0
            p_theta = min(d_theta / tau_theta, 1.0) if tau_theta > 0 else 1.0
            b_ij = w_t * p_t + w_s * p_s + w_theta * p_theta
            bridgeable_scores.append(b_ij)

    if not bridgeable_scores:
        return 1.0

    mean_penalty = float(np.mean(bridgeable_scores))
    return max(0.0, 1.0 - mean_penalty)


def compute_hybrid_fragmentation(
    tracks: List[TrackedObject],
    total_frames: int,
    tau_short: float = 0.10,
    tau_t: int = 5,
    tau_s: float = 50.0,
    tau_theta: float = 0.5,
    w_t: float = 0.3,
    w_s: float = 0.4,
    w_theta: float = 0.3,
    alpha_simple: float = 0.5,
) -> float:
    """Hybrid fragmentation score.

    Combines the coarse short-track proxy with the finer bridgeable-track penalty.
    This avoids ceiling effects where the bridgeable score saturates at 1.0 for strong
    trackers while preserving sensitivity to obvious fragmentation.
    """
    alpha_simple = min(max(alpha_simple, 0.0), 1.0)
    simple_score = compute_simple_fragmentation(tracks, total_frames, tau_short)
    bridgeable_score = compute_continuous_bridgeable_fragmentation(
        tracks,
        tau_t=tau_t,
        tau_s=tau_s,
        tau_theta=tau_theta,
        w_t=w_t,
        w_s=w_s,
        w_theta=w_theta,
    )
    return float(alpha_simple * simple_score + (1.0 - alpha_simple) * bridgeable_score)

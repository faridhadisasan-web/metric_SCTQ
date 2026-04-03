from typing import Any, Dict, List, Tuple

import numpy as np

from sctq.core.validators import ConfigValidator
from sctq.metrics.consistency import aggregate_consistency
from sctq.metrics.dynamics import aggregate_dynamics
from sctq.metrics.fragmentation import (
    compute_continuous_bridgeable_fragmentation,
    compute_simple_fragmentation,
)
from sctq.metrics.persistence import aggregate_persistence
from sctq.types import TrackSet


class SCTQEngine:
    """Core metric engine for computing SCTQ_core."""

    def __init__(self, config: Dict[str, Any]):
        ConfigValidator.validate_metrics_config(config)
        self.config = config

    def compute_sctq_core(self, trackset: TrackSet) -> Dict[str, Any]:
        """Compute all SCTQ core components and final score for a run."""
        tracks = list(trackset.tracks.values())
        total_frames = trackset.total_frames

        if not tracks or total_frames == 0:
            return {
                "persistence_aggregate": 0.0,
                "dynamic_aggregate": 0.0,
                "fragmentation_aggregate": 0.0,
                "consistency_aggregate": 0.0,
                "consistency_effective": 0.0,
                "sctq_core": 0.0,
                "per_track_metrics": [],
            }

        # 1. Persistence
        alpha_p = self.config.get("persistence", {}).get("alpha_p", 0.20)
        s_pers_agg, track_pers_scores = aggregate_persistence(tracks, total_frames, alpha_p)

        # 2. Dynamics
        dyn_cfg = self.config.get("dynamics", {})
        beta_1 = dyn_cfg.get("beta_1", 0.5)
        beta_2 = dyn_cfg.get("beta_2", 0.5)
        tau_phi = dyn_cfg.get("tau_phi", 0.35)
        tau_a = dyn_cfg.get("tau_a", 10.0)
        s_dyn_agg, track_dyn_scores, track_dyn_feats = aggregate_dynamics(
            tracks, beta_1, beta_2, tau_phi, tau_a
        )

        # 3. Fragmentation
        frag_cfg = self.config.get("fragmentation", {})
        method = frag_cfg.get("method", "simple")
        if method == "continuous_bridgeable":
            s_frag_agg = compute_continuous_bridgeable_fragmentation(
                tracks,
                tau_t=frag_cfg.get("tau_t", 5),
                tau_s=frag_cfg.get("tau_s", 50.0),
                tau_theta=frag_cfg.get("tau_theta", 0.5),
                w_t=frag_cfg.get("w_t", 0.3),
                w_s=frag_cfg.get("w_s", 0.4),
                w_theta=frag_cfg.get("w_theta", 0.3),
            )
        else:  # Default to simple
            tau_short = frag_cfg.get("tau_short", 0.10)
            s_frag_agg = compute_simple_fragmentation(tracks, total_frames, tau_short)

        # 4. Consistency
        cons_cfg = self.config.get("consistency", {})
        tau_size = cons_cfg.get("tau_size", 10.0)
        tau_conf = cons_cfg.get("tau_conf", 0.2)
        gamma_1 = cons_cfg.get("gamma_1", 0.5)
        gamma_2 = cons_cfg.get("gamma_2", 0.5)
        s_cons_agg, track_cons_scores, track_cons_feats = aggregate_consistency(
            tracks, tau_size, tau_conf, gamma_1, gamma_2
        )

        # Aggregate SCTQ_core
        w = self.config["sctq_weights"]
        use_gated_consistency = w.get("use_gated_consistency", True)
        if use_gated_consistency:
            consistency_term = s_cons_agg * float(np.sqrt(max(0.0, s_pers_agg * s_frag_agg)))
        else:
            consistency_term = s_cons_agg

        sctq_core = (
            w.get("w_p", 0.35) * s_pers_agg
            + w.get("w_d", 0.25) * s_dyn_agg
            + w.get("w_f", 0.20) * s_frag_agg
            + w.get("w_c", 0.20) * consistency_term
        )

        # Compile per-track metrics for detailed output
        per_track_metrics = []
        for i, t in enumerate(tracks):
            ptm = {
                "experiment_id": trackset.run_id,  # Can be overridden by reporter
                "run_id": trackset.run_id,
                "tracker_name": trackset.tracker_name,
                "video_or_scene_id": trackset.video_id,
                "track_id": t.track_id,
                "start_frame": t.start_frame,
                "end_frame": t.end_frame,
                "length": t.length,
                "relative_length": t.length / total_frames if total_frames > 0 else 0,
                "mean_speed": track_dyn_feats[i].get("mean_speed", 0.0),
                "std_speed": track_dyn_feats[i].get("std_speed", 0.0),
                "mean_turn": track_dyn_feats[i].get("mean_turn", 0.0),
                "max_turn": track_dyn_feats[i].get("max_turn", 0.0),
                "mean_acceleration": track_dyn_feats[i].get("mean_acceleration", 0.0),
                "bbox_size_std": track_cons_feats[i].get("bbox_size_std", 0.0),
                "persistence_score": track_pers_scores[i] if i < len(track_pers_scores) else 0.0,
                "dynamic_score": track_dyn_scores[i] if i < len(track_dyn_scores) else 0.0,
                "consistency_score": track_cons_scores[i] if i < len(track_cons_scores) else 0.0,
                "raw_point_count": t.length,
            }
            per_track_metrics.append(ptm)

        return {
            "persistence_aggregate": float(s_pers_agg),
            "dynamic_aggregate": float(s_dyn_agg),
            "fragmentation_aggregate": float(s_frag_agg),
            "consistency_aggregate": float(s_cons_agg),
            "consistency_effective": float(consistency_term),
            "sctq_core": float(sctq_core),
            "per_track_metrics": per_track_metrics,
        }

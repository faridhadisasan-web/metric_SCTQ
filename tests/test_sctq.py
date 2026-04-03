import unittest

from sctq.core.data_models import TrackPoint
from sctq.core.track_history import TrackedObject
from sctq.metrics.sctq import SCTQEngine


class TestSCTQEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "persistence": {"alpha_p": 0.20},
            "dynamics": {"beta_1": 0.5, "beta_2": 0.5, "tau_phi": 0.35, "tau_a": 10.0},
            "fragmentation": {"method": "simple", "tau_short": 0.10},
            "consistency": {"tau_size": 10.0, "tau_conf": 0.2, "gamma_1": 0.5, "gamma_2": 0.5},
            "sctq_weights": {"w_p": 0.35, "w_d": 0.25, "w_f": 0.20, "w_c": 0.20},
        }
        self.engine = SCTQEngine(self.config)

    def test_empty_trackset(self):
        from sctq.types import TrackSet

        ts = TrackSet(
            run_id="test", tracker_name="test", video_id="test", tracks={}, total_frames=100
        )
        res = self.engine.compute_sctq_core(ts)
        self.assertEqual(res["sctq_core"], 0.0)

    def test_single_perfect_track(self):
        # A perfectly linear track spanning all frames, consistent size
        pts = []
        for i in range(100):
            pts.append(
                TrackPoint(frame_idx=i, cx=100 + i * 5, cy=100 + i * 5, w=50, h=50, conf=1.0)
            )

        t = TrackedObject(track_id=1, points=pts)

        from sctq.types import TrackSet

        ts = TrackSet(
            run_id="test", tracker_name="test", video_id="test", tracks={1: t}, total_frames=100
        )

        res = self.engine.compute_sctq_core(ts)

        # persistence: length 100 / total 100 = 1.0, alpha_p 0.2 -> 1 - exp(-1/0.2) = 0.993
        # dynamics: perfectly linear (no turns, no accel) -> 1.0
        # fragmentation: length 100 / 100 = 1.0 > 0.10 -> 0 short tracks -> 1.0
        # consistency: w, h constant -> std 0 -> 1.0

        self.assertAlmostEqual(res["dynamic_aggregate"], 1.0, places=4)
        self.assertAlmostEqual(res["fragmentation_aggregate"], 1.0, places=4)
        self.assertAlmostEqual(res["consistency_aggregate"], 1.0, places=4)
        self.assertGreater(res["persistence_aggregate"], 0.99)
        self.assertGreater(res["sctq_core"], 0.99)


if __name__ == "__main__":
    unittest.main()


def test_gated_consistency_reduces_score_for_fragmented_case():
    base_config = {
        "persistence": {"alpha_p": 0.20},
        "dynamics": {"beta_1": 0.5, "beta_2": 0.5, "tau_phi": 0.35, "tau_a": 10.0},
        "fragmentation": {"method": "simple", "tau_short": 0.10},
        "consistency": {"tau_size": 10.0, "tau_conf": 0.2, "gamma_1": 0.5, "gamma_2": 0.5},
    }
    cfg_raw = {**base_config, "sctq_weights": {"w_p": 0.35, "w_d": 0.25, "w_f": 0.20, "w_c": 0.20, "use_gated_consistency": False}}
    cfg_gated = {**base_config, "sctq_weights": {"w_p": 0.35, "w_d": 0.25, "w_f": 0.20, "w_c": 0.20, "use_gated_consistency": True}}

    pts = []
    for i in range(5):
        pts.append(TrackPoint(frame_idx=i, cx=100 + i * 5, cy=100 + i * 5, w=50, h=50, conf=1.0))
    t = TrackedObject(track_id=1, points=pts)
    from sctq.types import TrackSet
    ts = TrackSet(run_id="test", tracker_name="test", video_id="test", tracks={1: t}, total_frames=100)

    res_raw = SCTQEngine(cfg_raw).compute_sctq_core(ts)
    res_gated = SCTQEngine(cfg_gated).compute_sctq_core(ts)

    assert res_gated["consistency_effective"] <= res_gated["consistency_aggregate"]
    assert res_gated["sctq_core"] < res_raw["sctq_core"]

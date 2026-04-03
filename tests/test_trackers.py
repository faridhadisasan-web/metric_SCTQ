import unittest
from typing import Dict, Any
from sctq.types import FrameDetections, DetectionRaw
from sctq.tracking.internal_sort import InternalSORTTracker
from sctq.tracking.internal_iou import InternalIOUTracker
from sctq.tracking.internal_centroid import InternalCentroidTracker, InternalCentroidKFTracker
from sctq.core.trackset import TrackSet
from sctq.metrics.fragmentation import compute_simple_fragmentation

class TestInternalTrackers(unittest.TestCase):
    def setUp(self):
        # Create a simple 10-frame sequence of 2 objects moving linearly
        self.detections = []
        for i in range(10):
            # Speed is 5 pixels per frame, box is 50x50. This guarantees high overlap (IoU)
            # between consecutive frames so the tracker doesn't lose them.
            d1 = DetectionRaw(frame_idx=i, det_id=1, cx=100 + i*5, cy=100 + i*5, w=50, h=50, conf=0.9, class_id=0)
            d2 = DetectionRaw(frame_idx=i, det_id=2, cx=800 - i*5, cy=800 - i*5, w=50, h=50, conf=0.9, class_id=0)
            self.detections.append(FrameDetections(frame_idx=i, detections=[d1, d2]))

    def test_internal_sort_survival(self):
        """Test that InternalSORTTracker doesn't prematurely kill tracks on a simple linear scene."""
        from sctq.tracking.internal_sort import KalmanBoxTracker
        KalmanBoxTracker.count = 0

        # Using a very low IOU threshold ensures any overlap is matched
        tracker = InternalSORTTracker("sort", {"max_age": 2, "min_hits": 1, "iou_threshold": 0.05})

        track_histories = {}

        for fd in self.detections:
            tracks = tracker.update(fd)
            # Make sure it returns 2 tracks after the initial frame
            if fd.frame_idx > 0:
                self.assertEqual(len(tracks), 2)

            for t in tracks:
                if t.track_id not in track_histories:
                    track_histories[t.track_id] = 0
                track_histories[t.track_id] += 1

        # The internal SORT assigns a new ID every time if it loses tracks.
        # So we should only have 2 unique track IDs total in the entire sequence.
        self.assertEqual(len(track_histories), 2)
        for count in track_histories.values():
            self.assertEqual(count, 10)

    def test_centroid_distinctness(self):
        """Test that standard Centroid and KF-Centroid produce mathematically distinct outputs under noise."""
        tracker_c = InternalCentroidTracker("centroid", {"max_lost": 2})
        tracker_kf = InternalCentroidKFTracker("centroidkf", {"max_lost": 2})

        # Add a noise spike in frame 5
        noisy_detections = []
        for i in range(10):
            if i == 5:
                # Jerk the detection way off
                d1 = DetectionRaw(frame_idx=i, det_id=1, cx=100 + i*5 + 100, cy=100 + i*5 + 100, w=50, h=50, conf=0.9, class_id=0)
            else:
                d1 = DetectionRaw(frame_idx=i, det_id=1, cx=100 + i*5, cy=100 + i*5, w=50, h=50, conf=0.9, class_id=0)
            noisy_detections.append(FrameDetections(frame_idx=i, detections=[d1]))

        c_tracks = {}
        kf_tracks = {}

        for fd in noisy_detections:
            c_update = tracker_c.update(fd)
            kf_update = tracker_kf.update(fd)

            for t in c_update:
                if t.track_id not in c_tracks:
                    c_tracks[t.track_id] = []
                c_tracks[t.track_id].append((t.points[0].cx, t.points[0].cy))

            for t in kf_update:
                if t.track_id not in kf_tracks:
                    kf_tracks[t.track_id] = []
                kf_tracks[t.track_id].append((t.points[0].cx, t.points[0].cy))

        # The outputs should not be identical because KF smooths the spike
        self.assertNotEqual(c_tracks, kf_tracks)


    def test_internal_iou_matching_and_deletion(self):
        """Test InternalIOUTracker matches correctly and deletes old tracks."""
        tracker = InternalIOUTracker("iou", {"max_lost": 2, "iou_threshold": 0.2})

        d1_f0 = DetectionRaw(frame_idx=0, det_id=1, cx=100, cy=100, w=50, h=50, conf=0.9, class_id=0)
        fd0 = FrameDetections(frame_idx=0, detections=[d1_f0])

        # Frame 0: Create track
        tracks0 = tracker.update(fd0)
        self.assertEqual(len(tracks0), 1)
        track_id = tracks0[0].track_id

        # Frame 1: Empty (lost age = 1)
        fd1 = FrameDetections(frame_idx=1, detections=[])
        tracks1 = tracker.update(fd1)
        self.assertEqual(len(tracks1), 0)

        # Frame 2: Empty (lost age = 2, max_lost hit so it should die after this)
        fd2 = FrameDetections(frame_idx=2, detections=[])
        tracks2 = tracker.update(fd2)
        self.assertEqual(len(tracks2), 0)

        # Frame 3: Re-appear in exact same spot. Should be a NEW track ID because the old one was deleted.
        d1_f3 = DetectionRaw(frame_idx=3, det_id=1, cx=100, cy=100, w=50, h=50, conf=0.9, class_id=0)
        fd3 = FrameDetections(frame_idx=3, detections=[d1_f3])
        tracks3 = tracker.update(fd3)
        self.assertEqual(len(tracks3), 1)

        # It must be a new ID, not the old one
        self.assertNotEqual(tracks3[0].track_id, track_id)

    def test_fragmentation_metric_on_deliberate_split(self):
        """Test that the fragmentation metric penalizes a deliberately split track."""
        from sctq.core.data_models import TrackedObject, TrackPoint
        from sctq.types import TrackSet

        # Create a single ground truth object moving linearly
        # But represented as two separate track IDs with a gap in the middle
        t1_pts = []
        for i in range(5):
             t1_pts.append(TrackPoint(frame_idx=i, cx=100+i*5, cy=100+i*5, w=50, h=50, conf=1.0, class_id=0))
        t1 = TrackedObject(track_id=1, points=t1_pts)

        t2_pts = []
        # Missing frames 5, 6. Reappears at frame 7.
        for i in range(7, 12):
             t2_pts.append(TrackPoint(frame_idx=i, cx=100+i*5, cy=100+i*5, w=50, h=50, conf=1.0, class_id=0))
        t2 = TrackedObject(track_id=2, points=t2_pts)

        # Simple fragmentation score should be high for both because they are short relative to the 12 total frames
        ts = TrackSet(video_id="test", run_id="run1", tracker_name="test_tracker", total_frames=12, tracks={1: t1, 2: t2})

        frag_1 = compute_simple_fragmentation([t1], 12, tau_short=0.5)
        frag_2 = compute_simple_fragmentation([t2], 12, tau_short=0.5)

        # simple frag returns S_frag = 1.0 - (n_short / n_tracks)
        # N=1, N_short=1 => 1.0 - 1.0 = 0.0
        # For both tracks combined:
        frag_combined = compute_simple_fragmentation([t1, t2], 12, tau_short=0.5)
        self.assertEqual(frag_combined, 0.0) # both tracks are short, 1 - (2/2) = 0


if __name__ == "__main__":
    unittest.main()

from typing import Any, Dict, List

from sctq.tracking.tracker_base import BaseTrackerAdapter
from sctq.types import FrameDetections, TrackedObject, TrackPoint


class UltralyticsAdapter(BaseTrackerAdapter):
    """Adapter for ultralytics trackers (ByteTrack, BoT-SORT)."""

    def __init__(self, name: str, params: Dict[str, Any], tracker_class: type):
        super().__init__(name, params)
        self.tracker_class = tracker_class
        self.tracker = None

        # Add default args that ultralytics trackers expect if not provided
        if "track_high_thresh" not in self.params:
            self.params["track_high_thresh"] = 0.5
        if "track_low_thresh" not in self.params:
            self.params["track_low_thresh"] = 0.1
        if "new_track_thresh" not in self.params:
            self.params["new_track_thresh"] = 0.6
        if "track_buffer" not in self.params:
            self.params["track_buffer"] = 30
        if "match_thresh" not in self.params:
            self.params["match_thresh"] = 0.8
        if "gmc_method" not in self.params:  # for bot-sort
            self.params["gmc_method"] = "sparseOptFlow"
        if "proximity_thresh" not in self.params:
            self.params["proximity_thresh"] = 0.5
        if "appearance_thresh" not in self.params:
            self.params["appearance_thresh"] = 0.25
        if "with_reid" not in self.params:
            self.params["with_reid"] = False
        if "fuse_score" not in self.params:
            self.params["fuse_score"] = True

    def reset(self) -> None:
        """Initialize or reset the tracker instance."""
        import types

        args = types.SimpleNamespace(**self.params)
        self.tracker = self.tracker_class(args, frame_rate=30)

    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        """Update tracker with frame detections."""
        if self.tracker is None:
            self.reset()

        import numpy as np
        import torch
        from ultralytics.engine.results import Results

        # Ultralytics expects detections in [x1, y1, x2, y2, conf, cls] format
        dets = []
        xywh_dets = []
        for det in frame_detections.detections:
            x1 = det.cx - det.w / 2
            y1 = det.cy - det.h / 2
            x2 = det.cx + det.w / 2
            y2 = det.cy + det.h / 2
            conf = det.conf if det.conf is not None else 1.0
            cls_id = det.class_id if det.class_id is not None else 0
            dets.append([x1, y1, x2, y2, conf, cls_id])
            xywh_dets.append([det.cx, det.cy, det.w, det.h])

        # Format required: shape (N, 6)
        dets_tensor = torch.tensor(dets) if dets else torch.empty((0, 6))
        xywh_tensor = torch.tensor(xywh_dets) if xywh_dets else torch.empty((0, 4))

        # We must construct a true ultralytics Results object
        class MockBoxes:
            def __init__(self, tensor, xywh):
                self.data = tensor
                self.conf = tensor[:, 4] if tensor.shape[0] > 0 else torch.empty((0,))
                self.cls = tensor[:, 5] if tensor.shape[0] > 0 else torch.empty((0,))
                self.xywh = xywh

        class MockResults:
            def __init__(self, tensor, xywh):
                self.boxes = MockBoxes(tensor, xywh)
                self.conf = self.boxes.conf
                self.cls = self.boxes.cls
                self.xywh = self.boxes.xywh

            def __getitem__(self, idx):
                # Returns a new MockResults sliced by boolean mask idx
                if hasattr(self.boxes.data, "shape") and self.boxes.data.shape[0] > 0:
                    return MockResults(self.boxes.data[idx], self.boxes.xywh[idx])
                else:
                    return MockResults(torch.empty((0, 6)), torch.empty((0, 4)))

            def __len__(self):
                return self.boxes.data.shape[0]

        results = MockResults(dets_tensor, xywh_tensor)

        empty_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        try:
            # Different trackers might expect either the tensor or the results object
            # We try the results object first as newer ultralytics API requires it
            tracks = self.tracker.update(results, empty_img)
        except AttributeError:
            try:
                # Fallback to tensor if it's an older API
                tracks = self.tracker.update(dets_tensor, empty_img)
            except Exception as e:
                import logging

                logging.error(f"Ultralytics tracker {self.name} failed during update: {e}")
                raise e
        except Exception as e:
            import logging

            logging.error(f"Ultralytics tracker {self.name} failed during update: {e}")
            raise e

        # Convert output back to TrackedObject format
        updated_tracks = []
        if len(tracks) > 0:
            for track in tracks:
                # Format depends on internal version, usually [x1, y1, x2, y2, track_id, conf, cls] or similar
                # Ultralytics track update returns tracks array
                if hasattr(track, "cpu"):
                    arr = track.cpu().numpy()
                else:
                    arr = track

                x1, y1, x2, y2 = arr[0], arr[1], arr[2], arr[3]
                track_id = arr[4]
                conf = arr[5] if len(arr) > 5 else 1.0
                cls_id = arr[6] if len(arr) > 6 else 0

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                tp = TrackPoint(
                    frame_idx=frame_detections.frame_idx,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    conf=conf,
                    class_id=int(cls_id),
                )
                updated_tracks.append(TrackedObject(track_id=int(track_id), points=[tp]))

        return updated_tracks

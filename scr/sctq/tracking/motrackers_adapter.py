from typing import Any, Dict, List

import numpy as np

from sctq.core.data_models import detection_to_trackpoint
from sctq.tracking.tracker_base import BaseTrackerAdapter
from sctq.types import FrameDetections, TrackedObject, TrackPoint


class MOTraackersAdapter(BaseTrackerAdapter):
    """Adapter for motrackers package trackers."""

    def __init__(self, name: str, params: Dict[str, Any], tracker_class: type):
        super().__init__(name, params)
        self.tracker_class = tracker_class
        self.tracker = None

    def reset(self) -> None:
        """Initialize or reset the tracker instance."""
        self.tracker = self.tracker_class(**self.params)

    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        """Update tracker with frame detections."""
        if self.tracker is None:
            self.reset()

        # Convert detections to motrackers format (bboxes as [xmin, ymin, w, h], confs, class_ids)
        bboxes = []
        confs = []
        class_ids = []

        for det in frame_detections.detections:
            xmin = det.cx - det.w / 2
            ymin = det.cy - det.h / 2
            bboxes.append([xmin, ymin, det.w, det.h])
            confs.append(det.conf if det.conf is not None else 1.0)
            class_ids.append(det.class_id if det.class_id is not None else 0)

        bboxes = np.array(bboxes) if bboxes else np.empty((0, 4))
        confs = np.array(confs) if confs else np.empty((0,))
        class_ids = np.array(class_ids) if class_ids else np.empty((0,))

        # Update tracker
        tracks = self.tracker.update(bboxes, confs, class_ids)

        # Convert output back to TrackedObject format (active tracks only for this frame)
        updated_tracks = []
        for track in tracks:
            # track format: (frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)
            # where bb_left, bb_top are x1, y1
            _, track_id, bb_left, bb_top, bb_width, bb_height, conf, _, _, _ = track

            cx = bb_left + bb_width / 2
            cy = bb_top + bb_height / 2

            tp = TrackPoint(
                frame_idx=frame_detections.frame_idx,
                cx=cx,
                cy=cy,
                w=bb_width,
                h=bb_height,
                conf=conf,
                class_id=None,  # Defaulting class ID to None for now, as motrackers doesn't always preserve it
            )
            updated_tracks.append(TrackedObject(track_id=int(track_id), points=[tp]))

        return updated_tracks

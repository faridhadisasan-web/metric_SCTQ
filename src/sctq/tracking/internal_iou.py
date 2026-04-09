from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from sctq.metrics.geometry import compute_iou
from sctq.tracking.tracker_base import BaseTrackerAdapter
from sctq.types import FrameDetections, TrackedObject, TrackPoint


class InternalIOUTracker(BaseTrackerAdapter):
    """Simple internal implementation of IOU Tracker."""

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.sigma_l = params.get("sigma_l", 0.0)
        self.sigma_h = params.get("sigma_h", 0.5)
        self.sigma_iou = params.get("sigma_iou", 0.3)
        self.t_min = params.get("t_min", 2)

        self.active_tracks = []  # List of dicts: {'id', 'bbox', 'max_score', 'frame_idx'}
        self.next_id = 1

    def reset(self) -> None:
        self.active_tracks = []
        self.next_id = 1

    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        frame_idx = frame_detections.frame_idx

        # Filter detections by high threshold
        dets = [
            d for d in frame_detections.detections if (d.conf is None or d.conf >= self.sigma_l)
        ]

        updated_tracks = []
        unmatched_dets = []

        # Compute IOU between active tracks and detections
        for det in dets:
            det_bbox = (det.cx, det.cy, det.w, det.h)
            best_iou = self.sigma_iou
            best_track_idx = -1

            for i, track in enumerate(self.active_tracks):
                if track["frame_idx"] < frame_idx - 1:
                    continue  # Track is lost

                iou = compute_iou(track["bbox"], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = i

            if best_track_idx != -1:
                # Match found
                track = self.active_tracks[best_track_idx]
                track["bbox"] = det_bbox
                track["frame_idx"] = frame_idx
                track["max_score"] = max(track["max_score"], det.conf if det.conf else 1.0)

                tp = TrackPoint(
                    frame_idx=frame_idx,
                    cx=det.cx,
                    cy=det.cy,
                    w=det.w,
                    h=det.h,
                    conf=det.conf,
                    class_id=det.class_id,
                )
                updated_tracks.append(TrackedObject(track_id=track["id"], points=[tp]))
            else:
                unmatched_dets.append(det)

        # Handle unmatched active tracks (they die if they miss a frame in pure IOU tracker)
        self.active_tracks = [t for t in self.active_tracks if t["frame_idx"] == frame_idx]

        # Create new tracks from high confidence unmatched detections
        for det in unmatched_dets:
            if det.conf is None or det.conf >= self.sigma_h:
                track_id = self.next_id
                self.next_id += 1
                det_bbox = (det.cx, det.cy, det.w, det.h)

                self.active_tracks.append(
                    {
                        "id": track_id,
                        "bbox": det_bbox,
                        "max_score": det.conf if det.conf else 1.0,
                        "frame_idx": frame_idx,
                    }
                )

                tp = TrackPoint(
                    frame_idx=frame_idx,
                    cx=det.cx,
                    cy=det.cy,
                    w=det.w,
                    h=det.h,
                    conf=det.conf,
                    class_id=det.class_id,
                )
                updated_tracks.append(TrackedObject(track_id=track_id, points=[tp]))

        return updated_tracks

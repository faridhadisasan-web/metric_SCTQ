import numpy as np
import math
from typing import Dict, Any, List, Tuple
from sctq.types import FrameDetections, TrackedObject, TrackPoint
from sctq.tracking.tracker_base import BaseTrackerAdapter
from scipy.optimize import linear_sum_assignment
import cv2

def convert_bbox_to_z(bbox):
    w = float(bbox[2])
    h = float(bbox[3])
    cx = float(bbox[0])
    cy = float(bbox[1])
    s = w * h
    r = w / h if h > 0 else 0.0
    return np.array([cx, cy, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    x_val = float(np.asarray(x[0]).item())
    y_val = float(np.asarray(x[1]).item())
    s_val = float(np.asarray(x[2]).item())
    r_val = float(np.asarray(x[3]).item())

    if s_val <= 0:
         w, h = 0.1, 0.1
    else:
        w = np.sqrt(s_val * r_val)
        h = s_val / w if w > 0 else 0.1

    if score is None:
        return np.array([x_val, y_val, w, h]).reshape((1, 4))
    else:
        return np.array([x_val, y_val, w, h, float(score)]).reshape((1, 5))

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        # Define constant velocity model
        self.kf = cv2.KalmanFilter(7, 4)

        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 1.0

        z = convert_bbox_to_z(bbox).astype(np.float32)
        state_post = self.kf.statePost.copy()
        state_post[:4] = z
        self.kf.statePost = state_post

        state_pre = self.kf.statePre.copy()
        state_pre[:4] = z
        self.kf.statePre = state_pre

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1 # Initialize hits to 1 since creation IS an update
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.correct(convert_bbox_to_z(bbox).astype(np.float32))

    def predict(self):
        if (self.kf.statePost[6] + self.kf.statePost[2]) <= 0:
            state_post = self.kf.statePost.copy()
            state_post[6] = 0.0
            self.kf.statePost = state_post

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.statePost))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.statePost)


class InternalSORTTracker(BaseTrackerAdapter):
    """Simple internal implementation of SORT."""

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.max_age = params.get('max_age', 1)
        self.min_hits = params.get('min_hits', 3)
        self.iou_threshold = params.get('iou_threshold', 0.3)
        self.trackers = []
        self.frame_count = 0

    def reset(self) -> None:
        KalmanBoxTracker.count = 0
        self.trackers = []
        self.frame_count = 0

    def _xywh_to_xyxy(self, cx, cy, w, h):
        """Convert from (cx, cy, w, h) to (x1, y1, x2, y2) for IoU."""
        return [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]

    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        self.frame_count += 1
        frame_idx = frame_detections.frame_idx

        dets = []
        dets_xyxy = []
        for det in frame_detections.detections:
            dets.append([det.cx, det.cy, det.w, det.h, det.conf, det.class_id])
            dets_xyxy.append(self._xywh_to_xyxy(det.cx, det.cy, det.w, det.h))

        dets = np.array(dets) if dets else np.empty((0, 6))
        dets_xyxy = np.array(dets_xyxy) if dets_xyxy else np.empty((0, 4))

        # Predict
        trks_xyxy = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            # `KalmanBoxTracker` returns cx, cy, w, h in get_state and predict.
            trk_xyxy = self._xywh_to_xyxy(pos[0], pos[1], pos[2], pos[3])
            trks_xyxy[t, :] = [trk_xyxy[0], trk_xyxy[1], trk_xyxy[2], trk_xyxy[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks_xyxy = np.ma.compress_rows(np.ma.masked_invalid(trks_xyxy))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets_xyxy, trks_xyxy, self.iou_threshold)

        # Update
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # Output
        updated_tracks = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                tp = TrackPoint(frame_detections.frame_idx, float(d[0]), float(d[1]), float(d[2]), float(d[3]), 1.0, 0)
                updated_tracks.append(TrackedObject(trk.id, [tp]))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return updated_tracks

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                xA = max(det[0], trk[0])
                yA = max(det[1], trk[1])
                xB = min(det[2], trk[2])
                yB = min(det[3], trk[3])

                interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
                boxAArea = (det[2] - det[0]) * (det[3] - det[1])
                boxBArea = (trk[2] - trk[0]) * (trk[3] - trk[1])

                union = boxAArea + boxBArea - interArea
                if union > 0:
                    iou = interArea / float(union)
                else:
                    iou = 0.0

                iou_matrix[d, t] = iou

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = []
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= iou_threshold:
                        matched_indices.append([r, c])
                matched_indices = np.array(matched_indices)
        else:
            matched_indices = np.empty(shape=(0, 2), dtype=int)

        if len(matched_indices) == 0:
             matched_indices = np.empty((0,2), dtype=int)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)

import numpy as np
import math
from typing import Dict, Any, List, Tuple
from sctq.types import FrameDetections, TrackedObject, TrackPoint
from sctq.tracking.tracker_base import BaseTrackerAdapter
from scipy.optimize import linear_sum_assignment

class InternalCentroidTracker(BaseTrackerAdapter):
    """Simple internal implementation of Centroid Tracker."""

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.max_lost = params.get('max_lost', 5)
        self.active_tracks = {} # id: {'centroid': (cx, cy), 'lost': 0, 'bbox': (w,h,conf,cls)}
        self.next_id = 1

    def reset(self) -> None:
        self.active_tracks = {}
        self.next_id = 1

    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        frame_idx = frame_detections.frame_idx

        updated_tracks = []

        # Detections
        input_centroids = np.zeros((len(frame_detections.detections), 2), dtype="float")
        for i, det in enumerate(frame_detections.detections):
            input_centroids[i] = (det.cx, det.cy)

        if len(self.active_tracks) == 0:
            for i in range(0, len(input_centroids)):
                det = frame_detections.detections[i]
                track_id = self.next_id
                self.next_id += 1
                self.active_tracks[track_id] = {
                    'centroid': (det.cx, det.cy),
                    'lost': 0,
                    'bbox': (det.w, det.h, det.conf, det.class_id)
                }
                tp = TrackPoint(frame_idx, det.cx, det.cy, det.w, det.h, det.conf, det.class_id)
                updated_tracks.append(TrackedObject(track_id, [tp]))
            return updated_tracks

        if len(input_centroids) == 0:
            for objectID in list(self.active_tracks.keys()):
                self.active_tracks[objectID]['lost'] += 1
                if self.active_tracks[objectID]['lost'] > self.max_lost:
                    del self.active_tracks[objectID]
            return updated_tracks

        objectIDs = list(self.active_tracks.keys())
        objectCentroids = [v['centroid'] for v in self.active_tracks.values()]

        # Compute distances
        D = np.zeros((len(objectCentroids), len(input_centroids)))
        for i, pt1 in enumerate(objectCentroids):
            for j, pt2 in enumerate(input_centroids):
                D[i, j] = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(D)

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(row_ind, col_ind):
            # Threshold distance
            if D[row, col] > 100: # hardcoded threshold for simplicity
                continue

            objectID = objectIDs[row]
            det = frame_detections.detections[col]

            self.active_tracks[objectID]['centroid'] = (det.cx, det.cy)
            self.active_tracks[objectID]['lost'] = 0
            self.active_tracks[objectID]['bbox'] = (det.w, det.h, det.conf, det.class_id)

            usedRows.add(row)
            usedCols.add(col)

            tp = TrackPoint(frame_idx, det.cx, det.cy, det.w, det.h, det.conf, det.class_id)
            updated_tracks.append(TrackedObject(objectID, [tp]))

        # Lost objects
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.active_tracks[objectID]['lost'] += 1
            if self.active_tracks[objectID]['lost'] > self.max_lost:
                del self.active_tracks[objectID]

        # New objects
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        for col in unusedCols:
            det = frame_detections.detections[col]
            track_id = self.next_id
            self.next_id += 1
            self.active_tracks[track_id] = {
                'centroid': (det.cx, det.cy),
                'lost': 0,
                'bbox': (det.w, det.h, det.conf, det.class_id)
            }
            tp = TrackPoint(frame_idx, det.cx, det.cy, det.w, det.h, det.conf, det.class_id)
            updated_tracks.append(TrackedObject(track_id, [tp]))

        return updated_tracks


class InternalCentroidKFTracker(BaseTrackerAdapter):
    """Internal implementation of Centroid Tracker with Kalman Filter for state prediction."""

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.max_lost = params.get('max_lost', 5)
        self.active_tracks = {} # id: {'kf': KalmanFilter, 'lost': 0, 'bbox': (w,h,conf,cls)}
        self.next_id = 1

    def reset(self) -> None:
        self.active_tracks = {}
        self.next_id = 1

    def _create_kf(self, cx: float, cy: float):
        import cv2
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        return kf

    def update(self, frame_detections: FrameDetections) -> List[TrackedObject]:
        frame_idx = frame_detections.frame_idx
        updated_tracks = []

        input_centroids = np.zeros((len(frame_detections.detections), 2), dtype="float")
        for i, det in enumerate(frame_detections.detections):
            input_centroids[i] = (det.cx, det.cy)

        # Predict all existing KFs
        predicted_centroids = {}
        for tid, tdata in self.active_tracks.items():
            pred = tdata['kf'].predict()
            predicted_centroids[tid] = (float(pred[0][0]), float(pred[1][0]))

        if len(self.active_tracks) == 0:
            for i in range(len(input_centroids)):
                det = frame_detections.detections[i]
                track_id = self.next_id
                self.next_id += 1
                self.active_tracks[track_id] = {
                    'kf': self._create_kf(det.cx, det.cy),
                    'lost': 0,
                    'bbox': (det.w, det.h, det.conf, det.class_id)
                }
                tp = TrackPoint(frame_idx, det.cx, det.cy, det.w, det.h, det.conf, det.class_id)
                updated_tracks.append(TrackedObject(track_id, [tp]))
            return updated_tracks

        if len(input_centroids) == 0:
            for objectID in list(self.active_tracks.keys()):
                self.active_tracks[objectID]['lost'] += 1
                if self.active_tracks[objectID]['lost'] > self.max_lost:
                    del self.active_tracks[objectID]
            return updated_tracks

        objectIDs = list(predicted_centroids.keys())
        objectCentroids = list(predicted_centroids.values())

        D = np.zeros((len(objectCentroids), len(input_centroids)))
        for i, pt1 in enumerate(objectCentroids):
            for j, pt2 in enumerate(input_centroids):
                D[i, j] = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

        row_ind, col_ind = linear_sum_assignment(D)

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(row_ind, col_ind):
            if D[row, col] > 100:
                continue

            objectID = objectIDs[row]
            det = frame_detections.detections[col]

            # Correct KF
            meas = np.array([[np.float32(det.cx)], [np.float32(det.cy)]])
            self.active_tracks[objectID]['kf'].correct(meas)
            self.active_tracks[objectID]['lost'] = 0
            self.active_tracks[objectID]['bbox'] = (det.w, det.h, det.conf, det.class_id)

            usedRows.add(row)
            usedCols.add(col)

            # Key difference: KF tracker returns the SMOOTHED state, not the raw detection centroid
            smoothed_state = self.active_tracks[objectID]['kf'].statePost
            smooth_cx = float(smoothed_state[0][0])
            smooth_cy = float(smoothed_state[1][0])

            tp = TrackPoint(frame_idx, smooth_cx, smooth_cy, det.w, det.h, det.conf, det.class_id)
            updated_tracks.append(TrackedObject(objectID, [tp]))

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.active_tracks[objectID]['lost'] += 1
            if self.active_tracks[objectID]['lost'] > self.max_lost:
                del self.active_tracks[objectID]

        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        for col in unusedCols:
            det = frame_detections.detections[col]
            track_id = self.next_id
            self.next_id += 1
            self.active_tracks[track_id] = {
                'kf': self._create_kf(det.cx, det.cy),
                'lost': 0,
                'bbox': (det.w, det.h, det.conf, det.class_id)
            }
            tp = TrackPoint(frame_idx, det.cx, det.cy, det.w, det.h, det.conf, det.class_id)
            updated_tracks.append(TrackedObject(track_id, [tp]))

        return updated_tracks

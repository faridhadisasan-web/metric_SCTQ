import math
from typing import List, Tuple

import numpy as np


def compute_iou(
    boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.
    Boxes are represented as (cx, cy, w, h).
    """
    xA = max(boxA[0] - boxA[2] / 2, boxB[0] - boxB[2] / 2)
    yA = max(boxA[1] - boxA[3] / 2, boxB[1] - boxB[3] / 2)
    xB = min(boxA[0] + boxA[2] / 2, boxB[0] + boxB[2] / 2)
    yB = min(boxA[1] + boxA[3] / 2, boxB[1] + boxB[3] / 2)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = (
        interArea / float(boxAArea + boxBArea - interArea)
        if (boxAArea + boxBArea - interArea) > 0
        else 0.0
    )
    return iou


def distance_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extrapolate_position(
    p1: Tuple[float, float], p2: Tuple[float, float], steps: int
) -> Tuple[float, float]:
    """Extrapolate a position based on linear motion between two points over a number of steps."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (p2[0] + dx * steps, p2[1] + dy * steps)

from typing import Tuple

import cv2
import numpy as np


def draw_bbox(
    img: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    """Draw a bounding box given center, width, height."""
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return img

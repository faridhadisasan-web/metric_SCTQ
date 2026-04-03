import random
from typing import Any, Dict, List

import numpy as np

from sctq.types import DetectionRaw, FrameDetections


def apply_synthetic_perturbations(
    detections: List[FrameDetections], config: Dict[str, Any]
) -> List[FrameDetections]:
    """Apply synthetic noise to a set of ideal detections."""

    if not config.get("apply_synthetic_noise", False):
        return detections

    noise_types = config.get("noise_types", [])
    perturbed_dets = []

    for fd in detections:
        new_dets = []
        for det in fd.detections:
            # Random drop
            if "random_drop" in noise_types and random.random() < 0.1:
                continue

            # Jitter
            cx = det.cx
            cy = det.cy
            w = det.w
            h = det.h
            conf = det.conf

            if "jitter" in noise_types:
                cx += random.gauss(0, w * 0.05)
                cy += random.gauss(0, h * 0.05)
                w *= random.uniform(0.9, 1.1)
                h *= random.uniform(0.9, 1.1)
                conf = max(0.1, min(1.0, conf * random.uniform(0.8, 1.0)))

            new_dets.append(
                DetectionRaw(
                    frame_idx=det.frame_idx,
                    det_id=det.det_id,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    conf=conf,
                    class_id=det.class_id,
                )
            )

        perturbed_dets.append(FrameDetections(frame_idx=fd.frame_idx, detections=new_dets))

    return perturbed_dets

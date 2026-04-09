import random
from typing import Any, Dict, List

import cv2
import numpy as np

from sctq.types import DetectionRaw, FrameDetections


class ImageCorruptions:
    """Applies image-level corruptions to frames."""

    @staticmethod
    def gaussian_noise(image: np.ndarray, severity: int) -> np.ndarray:
        c = [0.04, 0.06, 0.08, 0.10, 0.12][severity - 1]

        image = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(size=image.shape, scale=c)
        noisy = np.clip(image + noise, 0, 1) * 255.0
        return noisy.astype(np.uint8)

    @staticmethod
    def salt_and_pepper(image: np.ndarray, severity: int) -> np.ndarray:
        c = [0.03, 0.05, 0.07, 0.09, 0.11][severity - 1]

        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = c
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out

    @staticmethod
    def gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
        c = [1, 2, 3, 4, 5][severity - 1]

        kernel_size = c * 2 + 1  # Needs to be odd
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred


class DetectionCorruptions:
    """Applies detection-level corruptions to bounding boxes."""

    @staticmethod
    def gaussian_jitter_center(frame_dets: FrameDetections, severity: int) -> FrameDetections:
        c = [0.05, 0.10, 0.15, 0.20, 0.25][severity - 1]

        new_dets = []
        for det in frame_dets.detections:
            noise_x = random.gauss(0, det.w * c)
            noise_y = random.gauss(0, det.h * c)
            new_dets.append(
                DetectionRaw(
                    frame_idx=det.frame_idx,
                    det_id=det.det_id,
                    cx=det.cx + noise_x,
                    cy=det.cy + noise_y,
                    w=det.w,
                    h=det.h,
                    conf=det.conf,
                    class_id=det.class_id,
                )
            )
        return FrameDetections(frame_idx=frame_dets.frame_idx, detections=new_dets)

    @staticmethod
    def random_drop(frame_dets: FrameDetections, severity: int) -> FrameDetections:
        c = [0.05, 0.10, 0.20, 0.30, 0.40][severity - 1]

        new_dets = [det for det in frame_dets.detections if random.random() > c]
        return FrameDetections(frame_idx=frame_dets.frame_idx, detections=new_dets)

    @staticmethod
    def false_positives(
        frame_dets: FrameDetections, severity: int, width: int = 1920, height: int = 1080
    ) -> FrameDetections:
        c = [1, 2, 3, 4, 5][severity - 1]  # Avg extra detections per frame

        new_dets = list(frame_dets.detections)
        num_extra = np.random.poisson(c)

        for i in range(num_extra):
            w = random.uniform(20, 100)
            h = random.uniform(50, 200)
            cx = random.uniform(w / 2, width - w / 2)
            cy = random.uniform(h / 2, height - h / 2)
            new_dets.append(
                DetectionRaw(
                    frame_idx=frame_dets.frame_idx,
                    det_id=10000 + i,  # Fake IDs
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    conf=random.uniform(0.1, 0.9),
                    class_id=0,
                )
            )

        return FrameDetections(frame_idx=frame_dets.frame_idx, detections=new_dets)

from pathlib import Path
from typing import Any, Dict

import numpy as np

from sctq.detection.detector_base import BaseDetector
from sctq.exceptions import TrackingError
from sctq.types import DetectionRaw, FrameDetections


class YOLODetector(BaseDetector):
    """Wrapper for Ultralytics YOLO models with offline-safe loading."""

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        requested_model = str(params.get("model", "models/yolo11n.pt"))
        self.conf_threshold = float(params.get("confidence_threshold", 0.3))
        self.imgsz = int(params.get("image_size", 640))
        self.classes = params.get("classes", [0])
        self.allow_download = bool(params.get("allow_download", False))
        self.model_path = self._resolve_model_path(requested_model)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise TrackingError("Ultralytics package is required to use YOLODetector.") from exc

        if not self.allow_download and not self.model_path.exists():
            raise TrackingError(
                "YOLO weights not found locally. "
                f"Expected model file at '{self.model_path}'. "
                "Place the .pt file in that location or set detector.allow_download=true explicitly."
            )

        model_arg = str(self.model_path) if self.model_path.exists() or not self.allow_download else requested_model
        try:
            self.model = YOLO(model_arg)
        except Exception as exc:
            raise TrackingError(f"Failed to load YOLO model '{model_arg}': {exc}") from exc

    def _resolve_model_path(self, requested_model: str) -> Path:
        raw = Path(requested_model)
        if raw.is_absolute():
            return raw
        candidates = [
            Path.cwd() / raw,
            Path.cwd() / "models" / raw.name,
            Path.cwd() / "model" / raw.name,
            Path(__file__).resolve().parents[3] / raw,
            Path(__file__).resolve().parents[3] / "models" / raw.name,
            Path(__file__).resolve().parents[3] / "model" / raw.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]

    def detect(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            classes=self.classes,
            verbose=False,
        )[0]

        dets = []
        boxes = results.boxes.cpu().numpy() if results.boxes is not None else []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            w = float(x2 - x1)
            h = float(y2 - y1)
            cx = float(x1 + w / 2)
            cy = float(y1 + h / 2)
            dets.append(
                DetectionRaw(
                    frame_idx=int(frame_idx),
                    det_id=int(i),
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    conf=conf,
                    class_id=cls_id,
                )
            )

        return FrameDetections(frame_idx=int(frame_idx), detections=dets)

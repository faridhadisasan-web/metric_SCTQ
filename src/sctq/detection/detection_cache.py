import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sctq.types import DetectionRaw, FrameDetections
from sctq.utils.io_utils import ensure_dir


class DetectionCache:
    """Manages caching of detections to disk to avoid re-running object detectors."""

    def __init__(self, cache_dir: str):
        self.cache_dir = ensure_dir(cache_dir)

    def _compute_config_hash(self, detector_cfg: Dict[str, Any]) -> str:
        cfg_str = json.dumps(detector_cfg or {}, sort_keys=True, default=str)
        return hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:8]

    def _get_cache_path(self, video_id: str, detector_name: str, detector_cfg: Dict[str, Any] = None) -> Path:
        if detector_cfg is not None:
            cfg_hash = self._compute_config_hash(detector_cfg)
            return self.cache_dir / f"{video_id}_{detector_name}_{cfg_hash}_detections.json"
        return self.cache_dir / f"{video_id}_{detector_name}_detections.json"

    def _frame_to_dict(self, fd: FrameDetections) -> Dict[str, Any]:
        return {
            "frame_idx": int(fd.frame_idx),
            "detections": [
                {
                    "frame_idx": int(fd.frame_idx),
                    "bbox_xyxy": [
                        float(det.cx - det.w / 2.0),
                        float(det.cy - det.h / 2.0),
                        float(det.cx + det.w / 2.0),
                        float(det.cy + det.h / 2.0),
                    ],
                    "conf": float(det.conf),
                    "class_id": 0 if det.class_id is None else int(det.class_id),
                    "det_id": int(det.det_id),
                }
                for det in fd.detections
            ],
        }

    def save(self, video_id: str, detector_name: str, all_detections: List[FrameDetections], detector_cfg: Dict[str, Any] = None) -> None:
        path = self._get_cache_path(video_id, detector_name, detector_cfg)
        data = [self._frame_to_dict(fd) for fd in all_detections]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, video_id: str, detector_name: str, detector_cfg: Dict[str, Any] = None) -> Optional[List[FrameDetections]]:
        path = self._get_cache_path(video_id, detector_name, detector_cfg)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return None

        all_detections: List[FrameDetections] = []
        for frame_data in data:
            frame_idx = int(frame_data["frame_idx"])
            detections = []
            for det in frame_data.get("detections", []):
                if "bbox_xyxy" in det:
                    x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                else:
                    cx = float(det["cx"])
                    cy = float(det["cy"])
                    w = float(det["w"])
                    h = float(det["h"])
                detections.append(
                    DetectionRaw(
                        frame_idx=frame_idx,
                        det_id=int(det.get("det_id", 0)),
                        cx=float(cx),
                        cy=float(cy),
                        w=float(w),
                        h=float(h),
                        conf=float(det.get("conf", 0.0)),
                        class_id=int(det.get("class_id", 0)) if det.get("class_id") is not None else None,
                    )
                )
            all_detections.append(FrameDetections(frame_idx=frame_idx, detections=detections))
        return all_detections

    def drop_corrupt_cache(self, video_id: str, detector_name: str, detector_cfg: Dict[str, Any] = None) -> None:
        path = self._get_cache_path(video_id, detector_name, detector_cfg)
        if path.exists():
            path.unlink()

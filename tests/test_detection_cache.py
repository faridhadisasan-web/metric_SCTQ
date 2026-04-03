from sctq.detection.detection_cache import DetectionCache
from sctq.types import DetectionRaw, FrameDetections


def test_detection_cache_roundtrip(tmp_path):
    cache = DetectionCache(tmp_path)
    fd = FrameDetections(
        frame_idx=0,
        detections=[DetectionRaw(frame_idx=0, det_id=1, cx=10.0, cy=20.0, w=30.0, h=40.0, conf=0.9, class_id=0)],
    )
    cache.save("vid", "yolo", [fd], {"model": "yolo11x.pt", "confidence_threshold": 0.3})
    loaded = cache.load("vid", "yolo", {"model": "yolo11x.pt", "confidence_threshold": 0.3})
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].frame_idx == 0
    assert loaded[0].detections[0].cx == 10.0
    assert loaded[0].detections[0].class_id == 0

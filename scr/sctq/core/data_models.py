from sctq.types import DetectionRaw, FrameDetections, TrackedObject, TrackPoint, TrackSet


def detection_to_trackpoint(det: DetectionRaw) -> TrackPoint:
    """Convert a raw detection to a track point."""
    return TrackPoint(
        frame_idx=det.frame_idx,
        cx=det.cx,
        cy=det.cy,
        w=det.w,
        h=det.h,
        conf=det.conf,
        class_id=det.class_id,
        embedding=det.embedding,
    )

from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np

from sctq.types import TrackSet
from sctq.utils.image_utils import draw_bbox
from sctq.utils.video_utils import read_video_frames


def render_annotated_video(
    video_path: Union[str, Path], trackset: TrackSet, output_path: Union[str, Path], fps: int = 30
) -> None:
    """Render a real video with tracking annotations."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    # Pre-process tracks into a frame-indexed dictionary for fast lookup
    tracks_by_frame: Dict[int, List[tuple]] = {}
    for track_id, track in trackset.tracks.items():
        # Assign a random color to each track based on ID
        np.random.seed(track_id)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        for pt in track.points:
            f_idx = pt.frame_idx
            if f_idx not in tracks_by_frame:
                tracks_by_frame[f_idx] = []
            tracks_by_frame[f_idx].append((track_id, pt, color))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in tracks_by_frame:
            for track_id, pt, color in tracks_by_frame[frame_idx]:
                label = f"ID: {track_id}"
                frame = draw_bbox(frame, pt.cx, pt.cy, pt.w, pt.h, color=color, label=label)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def render_synthetic_video(
    trackset: TrackSet,
    output_path: Union[str, Path],
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    total_frames: int = 300,
    background_color: tuple = (255, 255, 255),  # White default
) -> None:
    """Render a synthetic video with blank background and tracking annotations."""

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Pre-process tracks
    tracks_by_frame: Dict[int, List[tuple]] = {}
    for track_id, track in trackset.tracks.items():
        np.random.seed(track_id)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        for pt in track.points:
            f_idx = pt.frame_idx
            if f_idx not in tracks_by_frame:
                tracks_by_frame[f_idx] = []
            tracks_by_frame[f_idx].append((track_id, pt, color))

    for frame_idx in range(total_frames):
        # Create blank background frame
        frame = np.full((height, width, 3), background_color, dtype=np.uint8)

        if frame_idx in tracks_by_frame:
            for track_id, pt, color in tracks_by_frame[frame_idx]:
                label = f"ID: {track_id}"
                frame = draw_bbox(frame, pt.cx, pt.cy, pt.w, pt.h, color=color, label=label)

        out.write(frame)

    out.release()

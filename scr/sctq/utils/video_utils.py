from typing import Iterator, List, Optional, Tuple

import cv2


def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def sample_frame_indices(
    total_frames: int,
    max_frames: Optional[int] = None,
    sample_mode: str = "head",
    frame_stride: int = 1,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> List[int]:
    """
    Return selected absolute frame indices.

    Args:
        total_frames: Total number of frames in the whole video.
        max_frames: Maximum number of frames to return.
        sample_mode: 'head' or 'uniform'.
        frame_stride: Keep every Nth frame within the selected range.
        start_frame: Inclusive absolute start frame.
        end_frame: Inclusive absolute end frame.
    """
    if total_frames <= 0:
        return []

    if frame_stride < 1:
        frame_stride = 1

    start = 0 if start_frame is None else max(0, int(start_frame))
    end = total_frames - 1 if end_frame is None else min(total_frames - 1, int(end_frame))

    if end < start:
        return []

    base_indices = list(range(start, end + 1, frame_stride))
    if not base_indices:
        return []

    if max_frames is None or max_frames <= 0 or max_frames >= len(base_indices):
        return base_indices

    if sample_mode == "uniform":
        if max_frames == 1:
            return [base_indices[0]]
        positions = [round(i * (len(base_indices) - 1) / (max_frames - 1)) for i in range(max_frames)]
        return [base_indices[pos] for pos in positions]

    # Default: take the first max_frames frames from the selected range
    return base_indices[:max_frames]


def read_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    sample_mode: str = "head",
    frame_stride: int = 1,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Iterator[Tuple[int, "cv2.Mat"]]:
    """
    Yield (absolute_frame_index, frame) pairs from the selected region.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = sample_frame_indices(
        total_frames=total_frames,
        max_frames=max_frames,
        sample_mode=sample_mode,
        frame_stride=frame_stride,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if not selected:
        cap.release()
        return

    selected_set = set(selected)
    first_needed = min(selected)
    last_needed = max(selected)

    if first_needed > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_needed)

    frame_idx = first_needed
    try:
        while frame_idx <= last_needed:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in selected_set:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()

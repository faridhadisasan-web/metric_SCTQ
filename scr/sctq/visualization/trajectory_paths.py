from __future__ import annotations

import csv
import colorsys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


PointRow = Dict[str, float]
TrackRows = Dict[int, List[PointRow]]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_raw_tracks_csv(path: Path) -> TrackRows:
    tracks: TrackRows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = _to_int(row.get("track_id"), default=-1)
            if track_id < 0:
                continue
            parsed = {
                "frame_idx": _to_int(row.get("frame_idx"), default=0),
                "cx": _to_float(row.get("cx"), default=0.0),
                "cy": _to_float(row.get("cy"), default=0.0),
                "w": _to_float(row.get("w"), default=0.0),
                "h": _to_float(row.get("h"), default=0.0),
            }
            tracks.setdefault(track_id, []).append(parsed)

    for values in tracks.values():
        values.sort(key=lambda r: int(r["frame_idx"]))
    return tracks


def _stable_rgb(track_id: int) -> Tuple[float, float, float]:
    hue = ((track_id * 0.61803398875) % 1.0)
    return colorsys.hsv_to_rgb(hue, 0.75, 0.95)


def _infer_canvas_size(tracks: TrackRows, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if width is not None and height is not None:
        return int(width), int(height)

    xs: List[float] = []
    ys: List[float] = []
    for rows in tracks.values():
        for row in rows:
            xs.append(float(row["cx"]))
            ys.append(float(row["cy"]))

    max_x = max(xs) if xs else 1920.0
    max_y = max(ys) if ys else 1080.0
    inferred_width = int(max(640.0, max_x + 40.0))
    inferred_height = int(max(480.0, max_y + 40.0))
    return int(width or inferred_width), int(height or inferred_height)


def extract_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _select_tracks(tracks: TrackRows, min_track_length: int, top_k: int) -> List[Tuple[int, List[PointRow]]]:
    filtered = [(track_id, rows) for track_id, rows in tracks.items() if len(rows) >= min_track_length]
    filtered.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)
    if top_k > 0:
        filtered = filtered[:top_k]
    return filtered


def plot_trajectory_paths(
    raw_tracks_csv: Path,
    output_path: Path,
    *,
    video_path: Optional[Path] = None,
    background_frame_idx: Optional[int] = None,
    mode: str = "frame",
    top_k: int = 15,
    min_track_length: int = 5,
    label_tracks: bool = False,
    show_start_end: bool = True,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
    title: Optional[str] = None,
) -> Mapping[str, object]:
    tracks = read_raw_tracks_csv(raw_tracks_csv)
    selected = _select_tracks(tracks, min_track_length=min_track_length, top_k=top_k)
    if not selected:
        raise ValueError(f"No tracks with length >= {min_track_length} found in {raw_tracks_csv}")

    if background_frame_idx is None:
        all_frames = [int(row["frame_idx"]) for _, rows in selected for row in rows]
        background_frame_idx = int(round((min(all_frames) + max(all_frames)) / 2.0)) if all_frames else 0

    background = None
    width = canvas_width
    height = canvas_height
    if mode == "frame" and video_path is not None:
        background = extract_video_frame(video_path, background_frame_idx)
        height, width = int(background.shape[0]), int(background.shape[1])
    else:
        width, height = _infer_canvas_size(tracks, canvas_width, canvas_height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(6.0, float(width) / 250.0)
    fig_h = max(4.0, float(height) / 250.0)
    plt.figure(figsize=(fig_w, fig_h))

    if background is not None:
        plt.imshow(background)
    else:
        plt.xlim(0, width)
        plt.ylim(height, 0)

    for track_id, rows in selected:
        color = _stable_rgb(track_id)
        xs = [float(row["cx"]) for row in rows]
        ys = [float(row["cy"]) for row in rows]
        plt.plot(xs, ys, linewidth=2.2, color=color, alpha=0.95)

        if show_start_end and xs and ys:
            plt.scatter([xs[0]], [ys[0]], marker="o", s=28, color=[color])
            plt.scatter([xs[-1]], [ys[-1]], marker="x", s=40, color=[color])

        if label_tracks and xs and ys:
            plt.text(xs[-1], ys[-1], str(track_id), fontsize=8)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis() if background is not None else None
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    return {
        "raw_tracks_csv": str(raw_tracks_csv),
        "output_path": str(output_path),
        "video_path": str(video_path) if video_path is not None else "",
        "background_frame_idx": int(background_frame_idx),
        "mode": mode,
        "num_tracks_total": len(tracks),
        "num_tracks_plotted": len(selected),
        "top_k": int(top_k),
        "min_track_length": int(min_track_length),
    }


def infer_video_path_from_run_name(run_name: str, dataset_root: Optional[Path]) -> Optional[Path]:
    if dataset_root is None:
        return None

    parts = run_name.split("_")
    if len(parts) < 4:
        return None

    video_id = parts[2]
    candidate = dataset_root / video_id / f"{video_id}.mp4"
    if candidate.exists():
        return candidate
    return None


def batch_export_trajectory_plots(
    per_video_root: Path,
    output_root: Path,
    *,
    dataset_root: Optional[Path] = None,
    mode: str = "frame",
    include_corruptions: bool = False,
    top_k: int = 15,
    min_track_length: int = 5,
    label_tracks: bool = False,
) -> List[Mapping[str, object]]:
    manifests: List[Mapping[str, object]] = []

    run_dirs = sorted([p for p in per_video_root.iterdir() if p.is_dir()])
    for run_dir in run_dirs:
        if not include_corruptions and run_dir.name.startswith("corrupted_"):
            continue
        raw_dir = run_dir / "raw_tracks"
        if not raw_dir.exists():
            continue

        video_path = infer_video_path_from_run_name(run_dir.name, dataset_root)
        for csv_path in sorted(raw_dir.glob("*_raw_tracks.csv")):
            tracker_name = csv_path.stem.replace("_raw_tracks", "")
            output_path = output_root / run_dir.name / f"{tracker_name}_paths.png"
            title = f"{run_dir.name} | {tracker_name}"
            manifest = plot_trajectory_paths(
                csv_path,
                output_path,
                video_path=video_path,
                mode=mode,
                top_k=top_k,
                min_track_length=min_track_length,
                label_tracks=label_tracks,
                title=title,
            )
            manifests.append(manifest)

    return manifests

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from sctq.visualization.trajectory_paths import batch_export_trajectory_plots, plot_trajectory_paths


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _write_manifest_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready trajectory path visualizations from raw track CSV files.")
    parser.add_argument("--results-root", type=str, default="data/outputs/real_article", help="Root folder containing protocol_snapshot.json and per_video/.")
    parser.add_argument("--per-video-root", type=str, default=None, help="Optional override for the per_video directory.")
    parser.add_argument("--dataset-root", type=str, default=".", help="Dataset root containing video1/, video2/, video3/.")
    parser.add_argument("--raw-tracks", type=str, default=None, help="Optional path to one raw_tracks CSV. If omitted, all runs under per_video are processed.")
    parser.add_argument("--video", type=str, default=None, help="Optional video path for single-file mode.")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination folder for generated PNG files.")
    parser.add_argument("--mode", choices=["frame", "blank"], default="frame", help="Use a video frame background or a blank canvas.")
    parser.add_argument("--top-k", type=int, default=15, help="Maximum number of longest tracks to draw per figure.")
    parser.add_argument("--min-track-length", type=int, default=5, help="Minimum track length to draw.")
    parser.add_argument("--label-tracks", action="store_true", help="Draw track IDs near trajectory endpoints.")
    parser.add_argument("--include-corruptions", action="store_true", help="Also export figures for corrupted runs.")
    parser.add_argument("--background-frame", type=int, default=None, help="Optional background frame index for single-file mode.")
    parser.add_argument("--canvas-width", type=int, default=None)
    parser.add_argument("--canvas-height", type=int, default=None)
    args = parser.parse_args()

    if args.raw_tracks:
        raw_tracks = Path(args.raw_tracks)
        if not raw_tracks.exists():
            raise FileNotFoundError(f"Raw tracks CSV not found: {raw_tracks}")
        output_dir = Path(args.output_dir) if args.output_dir else raw_tracks.parent.parent / "trajectory_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{raw_tracks.stem.replace('_raw_tracks', '')}_paths.png"
        manifest = plot_trajectory_paths(
            raw_tracks,
            output_path,
            video_path=Path(args.video) if args.video else None,
            background_frame_idx=args.background_frame,
            mode=args.mode,
            top_k=args.top_k,
            min_track_length=args.min_track_length,
            label_tracks=args.label_tracks,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            title=raw_tracks.stem.replace("_raw_tracks", ""),
        )
        rows = [dict(manifest)]
        _write_manifest(output_dir / "trajectory_manifest.json", rows)
        _write_manifest_csv(output_dir / "trajectory_manifest.csv", rows)
        print(json.dumps(rows, indent=2))
        return

    results_root = Path(args.results_root)
    per_video_root = Path(args.per_video_root) if args.per_video_root else results_root / "per_video"
    if not per_video_root.exists():
        raise FileNotFoundError(f"per_video directory not found: {per_video_root}")

    output_dir = Path(args.output_dir) if args.output_dir else results_root / "trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = batch_export_trajectory_plots(
        per_video_root,
        output_dir,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        mode=args.mode,
        include_corruptions=args.include_corruptions,
        top_k=args.top_k,
        min_track_length=args.min_track_length,
        label_tracks=args.label_tracks,
    )
    _write_manifest(output_dir / "trajectory_manifest.json", rows)
    _write_manifest_csv(output_dir / "trajectory_manifest.csv", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

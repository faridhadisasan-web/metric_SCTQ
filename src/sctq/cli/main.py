import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from sctq.cli.run_ablation import main as ablation_main
from sctq.cli.run_benchmark import main as benchmark_main
from sctq.cli.run_corruption_suite import main as corruption_main
from sctq.cli.run_real_video import main as real_video_main
from sctq.cli.run_synthetic import main as synthetic_main
from sctq.cli.run_calibration import main as calibration_main
from sctq.config import ConfigLoader
from sctq.constants import DEFAULT_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(description="SCTQ Framework Master CLI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "real", "benchmark", "corruption", "ablation", "calibration"],
        required=True,
        help="Execution mode.",
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to default.yaml"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to input video file (required for real/corruption modes).",
    )
    parser.add_argument(
        "--det_file",
        type=str,
        default="",
        help="Path to input MOT detection file (required for benchmark mode).",
    )

    # We use parse_known_args so sub-scripts can parse their own
    args, unknown = parser.parse_known_args()

    # Reconstruct sys.argv for the target script
    sys.argv = [sys.argv[0]] + unknown
    if args.config:
        sys.argv.extend(["--config", args.config])

    if args.mode == "synthetic":
        synthetic_main()
    elif args.mode == "benchmark":
        if not args.det_file:
            print("Error: --det_file argument is required for benchmark mode.")
            sys.exit(1)
        sys.argv.extend(["--det_file", args.det_file])
        benchmark_main()
    elif args.mode == "real":
        if not args.video:
            print("Error: --video argument is required for real video mode.")
            sys.exit(1)
        sys.argv.extend(["--video", args.video])
        real_video_main()
    elif args.mode == "corruption":
        if not args.video:
            print("Error: --video argument is required for corruption mode.")
            sys.exit(1)
        sys.argv.extend(["--video", args.video])
        corruption_main()
    elif args.mode == "ablation":
        ablation_main()
    elif args.mode == "calibration":
        calibration_main()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

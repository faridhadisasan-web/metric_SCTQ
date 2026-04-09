import argparse
import json
import os
import sys
from pathlib import Path

from sctq.reporting.markdown_reporter import MarkdownReporter


def main():
    parser = argparse.ArgumentParser(description="Export Markdown Report from JSON output")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON summary file.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the markdown report."
    )
    parser.add_argument("--title", type=str, default="SCTQ Experiment Report", help="Report Title")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    reporter = MarkdownReporter(args.output_dir)
    reporter.generate_report(args.title, summaries)


if __name__ == "__main__":
    main()

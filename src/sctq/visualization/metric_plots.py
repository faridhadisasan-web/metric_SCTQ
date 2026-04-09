from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import matplotlib.pyplot as plt

from sctq.utils.io_utils import ensure_dir


class PlottingManager:
    """Manages the creation and saving of evaluation plots."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = ensure_dir(Path(output_dir) / "plots")

    def _save(self, filename: str):
        path = self.output_dir / f"{filename}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()

    def plot_track_length_histogram(self, lengths: List[int], filename: str, title: str = "Track Length Distribution"):
        if not lengths:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=min(30, max(5, len(set(lengths)))))
        plt.title(title)
        plt.xlabel("Track Length (frames)")
        plt.ylabel("Count")
        self._save(filename)

    def plot_sctq_components(self, summary: Dict[str, Any], filename: str, title: str = "SCTQ Components"):
        components = {
            "Persistence": float(summary.get("persistence_aggregate", 0.0)),
            "Dynamics": float(summary.get("dynamic_aggregate", 0.0)),
            "Fragmentation": float(summary.get("fragmentation_aggregate", 0.0)),
            "Consistency": float(summary.get("consistency_aggregate", 0.0)),
            "C_eff": float(summary.get("consistency_effective", summary.get("consistency_aggregate", 0.0))),
        }
        plt.figure(figsize=(8, 5))
        plt.bar(list(components.keys()), list(components.values()))
        plt.title(title)
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        self._save(filename)

    def plot_ranking_chart(self, rows: Sequence[Mapping[str, Any]], metric: str, filename: str, title: str):
        if not rows:
            return
        labels = [str(r.get("tracker_name", r.get("tracker", "?"))) for r in rows]
        values = [float(r.get(metric, 0.0) or 0.0) for r in rows]
        errs = [float(r.get(f"{metric}_std", 0.0) or 0.0) for r in rows]
        plt.figure(figsize=(9, 5))
        if any(errs):
            plt.bar(labels, values, yerr=errs, capsize=4)
        else:
            plt.bar(labels, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        self._save(filename)

    def plot_component_comparison(self, rows: Sequence[Mapping[str, Any]], filename: str, title: str):
        if not rows:
            return
        trackers = [str(r.get("tracker_name", r.get("tracker", "?"))) for r in rows]
        comps = [
            ("persistence_aggregate", "Persistence"),
            ("dynamic_aggregate", "Dynamics"),
            ("fragmentation_aggregate", "Fragmentation"),
            ("consistency_effective", "C_eff"),
        ]
        x = list(range(len(trackers)))
        width = 0.18
        plt.figure(figsize=(10, 5))
        for idx, (key, label) in enumerate(comps):
            values = [float(r.get(key, r.get(key.replace("_aggregate", ""), 0.0)) or 0.0) for r in rows]
            offsets = [i + (idx - 1.5) * width for i in x]
            plt.bar(offsets, values, width=width, label=label)
        plt.xticks(x, trackers, rotation=25, ha="right")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.title(title)
        self._save(filename)

    def plot_scatter(self, xs: Sequence[float], ys: Sequence[float], xlabel: str, ylabel: str, filename: str, title: str):
        if not xs or not ys:
            return
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        self._save(filename)

    def plot_ablation(self, pivot_rows: Sequence[Mapping[str, Any]], filename: str, title: str):
        if not pivot_rows:
            return
        tracker_names = [str(row.get("tracker_name", "?")) for row in pivot_rows]
        ablation_keys = [key for key in pivot_rows[0].keys() if key != "tracker_name"]
        x = list(range(len(tracker_names)))
        width = max(0.12, 0.8 / max(1, len(ablation_keys)))
        plt.figure(figsize=(10, 5))
        for idx, key in enumerate(ablation_keys):
            vals = [float(row.get(key) or 0.0) if str(row.get(key, "")).strip() != "" else 0.0 for row in pivot_rows]
            offsets = [i + (idx - (len(ablation_keys)-1)/2) * width for i in x]
            plt.bar(offsets, vals, width=width, label=key)
        plt.xticks(x, tracker_names, rotation=25, ha="right")
        plt.ylabel("SCTQ core")
        plt.title(title)
        plt.legend()
        self._save(filename)

    def plot_robustness_curves(self, rows: Sequence[Mapping[str, Any]], metric_key: str, filename: str, title: str):
        if not rows:
            return
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            tracker = str(row.get("tracker_name", row.get("tracker", "?")))
            grouped.setdefault(tracker, []).append(row)
        plt.figure(figsize=(9, 5))
        std_key = f"{metric_key}_std"
        for tracker, tracker_rows in grouped.items():
            tracker_rows = sorted(tracker_rows, key=lambda r: float(r.get("severity", 0) or 0.0))
            xs = [float(r.get("severity", 0) or 0.0) for r in tracker_rows]
            ys = [float(r.get(metric_key, 0.0) or 0.0) for r in tracker_rows]
            yerr = [float(r.get(std_key, 0.0) or 0.0) for r in tracker_rows]
            if any(yerr):
                plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=tracker)
            else:
                plt.plot(xs, ys, marker="o", label=tracker)
        plt.xlabel("Severity")
        plt.ylabel(metric_key.replace("_", " ").title())
        plt.title(title)
        plt.legend()
        self._save(filename)

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sctq.utils.tabular import format_mean_std


class MarkdownReporter:
    """Generates concise markdown summary reports with plot references."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _pick(self, summary: Dict[str, Any], *keys, default=None):
        for key in keys:
            if key in summary:
                return summary[key]
        return default

    def _format_metric(self, summary: Dict[str, Any], key: str, digits: int = 4) -> str:
        value = self._pick(summary, key, default=None)
        if value is None:
            return "-"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        std_value = self._pick(summary, f"{key}_std", default=None)
        if std_value is not None:
            try:
                return format_mean_std(value, float(std_value), digits=digits)
            except (TypeError, ValueError):
                pass
        return f"{value:.{digits}f}"

    def _tracker_table(self, summaries: Sequence[Dict[str, Any]]) -> str:
        lines = [
            "| Tracker | SCTQ Core | Persistence | Dynamics | Fragmentation | Consistency | C_eff | Stability | SCTQ Final | Tracks |",
            "|---------|-----------|------------|---------|--------------|------------|------|----------|-----------|------:|",
        ]
        for s in summaries:
            name = self._pick(s, "tracker_name", "tracker", default="Unknown")
            cons = self._pick(s, "consistency_aggregate", "consistency_clean", default=None)
            if cons is None:
                cons = 0.0
            stability = self._pick(s, "stability_score", default=1.0 if self._pick(s, "sctq_final", default=None) is not None else None)
            final_score = self._pick(s, "sctq_final", default=self._pick(s, "sctq_core", default=None))
            tracks = int(self._pick(s, "number_of_tracks", "number_of_clean_tracks", "num_runs", default=0) or 0)
            lines.append(
                "| {name} | {sctq} | {pers} | {dyn} | {frag} | {cons} | {cons_eff} | {stability} | {final_score} | {tracks} |".format(
                    name=name,
                    sctq=self._format_metric(s, "sctq_core"),
                    pers=self._format_metric(s, "persistence_aggregate"),
                    dyn=self._format_metric(s, "dynamic_aggregate"),
                    frag=self._format_metric(s, "fragmentation_aggregate"),
                    cons=self._format_metric({**s, "consistency_aggregate": cons}, "consistency_aggregate"),
                    cons_eff=self._format_metric(s, "consistency_effective"),
                    stability=self._format_metric({**s, "stability_score": stability}, "stability_score"),
                    final_score=self._format_metric({**s, "sctq_final": final_score}, "sctq_final"),
                    tracks=tracks,
                )
            )
        return "\n".join(lines)

    def generate_report(
        self,
        title: str,
        summaries: List[Dict[str, Any]],
        filename: str = "report.md",
        narrative: Optional[Iterable[str]] = None,
        plot_titles: Optional[Dict[str, str]] = None,
        extra_sections: Optional[List[Dict[str, str]]] = None,
        plots_subdir: str = "plots",
    ) -> None:
        path = self.output_dir / filename
        plots_dir = self.output_dir.parent / plots_subdir
        plot_titles = plot_titles or {}
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            if narrative:
                for paragraph in narrative:
                    f.write(f"{paragraph}\n\n")
            if summaries:
                f.write("## Summary Table\n\n")
                f.write(self._tracker_table(summaries))
                f.write("\n\n")
            if extra_sections:
                for section in extra_sections:
                    f.write(f"## {section.get('title', 'Section')}\n\n")
                    f.write(f"{section.get('body', '')}\n\n")
            if plots_dir.exists():
                plot_files = sorted(plots_dir.glob("*.png"))
                if plot_files:
                    f.write("## Figures\n\n")
                    for plot_file in plot_files:
                        title_text = plot_titles.get(plot_file.name, plot_file.stem.replace("_", " ").title())
                        rel_path = f"../{plots_subdir}/{plot_file.name}"
                        f.write(f"### {title_text}\n\n")
                        f.write(f"![{title_text}]({rel_path})\n\n")
        print(f"Markdown report generated: {path}")

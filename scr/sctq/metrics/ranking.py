from typing import Any, Dict, List

from scipy.stats import kendalltau, pearsonr, spearmanr

from sctq.utils.tabular import sort_rows


def compute_ranking(
    summaries: List[Dict[str, Any]], metric: str = "sctq_core", ascending: bool = False
) -> List[Dict[str, Any]]:
    if not summaries:
        return []
    ranked = sort_rows(summaries, metric, ascending=ascending)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def compute_correlation(sctq_scores: List[float], gt_metrics: List[float]) -> Dict[str, float]:
    if len(sctq_scores) < 2 or len(gt_metrics) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
    p_corr, _ = pearsonr(sctq_scores, gt_metrics)
    s_corr, _ = spearmanr(sctq_scores, gt_metrics)
    k_corr, _ = kendalltau(sctq_scores, gt_metrics)
    def safe(v):
        try:
            if v != v:
                return 0.0
        except Exception:
            return 0.0
        return float(v)
    return {"pearson": safe(p_corr), "spearman": safe(s_corr), "kendall": safe(k_corr)}

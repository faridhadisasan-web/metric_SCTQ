import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if _is_number(value):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
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


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def mean_std(values: Sequence[Any]) -> Tuple[float, float]:
    vals = [float(v) for v in values if _to_float(v) is not None]
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.pstdev(vals)


def group_rows(rows: Sequence[Mapping[str, Any]], group_key: str) -> Dict[Any, List[Mapping[str, Any]]]:
    grouped: Dict[Any, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get(group_key), []).append(row)
    return grouped


def aggregate_numeric(rows: Sequence[Mapping[str, Any]], group_key: str, excluded: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    rows = list(rows)
    if not rows:
        return []
    excluded_keys = set(excluded or []) | {group_key}
    all_keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    grouped = group_rows(rows, group_key)
    out: List[Dict[str, Any]] = []
    for group_value, group in grouped.items():
        result: Dict[str, Any] = {group_key: group_value, "num_runs": len(group)}
        for key in all_keys:
            if key in excluded_keys:
                continue
            numeric = [_to_float(r.get(key)) for r in group]
            numeric = [x for x in numeric if x is not None]
            if numeric:
                m, s = mean_std(numeric)
                result[key] = m
                result[f"{key}_std"] = s
            else:
                exemplar = next((r.get(key) for r in group if r.get(key) is not None), None)
                if exemplar is not None:
                    result[key] = exemplar
        out.append(result)
    return out


def sort_rows(rows: Sequence[Mapping[str, Any]], key: str, ascending: bool = False) -> List[Dict[str, Any]]:
    def sort_value(row: Mapping[str, Any]) -> Tuple[int, float, str]:
        val = row.get(key)
        num = _to_float(val)
        if num is not None:
            return (0, num, "")
        return (1, 0.0, str(val))

    return [dict(r) for r in sorted(rows, key=sort_value, reverse=not ascending)]


def pivot_mean(rows: Sequence[Mapping[str, Any]], index_key: str, column_key: str, value_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any], List[float]] = {}
    index_values = []
    column_values = []
    for row in rows:
        index = row.get(index_key)
        column = row.get(column_key)
        value = _to_float(row.get(value_key))
        if value is None:
            continue
        grouped.setdefault((index, column), []).append(value)
        if index not in index_values:
            index_values.append(index)
        if column not in column_values:
            column_values.append(column)

    out: List[Dict[str, Any]] = []
    for index in index_values:
        new_row: Dict[str, Any] = {index_key: index}
        for column in column_values:
            values = grouped.get((index, column), [])
            new_row[str(column)] = statistics.fmean(values) if values else ""
        out.append(new_row)
    return out


def linear_regression_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return float(numer / denom)


def trapezoid_area(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return float(area)


def format_mean_std(mean_value: float, std_value: float, digits: int = 4) -> str:
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


RESPONSE_TYPES = ("noul_like", "choice_like", "score_like")


def benchmark_summary(rows: list[dict[str, Any]], ncs_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the fixed benchmark headline table.

    The summary is intentionally small: argmax accuracy, pointwise probability
    fidelity, and pairwise primitive R^2 values. Deeper probability-recovery
    and semantic analyses stay in the detailed report tables.
    """

    valid_rows = [row for row in rows if not row.get("error")]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        base_model = base_model_name(str(row.get("model", "")))
        response_type = response_type_for_model(str(row.get("model", "")))
        grouped[(base_model, response_type)].append(row)
        by_base[base_model].append(row)

    r2_by_base = _primitive_r2_by_base(ncs_records)
    output = []
    for base_model in sorted(by_base):
        pairwise = r2_by_base.get(base_model, {})
        for response_type in RESPONSE_TYPES:
            summary_rows = grouped.get((base_model, response_type), [])
            if not summary_rows:
                continue
            output.append(_summary_row(base_model, response_type, summary_rows, pairwise))
        output.append(_summary_row(base_model, "overall", by_base[base_model], pairwise))
    return output


def response_type_for_model(model: str) -> str:
    lowered = model.lower()
    if "noul" in lowered:
        return "noul_like"
    if "score" in lowered:
        return "score_like"
    if "choice" in lowered or lowered == "jev":
        return "choice_like"
    return "overall"


def primitive_for_model(model: str) -> str | None:
    response_type = response_type_for_model(model)
    if response_type == "noul_like":
        return "noul"
    if response_type == "choice_like":
        return "choice"
    if response_type == "score_like":
        return "score"
    return None


def base_model_name(model: str) -> str:
    lowered = model.lower()
    for suffix in ("_noul", "-noul", ".noul", "_choice", "-choice", ".choice", "_score", "-score", ".score"):
        if lowered.endswith(suffix):
            return model[: -len(suffix)]
    if lowered == "jev":
        return model
    return model


def mean_tv_error(rows: list[dict[str, Any]]) -> float:
    tv_values = [float(row["metrics"]["tv"]) for row in rows if row.get("metrics", {}).get("tv") is not None]
    return statistics.fmean(tv_values) if tv_values else 0.0


def pointwise_probability_fidelity(rows: list[dict[str, Any]]) -> float:
    return 1.0 - mean_tv_error(rows)


def _summary_row(base_model: str, response_type: str, rows: list[dict[str, Any]], pairwise: dict[str, float | None]) -> dict[str, Any]:
    tv_error = mean_tv_error(rows)
    return {
        "model": base_model,
        "response_type": response_type,
        "n": len(rows),
        "argmax_accuracy": statistics.fmean(1.0 if row["argmax_correct"] else 0.0 for row in rows) if rows else 0.0,
        "pointwise_probability_fidelity": 1.0 - tv_error,
        "mean_tv_error": tv_error,
        "noul_choice_r2": pairwise.get("noul_choice_r2"),
        "noul_score_r2": pairwise.get("noul_score_r2"),
        "score_choice_r2": pairwise.get("score_choice_r2"),
    }


def _primitive_r2_by_base(records: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("base_model", "jev"))].append(record)
    output = {}
    for base_model, model_records in grouped.items():
        output[base_model] = {
            "noul_choice_r2": _squared_correlation([row["p_noul"] for row in model_records], [row["p_choice"] for row in model_records]),
            "noul_score_r2": _squared_correlation([row["p_noul"] for row in model_records], [row["mu_score"] for row in model_records]),
            "score_choice_r2": _squared_correlation([row["mu_score"] for row in model_records], [row["p_choice"] for row in model_records]),
        }
    return output


def _squared_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    ss_left = sum((value - mean_left) ** 2 for value in left)
    ss_right = sum((value - mean_right) ** 2 for value in right)
    if ss_left <= 0.0 or ss_right <= 0.0:
        return None
    cov = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    return (cov * cov) / (ss_left * ss_right)

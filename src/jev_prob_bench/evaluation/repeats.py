from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from jev_prob_bench.evaluation.metrics import argmax_correct, item_metrics
from jev_prob_bench.schemas import argmax, normalize_distribution


def aggregate_repeats(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method not in {"mean", "median"}:
        raise ValueError(f"unknown repeat aggregation method: {method}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            grouped[(row["model"], row["instance_id"])].append(row)

    aggregated = []
    for (_, _), items in sorted(grouped.items()):
        first = items[0]
        outcomes = list(first["gold_distribution"])
        pred = {}
        pred_std = {}
        for outcome in outcomes:
            values = [float(item["predicted_distribution"][outcome]) for item in items]
            pred[outcome] = statistics.fmean(values) if method == "mean" else statistics.median(values)
            pred_std[f"std_{outcome}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        pred = normalize_distribution(pred, outcomes)
        metrics = item_metrics(first["gold_distribution"], pred, outcomes)
        raw_response = _aggregate_raw_response(first["model"], items, method)
        row = {
            "model": first["model"],
            "model_version": first.get("model_version"),
            "instance_id": first["instance_id"],
            "latent_instance_id": first["latent_instance_id"],
            "family": first["family"],
            "difficulty": first.get("difficulty"),
            "representation": first["representation"],
            "probability_band": first.get("probability_band"),
            "generator_metadata": first.get("generator_metadata", {}),
            "gold_distribution": first["gold_distribution"],
            "predicted_distribution": pred,
            "metrics": metrics,
            "predicted_argmax": argmax(pred),
            "gold_argmax": first["gold_argmax"],
            "argmax_correct": argmax_correct(first["gold_distribution"], pred),
            "latency_ms": statistics.fmean(item.get("latency_ms", 0.0) for item in items),
            "repeat_index": method,
            "repeat_aggregation": method,
            "n_repeats": len(items),
            "prediction_std": pred_std,
            "error": None,
            "raw_response": raw_response,
            "api_metadata": {"repeat_aggregation": method, "n_repeats": len(items)},
        }
        aggregated.append(row)
    return aggregated


def _aggregate_raw_response(model: str, items: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if model != "jev_score":
        return {}
    probabilities_by_level: dict[str, list[float]] = {str(index): [] for index in range(10)}
    for item in items:
        answer = ((item.get("raw_response") or {}).get("answers") or {}).get("truth_degree") or {}
        probabilities = answer.get("probabilities") or answer.get("distribution") or {}
        for index in range(10):
            probabilities_by_level[str(index)].append(float(probabilities.get(str(index), 0.0)))
    aggregated = {}
    for index, values in probabilities_by_level.items():
        aggregated[index] = statistics.fmean(values) if method == "mean" else statistics.median(values)
    total = sum(aggregated.values())
    if total > 0:
        aggregated = {index: value / total for index, value in aggregated.items()}
    return {
        "answers": {
            "truth_degree": {
                "type": "score",
                "probabilities": aggregated,
                "aggregation": method,
            }
        }
    }


def repeat_count_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            grouped[(row["model"], row["instance_id"])].append(row)
    by_model: dict[str, list[int]] = defaultdict(list)
    for (model, _), items in grouped.items():
        by_model[model].append(len(items))
    return [
        {
            "model": model,
            "n_items": len(counts),
            "min_repeats": min(counts),
            "median_repeats": statistics.median(counts),
            "max_repeats": max(counts),
            "mean_repeats": statistics.fmean(counts),
        }
        for model, counts in sorted(by_model.items())
    ]

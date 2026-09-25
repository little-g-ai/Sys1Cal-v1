from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from jev_prob_bench.evaluation.metrics import summarize


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for key_tuple, items in sorted(grouped.items()):
        tv_summary = summarize([item["metrics"]["tv"] for item in items])
        record = {key: value for key, value in zip(keys, key_tuple)}
        record.update(
            {
                "n": len(items),
                "mean_tv": tv_summary["mean"],
                "median_tv": tv_summary["median"],
                "std_tv": tv_summary["std"],
                "p90_tv": tv_summary["p90"],
                "p95_tv": tv_summary["p95"],
                "max_tv": tv_summary["max"],
                "mean_mae": _mean_optional(item["metrics"].get("mae_binary") for item in items),
                "mean_brier_regret": statistics.fmean(item["metrics"]["brier_regret"] for item in items),
                "mean_kl": statistics.fmean(item["metrics"]["kl"] for item in items),
                "argmax_accuracy": sum(1 for item in items if item["argmax_correct"]) / len(items),
            }
        )
        output.append(record)
    return output


def _mean_optional(values):
    cleaned = [value for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


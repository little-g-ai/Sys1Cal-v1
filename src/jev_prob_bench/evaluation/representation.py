from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from typing import Any

from jev_prob_bench.evaluation.metrics import total_variation


def representation_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        grouped[(row["model"], row["latent_instance_id"])].append(row)
    output = []
    for (model, latent_id), items in grouped.items():
        if len(items) < 2:
            continue
        pairwise = []
        gold_errors = []
        binary_values = []
        for a, b in itertools.combinations(items, 2):
            outcomes = list(a["gold_distribution"])
            pairwise.append(total_variation(a["predicted_distribution"], b["predicted_distribution"], outcomes))
        for item in items:
            outcomes = list(item["gold_distribution"])
            gold_errors.append(total_variation(item["gold_distribution"], item["predicted_distribution"], outcomes))
            if len(outcomes) == 2:
                binary_values.append(item["predicted_distribution"][outcomes[0]])
        output.append(
            {
                "model": model,
                "latent_instance_id": latent_id,
                "mean_pairwise_tv": statistics.fmean(pairwise),
                "max_pairwise_tv": max(pairwise),
                "representation_variance": statistics.pvariance(binary_values) if len(binary_values) > 1 else None,
                "gold_relative_gap": max(gold_errors) - min(gold_errors),
            }
        )
    return output


def aggregate_representation_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    output = []
    for model, items in by_model.items():
        mean_pairwise = statistics.fmean(item["mean_pairwise_tv"] for item in items) if items else 0.0
        output.append(
            {
                "model": model,
                "mean_pairwise_tv": mean_pairwise,
                "mean_max_tv": statistics.fmean(item["max_pairwise_tv"] for item in items) if items else 0.0,
                "ris": 1.0 - mean_pairwise,
                "mean_gold_relative_gap": statistics.fmean(item["gold_relative_gap"] for item in items) if items else 0.0,
            }
        )
    return output


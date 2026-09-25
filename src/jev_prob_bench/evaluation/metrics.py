from __future__ import annotations

import math
import statistics

from jev_prob_bench.schemas import argmax, normalize_distribution


def total_variation(gold: dict[str, float], pred: dict[str, float], outcomes: list[str]) -> float:
    return 0.5 * sum(abs(gold[label] - pred[label]) for label in outcomes)


def brier_regret(gold: dict[str, float], pred: dict[str, float], outcomes: list[str]) -> float:
    return sum((pred[label] - gold[label]) ** 2 for label in outcomes)


def kl_divergence(gold: dict[str, float], pred: dict[str, float], outcomes: list[str], eps: float = 1e-12) -> float:
    clipped = {label: max(eps, pred[label]) for label in outcomes}
    clipped = normalize_distribution(clipped, outcomes)
    value = sum(gold[label] * math.log(gold[label] / clipped[label]) for label in outcomes if gold[label] > 0)
    return max(0.0, value)


def item_metrics(gold: dict[str, float], pred: dict[str, float], outcomes: list[str], eps: float = 1e-12) -> dict[str, float | None]:
    tv = total_variation(gold, pred, outcomes)
    brier = brier_regret(gold, pred, outcomes)
    kl = kl_divergence(gold, pred, outcomes, eps)
    if len(outcomes) == 2:
        first = outcomes[0]
        mae_binary = abs(pred[first] - gold[first])
        rmse_binary = mae_binary
    else:
        mae_binary = None
        rmse_binary = None
    return {"tv": tv, "mae_binary": mae_binary, "rmse_binary": rmse_binary, "brier_regret": brier, "kl": kl}


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": max(values),
    }


def argmax_correct(gold: dict[str, float], pred: dict[str, float]) -> bool:
    return argmax(gold) == argmax(pred)


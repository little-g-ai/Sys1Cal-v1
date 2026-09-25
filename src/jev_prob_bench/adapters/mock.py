from __future__ import annotations

from typing import Any

from jev_prob_bench.adapters.base import ModelAdapter


class MockAdapter(ModelAdapter):
    def __init__(self, name: str = "mock", strategy: str = "uniform", eps: float = 1e-6) -> None:
        super().__init__(name)
        self.strategy = strategy
        self.eps = eps

    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        self.last_metadata = {"raw_response": {"strategy": self.strategy}, "model_version": "mock-0.1"}
        if self.strategy == "oracle" and item is not None:
            return dict(item["gold_distribution"])
        if self.strategy == "argmax" and item is not None:
            winner = item["gold_argmax"]
            loser_mass = self.eps / max(1, len(outcomes) - 1)
            return {outcome: (1.0 - self.eps if outcome == winner else loser_mass) for outcome in outcomes}
        if self.strategy == "frequency":
            extracted = _extract_counts(state, outcomes)
            if extracted:
                total = sum(extracted.values())
                return {outcome: extracted.get(outcome, 0.0) / total for outcome in outcomes}
        return {outcome: 1.0 / len(outcomes) for outcome in outcomes}


def _extract_counts(state: dict[str, Any], outcomes: list[str]) -> dict[str, float] | None:
    for key in ("counts", "prior_counts"):
        if isinstance(state.get(key), dict) and all(label in state[key] for label in outcomes):
            return {label: float(state[key][label]) for label in outcomes}
    nested = state.get("sufficient_statistics")
    if isinstance(nested, dict):
        return _extract_counts(nested, outcomes)
    table = state.get("table")
    if isinstance(table, list):
        counts: dict[str, float] = {}
        for row in table:
            label = row.get("label")
            if label in outcomes and "count" in row:
                counts[label] = float(row["count"])
        if counts:
            return counts
    return None


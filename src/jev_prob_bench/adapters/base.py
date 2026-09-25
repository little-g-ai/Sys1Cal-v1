from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from jev_prob_bench.schemas import normalize_distribution


class ModelAdapter(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_metadata: dict[str, Any] = {}

    @abstractmethod
    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError

    def timed_predict(self, state: dict[str, Any], prompt: str, outcomes: list[str], item: dict[str, Any] | None = None) -> tuple[dict[str, float], float]:
        start = time.perf_counter()
        prediction = self.predict_distribution(state, prompt, outcomes, item=item)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return normalize_distribution(prediction, outcomes), latency_ms


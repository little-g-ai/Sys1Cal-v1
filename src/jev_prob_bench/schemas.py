from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TOLERANCE = 1e-10


@dataclass(frozen=True)
class LatentProblem:
    benchmark_version: str
    latent_instance_id: str
    family: str
    difficulty: int
    seed: int
    state: dict[str, Any]
    prompt: str
    outcomes: list[str]
    gold_distribution: dict[str, float]
    generator_metadata: dict[str, Any] = field(default_factory=dict)
    probability_band: str | None = None


@dataclass(frozen=True)
class BenchmarkItem:
    benchmark_version: str
    instance_id: str
    latent_instance_id: str
    family: str
    difficulty: int
    seed: int
    representation: str
    state: dict[str, Any]
    prompt: str
    outcomes: list[str]
    gold_distribution: dict[str, float]
    gold_argmax: str
    generator_metadata: dict[str, Any] = field(default_factory=dict)
    probability_band: str | None = None


def argmax(distribution: dict[str, float]) -> str:
    return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]


def normalize_distribution(
    distribution: dict[str, float], outcomes: list[str] | None = None
) -> dict[str, float]:
    labels = outcomes or list(distribution)
    cleaned = {label: max(0.0, float(distribution.get(label, 0.0))) for label in labels}
    total = sum(cleaned.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    if abs(total - 1.0) < 1e-12:
        return cleaned
    return {label: value / total for label, value in cleaned.items()}


def validate_distribution(distribution: dict[str, float], outcomes: list[str]) -> None:
    missing = set(outcomes) - set(distribution)
    if missing:
        raise ValueError(f"distribution is missing outcomes: {sorted(missing)}")
    if any(distribution[label] < -TOLERANCE for label in outcomes):
        raise ValueError("distribution contains a negative probability")
    total = sum(distribution[label] for label in outcomes)
    if abs(total - 1.0) >= TOLERANCE:
        raise ValueError(f"distribution sums to {total}, not 1")


def probability_band(p: float) -> str:
    if p == 0.0:
        return "0"
    if p == 1.0:
        return "1"
    bins = [
        (0.0, 0.01, "(0,0.01)"),
        (0.01, 0.10, "[0.01,0.10]"),
        (0.10, 0.30, "(0.10,0.30]"),
        (0.30, 0.45, "(0.30,0.45]"),
        (0.45, 0.55, "(0.45,0.55]"),
        (0.55, 0.70, "(0.55,0.70]"),
        (0.70, 0.90, "(0.70,0.90]"),
        (0.90, 0.99, "(0.90,0.99]"),
        (0.99, 1.0, "(0.99,1)"),
    ]
    for lo, hi, name in bins:
        if lo < p <= hi:
            return name
    return "outside"


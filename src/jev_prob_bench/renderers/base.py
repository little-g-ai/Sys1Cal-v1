from __future__ import annotations

import copy
import math
from typing import Any

from jev_prob_bench.schemas import LatentProblem


def render_problem(problem: LatentProblem, representation: str) -> tuple[dict[str, Any], str]:
    state = copy.deepcopy(problem.state)
    gold = problem.gold_distribution
    if representation == "direct":
        rendered = {"representation": "direct", "sufficient_statistics": state}
    elif representation == "counts":
        rendered = _counts(problem, scale=1000)
    elif representation == "scaled_counts":
        rendered = _counts(problem, scale=100000)
    elif representation == "ratio":
        rendered = _ratio(problem)
    elif representation == "table":
        rendered = {"representation": "table", "table": _table(problem), "state": state}
    elif representation == "nested":
        rendered = {"representation": "nested_state", "nested": state}
    elif representation == "prose":
        rendered = {"representation": "natural_language", "description": _prose(problem)}
    elif representation == "distractor":
        rendered = {"representation": "distractor", "sufficient_statistics": state, "irrelevant": _distractors(problem.seed)}
    else:
        raise ValueError(f"unknown representation: {representation}")
    prompt = f"{problem.prompt} Return a normalized probability for each of: {', '.join(problem.outcomes)}."
    return rendered, prompt


def _counts(problem: LatentProblem, scale: int) -> dict[str, Any]:
    if "counts" in problem.state:
        counts = problem.state["counts"]
        total = sum(counts.values())
        factor = max(1, scale // total)
        return {"representation": "counts", "counts": {k: v * factor for k, v in counts.items()}, "sampling": "uniform"}
    if problem.family == "bayes":
        priors = problem.state["priors"]
        prior_counts = {k: max(1, round(v * scale)) for k, v in priors.items()}
        likelihood_event = problem.state["likelihood_event"]
        event_counts = {k: round(prior_counts[k] * likelihood_event[k]) for k in priors}
        return {"representation": "bayes_counts", "prior_counts": prior_counts, "event_counts_by_hypothesis": event_counts, "observed_event": True}
    return _counts_from_gold(problem, scale)


def _counts_from_gold(problem: LatentProblem, scale: int) -> dict[str, Any]:
    counts = {k: max(0, round(v * scale)) for k, v in problem.gold_distribution.items()}
    diff = scale - sum(counts.values())
    first = next(iter(counts))
    counts[first] += diff
    return {"representation": "counts_equivalent", "counts": counts, "sampling": "uniform"}


def _ratio(problem: LatentProblem) -> dict[str, Any]:
    if len(problem.outcomes) == 2:
        a, b = problem.outcomes
        pa = problem.gold_distribution[a]
        left = round(pa * 100)
        right = 100 - left
        divisor = math.gcd(left, right) or 1
        return {"representation": "ratio", f"probability_ratio_{a}_to_{b}": f"{left // divisor}:{right // divisor}"}
    counts = {k: max(0, round(v * 100)) for k, v in problem.gold_distribution.items()}
    return {"representation": "ratio", "parts": counts}


def _table(problem: LatentProblem) -> list[dict[str, Any]]:
    if problem.family == "conditional":
        table = problem.state["contingency"]
        return [
            {"row": "A", "B": table["A"]["B"], "not_B": table["A"]["not_B"]},
            {"row": "not_A", "B": table["not_A"]["B"], "not_B": table["not_A"]["not_B"]},
        ]
    if problem.family == "bayes":
        return [
            {"hypothesis": h, "prior": problem.state["priors"][h], "P(E|hypothesis)": problem.state["likelihood_event"][h]}
            for h in problem.outcomes
        ]
    if problem.family == "sequential_bayes":
        return [
            {"step": i + 1, **likelihoods}
            for i, likelihoods in enumerate(problem.state["evidence_sequence"])
        ]
    if problem.family == "frequency" and "counts" in problem.state:
        return [{"label": k, "count": v} for k, v in problem.state["counts"].items()]
    return [{"field": key, "value": value} for key, value in problem.state.items()]


def _prose(problem: LatentProblem) -> str:
    if len(problem.outcomes) == 2:
        first = problem.outcomes[0]
        p = problem.gold_distribution[first]
        return f"The state fully determines the target distribution. The probability of {first} is {p:.12g}, and the remaining probability belongs to the other outcome. {problem.state}"
    pieces = ", ".join(f"{k}: {v:.12g}" for k, v in problem.gold_distribution.items())
    return f"The state fully determines this multiclass distribution: {pieces}. Source state: {problem.state}"


def _distractors(seed: int) -> dict[str, Any]:
    return {
        "container_temperature_c": round(12.5 + (seed % 17) * 0.7, 2),
        "operator_shift": ["morning", "swing", "night"][seed % 3],
        "batch_code": f"ZX-{seed % 997:03d}",
        "sensor_revision": f"rev-{1 + seed % 5}",
        "warehouse_lane": chr(65 + seed % 6),
    }


from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jev_prob_bench.schemas import argmax


SCORE_LEVELS = [
    "Completely false",
    "Very strongly false",
    "Strongly false",
    "Moderately false",
    "Slightly false",
    "Slightly true",
    "Moderately true",
    "Strongly true",
    "Very strongly true",
    "Completely true",
]

SCORE_NORMALIZED_LEVEL_VALUES = [round(index / 9, 10) for index in range(10)]


def make_parallel_primitive_dataset(input_path: str | Path, output_path: str | Path) -> Path:
    """Create one row per rendered binary instance with Noul/Choice/Score queries.

    The source dataset remains the existing rendered benchmark format. The
    output format makes the shared proposition explicit and places the three
    primitive-specific wordings under `queries`. Legacy binary fields are kept
    so the current evaluator can still score Noul and Choice runs.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if len(item["outcomes"]) != 2:
                continue
            parallel = make_parallel_primitive_item(item)
            target.write(json.dumps(parallel, sort_keys=True) + "\n")
    return output_path


def make_parallel_primitive_item(item: dict[str, Any]) -> dict[str, Any]:
    if len(item["outcomes"]) != 2:
        raise ValueError("parallel primitive items require binary source outcomes")

    positive, negative = item["outcomes"]
    p_true = float(item["gold_distribution"][positive])
    p_false = 1.0 - p_true
    proposition = _proposition_for_positive_outcome(item, positive, negative)
    instance_id = f"{item['instance_id']}__parallel_primitives"

    parallel = {
        "benchmark_version": item.get("benchmark_version"),
        "instance_id": instance_id,
        "latent_instance_id": item["latent_instance_id"],
        "family": item["family"],
        "difficulty": item.get("difficulty"),
        "seed": item.get("seed"),
        "representation": item["representation"],
        "state": copy.deepcopy(item["state"]),
        "proposition": proposition,
        "gold": {
            "probability_true": p_true,
            "probability_false": p_false,
        },
        "queries": {
            "noul": {
                "proposition": proposition,
            },
            "choice": {
                "question": "What is the truth status of the following proposition?",
                "proposition": proposition,
                "options": [
                    "False",
                    "True",
                ],
            },
            "score": {
                "question": "To what degree is the following proposition true?",
                "proposition": proposition,
                "levels": list(SCORE_LEVELS),
                "normalized_level_values": list(SCORE_NORMALIZED_LEVEL_VALUES),
            },
        },
        "generator_metadata": _metadata(item, positive, negative),
        "probability_band": item.get("probability_band"),
        # Compatibility with the existing evaluator. The parallel schema's
        # normative gold remains `gold`; these fields let older metrics score a
        # binary true/false distribution without splitting the dataset rows.
        "prompt": proposition,
        "outcomes": ["True", "False"],
        "gold_distribution": {
            "True": p_true,
            "False": p_false,
        },
    }
    parallel["gold_argmax"] = argmax(parallel["gold_distribution"])
    return parallel


def _metadata(item: dict[str, Any], positive: str, negative: str) -> dict[str, Any]:
    metadata = copy.deepcopy(item.get("generator_metadata", {}))
    metadata["parallel_primitives"] = {
        "source_instance_id": item["instance_id"],
        "source_representation": item["representation"],
        "source_prompt": item.get("prompt"),
        "source_outcomes": item["outcomes"],
        "positive_source_outcome": positive,
        "negative_source_outcome": negative,
        "score_levels": 10,
        "score_value_rule": "z_j = j / 9 for j = 0,...,9",
    }
    return metadata


def _proposition_for_positive_outcome(item: dict[str, Any], positive: str, negative: str) -> str:
    if {positive, negative} == {"A", "not_A"}:
        return "Event A is true."
    if {positive, negative} == {"H", "not_H"}:
        return "The latent hypothesis H is true."
    if {positive, negative} == {"target", "not_target"}:
        return "The target compound event is true."
    if _is_complement_label(negative, positive):
        return f"The proposition labelled {positive} is true."
    if item["family"] == "frequency":
        return f"The sampled object is a {positive}."
    return f"The realized outcome is {positive}."


def _is_complement_label(label: str, positive: str) -> bool:
    return label in {f"not_{positive}", f"not {positive}", f"non_{positive}"}

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def make_choice_explicit_dataset(input_path: str | Path, output_path: str | Path) -> Path:
    """Create a dataset variant with explicit Choice partition semantics.

    This preserves gold distributions and instance IDs except for adding a
    variant suffix. It is intended for comparing implicit Choice prompts against
    prompts that explicitly state the offered outcomes are mutually exclusive
    and exhaustive.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            explicit = make_choice_explicit_item(item)
            target.write(json.dumps(explicit, sort_keys=True) + "\n")
    return output_path


def make_choice_explicit_item(item: dict[str, Any]) -> dict[str, Any]:
    explicit = copy.deepcopy(item)
    outcomes = explicit["outcomes"]
    explicit["instance_id"] = f"{explicit['instance_id']}__choice_explicit"
    explicit["representation"] = f"{explicit['representation']}__choice_explicit"
    explicit["state"] = _explicit_state(explicit["state"], outcomes)
    explicit["prompt"] = _explicit_prompt(explicit["prompt"], outcomes)
    metadata = explicit.setdefault("generator_metadata", {})
    metadata["choice_explicit_variant"] = {
        "source_instance_id": item["instance_id"],
        "source_representation": item["representation"],
        "semantics": "outcomes are mutually exclusive and exhaustive",
    }
    return explicit


def _explicit_state(state: dict[str, Any], outcomes: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(state)
    updated["choice_option_semantics"] = {
        "mutually_exclusive": True,
        "exhaustive": True,
        "exactly_one_outcome_is_true": True,
        "outcomes": outcomes,
        "normalization": "The probabilities assigned to these outcomes must sum to 1.",
    }
    if len(outcomes) == 2:
        updated["choice_option_semantics"]["exact_complements"] = {
            outcomes[0]: outcomes[1],
            outcomes[1]: outcomes[0],
        }
    return updated


def _explicit_prompt(prompt: str, outcomes: list[str]) -> str:
    options = ", ".join(outcomes)
    return (
        f"{prompt} In this Choice formulation, the offered options {{{options}}} are an exact "
        "mutually exclusive and exhaustive partition of the outcome space. Exactly one option is true. "
        "Interpret each Choice probability as the absolute probability of that option under the supplied "
        "state, not as a separate confidence score."
    )


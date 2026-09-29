#!/usr/bin/env python3
"""Evaluate Sys1Cal-v1 predictions against an exact-probability JSONL split."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def normalize(distribution: dict[str, Any], outcomes: list[str]) -> dict[str, float]:
    cleaned = {label: max(0.0, float(distribution.get(label, 0.0))) for label in outcomes}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {label: 1.0 / len(outcomes) for label in outcomes}
    return {label: value / total for label, value in cleaned.items()}


def argmax(distribution: dict[str, float]) -> str:
    return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]


def total_variation(
    gold: dict[str, float], prediction: dict[str, float], outcomes: list[str]
) -> float:
    return 0.5 * sum(abs(gold[label] - prediction[label]) for label in outcomes)


def brier_regret(
    gold: dict[str, float], prediction: dict[str, float], outcomes: list[str]
) -> float:
    return sum((prediction[label] - gold[label]) ** 2 for label in outcomes)


def extract_distribution(row: dict[str, Any], item: dict[str, Any]) -> dict[str, float]:
    outcomes = item["outcomes"]
    for key in ("predicted_distribution", "distribution", "prediction", "probabilities"):
        value = row.get(key)
        if isinstance(value, dict):
            return normalize(map_labels(value, outcomes), outcomes)

    if "probability_true" in row:
        p_true = float(row["probability_true"])
        return normalize({"True": p_true, "False": 1.0 - p_true}, outcomes)

    if "p_true" in row:
        p_true = float(row["p_true"])
        return normalize({"True": p_true, "False": 1.0 - p_true}, outcomes)

    raise ValueError(
        "prediction row must contain predicted_distribution, distribution, "
        "prediction, probabilities, probability_true, or p_true"
    )


def map_labels(distribution: dict[str, Any], outcomes: list[str]) -> dict[str, Any]:
    if all(label in distribution for label in outcomes):
        return distribution
    lower_to_label = {str(label).lower(): label for label in outcomes}
    mapped = {}
    for raw_label, value in distribution.items():
        label = lower_to_label.get(str(raw_label).lower())
        if label is not None:
            mapped[label] = value
    return mapped


def prediction_group(row: dict[str, Any]) -> str:
    model = row.get("model") or row.get("model_name") or "submission"
    response_type = row.get("response_type")
    api_metadata = row.get("api_metadata")
    if response_type is None and isinstance(api_metadata, dict):
        response_type = api_metadata.get("question_type")
    return f"{model}:{response_type}" if response_type else str(model)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": max(values),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tv_values = [row["tv"] for row in rows]
    brier_values = [row["brier_regret"] for row in rows]
    accuracy_values = [1.0 if row["argmax_correct"] else 0.0 for row in rows]
    return {
        "n_predictions": len(rows),
        "accuracy": statistics.fmean(accuracy_values) if accuracy_values else 0.0,
        "pointwise_probability_fidelity": 1.0 - statistics.fmean(tv_values)
        if tv_values
        else 0.0,
        "tv": summarize(tv_values),
        "brier_regret": summarize(brier_values),
    }


def evaluate(dataset_path: Path, predictions_path: Path, strict: bool) -> dict[str, Any]:
    items = {row["instance_id"]: row for row in read_jsonl(dataset_path)}
    prediction_rows = read_jsonl(predictions_path)
    scored_rows = []
    errors = []

    for index, row in enumerate(prediction_rows, start=1):
        instance_id = row.get("instance_id")
        if instance_id not in items:
            errors.append(f"prediction row {index}: unknown instance_id {instance_id!r}")
            continue
        item = items[instance_id]
        try:
            pred = extract_distribution(row, item)
        except Exception as exc:
            errors.append(f"prediction row {index}: {exc}")
            continue
        gold = {key: float(value) for key, value in item["gold_distribution"].items()}
        outcomes = item["outcomes"]
        tv = total_variation(gold, pred, outcomes)
        scored_rows.append(
            {
                "group": prediction_group(row),
                "instance_id": instance_id,
                "latent_instance_id": item["latent_instance_id"],
                "family": item["family"],
                "representation": item["representation"],
                "probability_band": item.get("probability_band"),
                "tv": tv,
                "pointwise_probability_fidelity": 1.0 - tv,
                "brier_regret": brier_regret(gold, pred, outcomes),
                "gold_argmax": item["gold_argmax"],
                "predicted_argmax": argmax(pred),
                "argmax_correct": argmax(gold) == argmax(pred),
            }
        )

    covered = {row["instance_id"] for row in scored_rows}
    missing = sorted(set(items) - covered)
    if strict and (errors or missing):
        for error in errors[:20]:
            print(error, file=sys.stderr)
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more prediction errors", file=sys.stderr)
        if missing:
            print(f"missing predictions for {len(missing)} dataset rows", file=sys.stderr)
        raise SystemExit(1)

    by_group = defaultdict(list)
    by_family = defaultdict(list)
    by_representation = defaultdict(list)
    by_probability_band = defaultdict(list)
    for row in scored_rows:
        by_group[row["group"]].append(row)
        by_family[(row["group"], row["family"])].append(row)
        by_representation[(row["group"], row["representation"])].append(row)
        by_probability_band[(row["group"], row["probability_band"])].append(row)

    return {
        "dataset_path": str(dataset_path),
        "predictions_path": str(predictions_path),
        "n_dataset_items": len(items),
        "n_prediction_rows": len(prediction_rows),
        "n_scored_rows": len(scored_rows),
        "n_missing_items": len(missing),
        "n_errors": len(errors),
        "errors": errors,
        "overall": aggregate(scored_rows),
        "by_group": {str(key): aggregate(rows) for key, rows in sorted(by_group.items())},
        "by_family": {
            f"{key[0]} / {key[1]}": aggregate(rows)
            for key, rows in sorted(by_family.items())
        },
        "by_representation": {
            f"{key[0]} / {key[1]}": aggregate(rows)
            for key, rows in sorted(by_representation.items())
        },
        "by_probability_band": {
            f"{key[0]} / {key[1]}": aggregate(rows)
            for key, rows in sorted(by_probability_band.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/v0.1.0_tiny_cleanvars_parallel_primitives.jsonl"),
        help="Sys1Cal-v1 JSONL file.",
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL.")
    parser.add_argument("--output", type=Path, help="Optional path for summary JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unknown instance IDs, malformed predictions, or missing items.",
    )
    args = parser.parse_args()

    summary = evaluate(args.dataset, args.predictions, args.strict)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

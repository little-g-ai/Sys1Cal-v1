from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_prob_bench.adapters import build_adapter
from jev_prob_bench.evaluation.metrics import argmax_correct, item_metrics
from jev_prob_bench.schemas import argmax, normalize_distribution


def run_benchmark(
    dataset_path: str | Path,
    model_configs: list[dict[str, Any]],
    output_dir: str | Path,
    repeats: int = 1,
    resume: bool = True,
    kl_epsilon: float = 1e-12,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    completed = _completed_keys(result_path) if resume else set()
    adapters = [build_adapter(config) for config in model_configs]
    items = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    with result_path.open("a", encoding="utf-8") as results:
        for adapter in adapters:
            if hasattr(adapter, "predict_batch"):
                for repeat_index in range(repeats):
                    pending = [
                        item
                        for item in items
                        if (adapter.name, item["instance_id"], repeat_index) not in completed
                    ]
                    for row in _evaluate_batch(adapter, pending, repeat_index, kl_epsilon):
                        results.write(json.dumps(row, sort_keys=True) + "\n")
                        results.flush()
                continue
            for item in items:
                for repeat_index in range(repeats):
                    key = (adapter.name, item["instance_id"], repeat_index)
                    if key in completed:
                        continue
                    row = _evaluate_item(adapter, item, repeat_index, kl_epsilon)
                    results.write(json.dumps(row, sort_keys=True) + "\n")
                    results.flush()
    return result_path


def _evaluate_batch(adapter, items: list[dict[str, Any]], repeat_index: int, kl_epsilon: float) -> list[dict[str, Any]]:
    if not items:
        return []
    try:
        predictions = adapter.predict_batch(items)
    except Exception as exc:
        return [_error_row(adapter.name, item, repeat_index, repr(exc)) for item in items]
    rows = []
    for item in items:
        result = predictions.get(item["instance_id"])
        if result is None:
            rows.append(_error_row(adapter.name, item, repeat_index, "missing batch prediction"))
            continue
        pred = normalize_distribution(result["distribution"], item["outcomes"])
        metrics = item_metrics(item["gold_distribution"], pred, item["outcomes"], kl_epsilon)
        metadata = result.get("metadata", {})
        rows.append(
            _result_row(
                adapter.name,
                item,
                pred,
                metrics,
                argmax(pred),
                argmax_correct(item["gold_distribution"], pred),
                result.get("latency_ms", 0.0),
                repeat_index,
                None,
                metadata,
            )
        )
    return rows


def _evaluate_item(adapter, item: dict[str, Any], repeat_index: int, kl_epsilon: float) -> dict[str, Any]:
    try:
        pred, latency_ms = adapter.timed_predict(item["state"], item["prompt"], item["outcomes"], item=item)
        pred = normalize_distribution(pred, item["outcomes"])
        metrics = item_metrics(item["gold_distribution"], pred, item["outcomes"], kl_epsilon)
        metadata = getattr(adapter, "last_metadata", {}) or {}
        return _result_row(
            adapter.name,
            item,
            pred,
            metrics,
            argmax(pred),
            argmax_correct(item["gold_distribution"], pred),
            latency_ms,
            repeat_index,
            None,
            metadata,
        )
    except Exception as exc:
        return _error_row(adapter.name, item, repeat_index, repr(exc))


def _error_row(model_name: str, item: dict[str, Any], repeat_index: int, error: str) -> dict[str, Any]:
    return _result_row(
        model_name,
        item,
        {outcome: 0.0 for outcome in item["outcomes"]},
        {},
        None,
        False,
        0.0,
        repeat_index,
        error,
        {},
    )


def _result_row(
    model_name: str,
    item: dict[str, Any],
    pred: dict[str, float],
    metrics: dict[str, Any],
    predicted_argmax: str | None,
    is_argmax_correct: bool,
    latency_ms: float,
    repeat_index: int,
    error: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "model_version": metadata.get("model_version"),
        "instance_id": item["instance_id"],
        "latent_instance_id": item["latent_instance_id"],
        "family": item["family"],
        "difficulty": item["difficulty"],
        "representation": item["representation"],
        "probability_band": item.get("probability_band"),
        "generator_metadata": item.get("generator_metadata", {}),
        "gold_distribution": item["gold_distribution"],
        "predicted_distribution": pred,
        "metrics": metrics,
        "predicted_argmax": predicted_argmax,
        "gold_argmax": item["gold_argmax"],
        "argmax_correct": is_argmax_correct,
        "latency_ms": latency_ms,
        "repeat_index": repeat_index,
        "error": error,
        "raw_response": metadata.get("raw_response", {}),
        "api_metadata": metadata.get("api_metadata", {}),
    }


def _completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["model"], row["instance_id"], row.get("repeat_index", 0)))
    return keys


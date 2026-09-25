from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from jev_prob_bench.adapters.base import ModelAdapter


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class SemIfAdapter(ModelAdapter):
    """Adapter for TheoLeeCJ/SemIf via the `semif-score` CLI.

    SemIf consumes JSONL rows shaped as:
    {id, state, question, options:[{id, description}]}
    and writes rows containing option_ids and aligned probabilities.
    """

    def __init__(
        self,
        name: str = "semif",
        command: str | None = None,
        mode: str = "direct",
        model: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        timeout: float | None = None,
        extra_args: list[str] | None = None,
        keep_files: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name)
        self.command = command or os.environ.get("SEMIF_SCORE_BIN", "semif-score")
        self.mode = mode
        self.model = model
        self.revision = revision
        self.timeout = timeout
        self.extra_args = extra_args or []
        self.keep_files = keep_files
        self.env = env or {}

    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        benchmark_item = {
            "instance_id": (item or {}).get("instance_id", "benchmark-item"),
            "state": (item or {}).get("state", state),
            "prompt": (item or {}).get("prompt", prompt),
            "outcomes": (item or {}).get("outcomes", outcomes),
        }
        result = self.predict_batch([benchmark_item])[benchmark_item["instance_id"]]
        self.last_metadata = result["metadata"]
        return result["distribution"]

    def predict_batch(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not items:
            return {}
        with tempfile.TemporaryDirectory(prefix="jev-prob-bench-semif-") as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "input.jsonl"
            output_path = tmpdir / "output.jsonl"
            decisions = [_decision_from_item(item) for item in items]
            input_path.write_text(
                "".join(json.dumps(decision, sort_keys=True) + "\n" for decision in decisions),
                encoding="utf-8",
            )
            command = [
                self.command,
                "--mode",
                self.mode,
                "--model",
                self.model,
                "--revision",
                self.revision,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *self.extra_args,
            ]
            start = time.perf_counter()
            env = os.environ.copy()
            env.update(self.env)
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
                env=env,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if completed.returncode != 0:
                raise RuntimeError(f"SemIf command failed with code {completed.returncode}: {completed.stderr.strip()}")
            raw_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            rows_by_id = {str(row.get("id")): row for row in raw_rows}
            outputs: dict[str, dict[str, Any]] = {}
            average_latency_ms = elapsed_ms / max(1, len(items))
            for item in items:
                item_id = item["instance_id"]
                raw = rows_by_id.get(item_id)
                if raw is None:
                    raise RuntimeError(f"SemIf output is missing result for {item_id}")
                distribution = _parse_semif_distribution(raw, item["outcomes"])
                row_latency_ms = float(raw.get("total_seconds", 0.0)) * 1000.0 or average_latency_ms
                metadata = {
                    "raw_response": raw,
                    "parsed_distribution": distribution,
                    "model_version": _model_version(raw, self.model, self.revision),
                    "api_metadata": {
                        "command": command,
                        "stderr": completed.stderr,
                        "stdout": completed.stdout,
                        "mode": self.mode,
                        "batch_size": len(items),
                        "batch_elapsed_ms": elapsed_ms,
                    },
                }
                if self.keep_files:
                    metadata["api_metadata"]["input_path"] = str(input_path)
                    metadata["api_metadata"]["output_path"] = str(output_path)
                outputs[item_id] = {
                    "distribution": distribution,
                    "latency_ms": row_latency_ms,
                    "metadata": metadata,
                }
            return outputs


def _decision_from_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["instance_id"],
        "state": item["state"],
        "question": item["prompt"],
        "options": [
            {
                "id": outcome,
                "description": _option_description(outcome, item["prompt"]),
            }
            for outcome in item["outcomes"]
        ],
    }


def _option_description(outcome: str, prompt: str) -> str:
    return f"The correct probability mass belongs to benchmark outcome `{outcome}` for this question: {prompt}"


def _parse_semif_distribution(raw: dict[str, Any], outcomes: list[str]) -> dict[str, float]:
    if isinstance(raw.get("probabilities"), dict):
        return {outcome: float(raw["probabilities"].get(outcome, 0.0)) for outcome in outcomes}
    option_ids = raw.get("option_ids")
    probabilities = raw.get("probabilities")
    if not isinstance(option_ids, list) or not isinstance(probabilities, list):
        raise RuntimeError(f"SemIf output is missing option_ids/probabilities: {raw}")
    aligned = {str(option_id): float(probability) for option_id, probability in zip(option_ids, probabilities)}
    return {outcome: aligned.get(outcome, 0.0) for outcome in outcomes}


def _model_version(raw: dict[str, Any], model: str, revision: str) -> str:
    model_info = raw.get("model")
    if isinstance(model_info, dict):
        source = model_info.get("source", model)
        rev = model_info.get("revision", revision)
        return f"{source}@{rev}"
    return f"{model}@{revision}"

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import urllib.request

from jev_prob_bench.adapters.jev import _load_dotenv
from jev_prob_bench.evaluation.metrics import argmax_correct, item_metrics
from jev_prob_bench.schemas import argmax, normalize_distribution


MODEL_BY_PRIMITIVE = {
    "noul": "jev_noul",
    "choice": "jev_choice",
    "score": "jev_score",
}


def run_jev_parallel_primitives(
    dataset_path: str | Path,
    output_dir: str | Path,
    repeats: int = 10,
    parallelism: int = 16,
    endpoint: str | None = None,
    api_key_env: str = "JEV_API_KEY",
    model: str = "jev-latest",
    timeout: float = 60.0,
    retries: int = 2,
    requests_per_second: float = 20.0,
) -> Path:
    """Run Noul, Choice, and Score for parallel-primitive rows efficiently.

    Each HTTP request contains all missing primitive questions for one
    `(item, repeat)` pair. This preserves the normal result-row schema while
    avoiding three separate network round trips per item.
    """

    _load_dotenv()
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Jev parallel primitive runner requires {api_key_env}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    (output / "dataset.path").write_text(str(Path(dataset_path).resolve()), encoding="utf-8")

    items = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = _completed_keys(result_path)
    tasks = []
    for repeat_index in range(repeats):
        for item in items:
            missing = [
                primitive
                for primitive, model_name in MODEL_BY_PRIMITIVE.items()
                if (model_name, item["instance_id"], repeat_index) not in completed
            ]
            if missing:
                tasks.append((item, repeat_index, missing))

    endpoint = endpoint or os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
    rate_limiter = _RateLimiter(requests_per_second)
    with result_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
            futures = [
                executor.submit(
                    _evaluate_parallel_item,
                    item,
                    repeat_index,
                    missing,
                    endpoint,
                    api_key,
                    model,
                    timeout,
                    retries,
                    rate_limiter,
                )
                for item, repeat_index, missing in tasks
            ]
            for future in as_completed(futures):
                for row in future.result():
                    key = (row["model"], row["instance_id"], row.get("repeat_index", 0))
                    if key in completed:
                        continue
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    completed.add(key)
    return result_path


class _RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 0.0 if requests_per_second <= 0 else 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.perf_counter()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = time.perf_counter()
            self.next_at = now + self.interval


def _evaluate_parallel_item(
    item: dict[str, Any],
    repeat_index: int,
    primitives: list[str],
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
    rate_limiter: "_RateLimiter",
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    try:
        raw = _post_with_retries(
            endpoint,
            api_key,
            _payload(item, model, primitives),
            timeout,
            retries,
            rate_limiter,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows = []
        for primitive in primitives:
            rows.append(_success_row(item, primitive, repeat_index, raw, latency_ms))
        return rows
    except Exception as exc:
        return [_error_row(MODEL_BY_PRIMITIVE[primitive], item, repeat_index, repr(exc)) for primitive in primitives]


def _payload(item: dict[str, Any], model: str, primitives: list[str]) -> dict[str, Any]:
    queries = item["queries"]
    questions: dict[str, Any] = {}
    if "noul" in primitives:
        proposition = queries["noul"]["proposition"]
        questions["positive_probability"] = {
            "type": "noul",
            "instructions": (
                "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
                "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
                "irrelevant or distractor. Return the probability that this statement is true: "
                f"{proposition}"
            ),
            "criteria": {
                "true": f"The proposition is true: {proposition}",
                "false": f"The proposition is false: {proposition}",
            },
        }
    if "choice" in primitives:
        query = queries["choice"]
        questions["probability_distribution"] = {
            "type": "choice",
            "instructions": (
                "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
                "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
                "irrelevant or distractor. Return the probability distribution over the exact truth-status "
                "options for the proposition; do not report confidence in your reasoning. "
                f"{query['question']} Proposition: {query['proposition']}"
            ),
            "criteria": {
                option: f"The proposition is {option.lower()}: {query['proposition']}"
                for option in query["options"]
            },
        }
    if "score" in primitives:
        query = queries["score"]
        questions["truth_degree"] = {
            "type": "score",
            "instructions": (
                "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
                "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
                "irrelevant or distractor. Use the ordered rubric to score the truth degree of the proposition. "
                f"{query['question']} Proposition: {query['proposition']}"
            ),
            "criteria": query["levels"],
        }
    return {"model": model, "state": item["state"], "questions": questions}


def _post_with_retries(endpoint: str, api_key: str, payload: dict[str, Any], timeout: float, retries: int, rate_limiter: "_RateLimiter") -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            rate_limiter.wait()
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception as exc:  # pragma: no cover - exercised by integration runs.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Jev request failed after {retries + 1} attempts: {last_error!r}")


def _success_row(
    item: dict[str, Any],
    primitive: str,
    repeat_index: int,
    raw: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    model_name = MODEL_BY_PRIMITIVE[primitive]
    pred = normalize_distribution(_distribution_for_primitive(item, primitive, raw), item["outcomes"])
    metrics = item_metrics(item["gold_distribution"], pred, item["outcomes"])
    answer = _answer_for_primitive(primitive, raw)
    return {
        "model": model_name,
        "model_version": raw.get("model") or answer.get("model"),
        "instance_id": item["instance_id"],
        "latent_instance_id": item["latent_instance_id"],
        "family": item["family"],
        "difficulty": item.get("difficulty"),
        "representation": item["representation"],
        "probability_band": item.get("probability_band"),
        "generator_metadata": item.get("generator_metadata", {}),
        "gold_distribution": item["gold_distribution"],
        "predicted_distribution": pred,
        "metrics": metrics,
        "predicted_argmax": argmax(pred),
        "gold_argmax": item["gold_argmax"],
        "argmax_correct": argmax_correct(item["gold_distribution"], pred),
        "latency_ms": latency_ms,
        "repeat_index": repeat_index,
        "error": None,
        "raw_response": raw,
        "api_metadata": {
            "usage": raw.get("usage"),
            "quota": raw.get("quota"),
            "question_type": primitive,
            "batched_primitives": sorted((raw.get("answers") or {}).keys()),
        },
    }


def _distribution_for_primitive(item: dict[str, Any], primitive: str, raw: dict[str, Any]) -> dict[str, float]:
    answer = _answer_for_primitive(primitive, raw)
    if primitive == "noul":
        if "noul" not in answer:
            raise RuntimeError(f"Jev response did not include noul probability: {raw}")
        p_true = float(answer["noul"])
        return {"True": p_true, "False": 1.0 - p_true}
    if primitive == "choice":
        probabilities = answer.get("probabilities") or answer.get("distribution")
        if not probabilities:
            raise RuntimeError(f"Jev response did not include choice probabilities: {raw}")
        return {"True": float(probabilities.get("True", 0.0)), "False": float(probabilities.get("False", 0.0))}
    if primitive == "score":
        probabilities = answer.get("probabilities") or answer.get("distribution") or {}
        if probabilities:
            values = item["queries"]["score"]["normalized_level_values"]
            total = sum(max(0.0, float(probabilities.get(str(index), 0.0))) for index in range(10))
            if total <= 0:
                raise RuntimeError(f"Jev score probabilities sum to zero: {raw}")
            p_true = sum(
                max(0.0, float(probabilities.get(str(index), 0.0))) / total * float(values[index])
                for index in range(10)
            )
        elif "score" in answer:
            p_true = float(answer["score"]) / 9.0
        else:
            raise RuntimeError(f"Jev response did not include score probability data: {raw}")
        return {"True": p_true, "False": 1.0 - p_true}
    raise ValueError(f"unknown primitive: {primitive}")


def _answer_for_primitive(primitive: str, raw: dict[str, Any]) -> dict[str, Any]:
    answers = raw.get("answers") or {}
    if primitive == "noul":
        return answers.get("positive_probability", {})
    if primitive == "choice":
        return answers.get("probability_distribution", {})
    if primitive == "score":
        return answers.get("truth_degree", {})
    return {}


def _error_row(model_name: str, item: dict[str, Any], repeat_index: int, error: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "model_version": None,
        "instance_id": item["instance_id"],
        "latent_instance_id": item["latent_instance_id"],
        "family": item["family"],
        "difficulty": item.get("difficulty"),
        "representation": item["representation"],
        "probability_band": item.get("probability_band"),
        "generator_metadata": item.get("generator_metadata", {}),
        "gold_distribution": item["gold_distribution"],
        "predicted_distribution": {outcome: 0.0 for outcome in item["outcomes"]},
        "metrics": {},
        "predicted_argmax": None,
        "gold_argmax": item["gold_argmax"],
        "argmax_correct": False,
        "latency_ms": 0.0,
        "repeat_index": repeat_index,
        "error": error,
        "raw_response": {},
        "api_metadata": {},
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

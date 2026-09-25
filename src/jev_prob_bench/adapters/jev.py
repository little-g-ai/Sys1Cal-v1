from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request
from typing import Any

from jev_prob_bench.adapters.base import ModelAdapter


class JevAdapter(ModelAdapter):
    """HTTP adapter for the documented Jev typed-question API shape.

    Current public references describe a System One request with top-level
    state/model/questions fields and choice answers that include probabilities.
    Endpoint, model, and key variable are configurable because Jev is available
    through more than one gateway.
    """

    def __init__(
        self,
        name: str = "jev",
        endpoint: str | None = None,
        api_key_env: str = "JEV_API_KEY",
        model: str = "jev-latest",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(name)
        self.endpoint = endpoint or os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout

    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        _load_dotenv()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Jev adapter requires {self.api_key_env}")
        if item and "queries" in item and "choice" in item["queries"]:
            query = item["queries"]["choice"]
            criteria = {
                option: f"The proposition is {option.lower()}: {query['proposition']}"
                for option in query["options"]
            }
            instructions = (
                "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
                "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
                "irrelevant or distractor. Return the probability distribution over the exact truth-status "
                "options for the proposition; do not report confidence in your reasoning. "
                f"{query['question']} Proposition: {query['proposition']}"
            )
        else:
            criteria = _choice_criteria(outcomes, prompt)
            instructions = (
                "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
                "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
                "irrelevant or distractor. Return the probability distribution for the random outcome or latent "
                "hypothesis described by the benchmark question; do not report confidence in your reasoning. "
                f"{prompt}"
            )
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "probability_distribution": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode())
        answer = (raw.get("answers") or {}).get("probability_distribution", {})
        probabilities = answer.get("probabilities") or answer.get("distribution")
        if not probabilities:
            raise RuntimeError(f"Jev response did not include probabilities: {raw}")
        self.last_metadata = {
            "raw_response": raw,
            "parsed_distribution": probabilities,
            "model_version": raw.get("model") or answer.get("model"),
            "api_metadata": {"usage": raw.get("usage"), "quota": raw.get("quota"), "confidence": answer.get("confidence")},
        }
        return {outcome: float(probabilities.get(outcome, 0.0)) for outcome in outcomes}


class JevScoreAdapter(JevAdapter):
    """Jev adapter using the ordered `score` primitive for parallel items.

    Jev Score returns a distribution over rubric indices and a weighted score on
    0..n-1. For compatibility with the current binary evaluator, this adapter
    reports the expected normalized truth degree as P(True). The raw score
    distribution is preserved in metadata for richer downstream analysis.
    """

    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if not item or "queries" not in item or "score" not in item["queries"]:
            raise RuntimeError("Jev Score adapter requires a parallel primitive item with queries.score")
        if set(outcomes) != {"True", "False"}:
            raise RuntimeError("Jev Score adapter currently supports binary truth-status outcomes")
        _load_dotenv()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Jev adapter requires {self.api_key_env}")
        query = item["queries"]["score"]
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "truth_degree": {
                    "type": "score",
                    "instructions": (
                        "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
                        "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
                        "irrelevant or distractor. Use the ordered rubric to score the truth degree of the proposition. "
                        f"{query['question']} Proposition: {query['proposition']}"
                    ),
                    "criteria": query["levels"],
                }
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode())
        answer = (raw.get("answers") or {}).get("truth_degree", {})
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
        p_true = max(0.0, min(1.0, p_true))
        distribution = {"True": p_true, "False": 1.0 - p_true}
        self.last_metadata = {
            "raw_response": raw,
            "parsed_distribution": distribution,
            "score_distribution": probabilities,
            "model_version": raw.get("model") or answer.get("model"),
            "api_metadata": {
                "usage": raw.get("usage"),
                "quota": raw.get("quota"),
                "confidence": answer.get("confidence"),
                "question_type": "score",
            },
        }
        return {outcome: distribution[outcome] for outcome in outcomes}


class JevNoulAdapter(JevAdapter):
    """Binary Jev adapter using the `noul` primitive.

    Noul returns a single probability that a statement is true. For binary
    benchmark items this is mapped to the first outcome and its complement.
    """

    def predict_distribution(
        self,
        state: dict[str, Any],
        prompt: str,
        outcomes: list[str],
        item: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if len(outcomes) != 2:
            raise RuntimeError("Jev Noul adapter only supports binary benchmark items")
        _load_dotenv()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Jev adapter requires {self.api_key_env}")
        positive, negative = outcomes
        if item and "queries" in item and "noul" in item["queries"]:
            statement = item["queries"]["noul"]["proposition"]
            criteria = {
                "true": f"The proposition is true: {statement}",
                "false": f"The proposition is false: {statement}",
            }
        else:
            statement = _noul_statement(positive, negative, prompt)
            criteria = {
                "true": f"The correct outcome is {positive}.",
                "false": f"The correct outcome is {negative}.",
            }
        instructions = (
            "The state fully specifies a probability problem. Treat all numeric probabilities, counts, "
            "ratios, tables, filters, and likelihoods in the state as authoritative. Ignore fields marked "
            "irrelevant or distractor. Return the probability that this statement is true: "
            f"{statement}"
        )
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "positive_probability": {
                    "type": "noul",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode())
        answer = (raw.get("answers") or {}).get("positive_probability", {})
        if "noul" not in answer:
            raise RuntimeError(f"Jev response did not include noul probability: {raw}")
        p_positive = float(answer["noul"])
        distribution = {positive: p_positive, negative: 1.0 - p_positive}
        self.last_metadata = {
            "raw_response": raw,
            "parsed_distribution": distribution,
            "model_version": raw.get("model") or answer.get("model"),
            "api_metadata": {"usage": raw.get("usage"), "quota": raw.get("quota"), "question_type": "noul"},
        }
        return distribution


def _noul_statement(positive: str, negative: str, prompt: str) -> str:
    if {positive, negative} == {"A", "not_A"}:
        return "The realized outcome is A, or the queried item belongs to class A."
    if {positive, negative} == {"H", "not_H"}:
        return "The latent hypothesis H is true after conditioning on the supplied evidence."
    if {positive, negative} == {"target", "not_target"}:
        return "The compound event named in the benchmark question occurs."
    return f"The realized outcome, class, or latent hypothesis is exactly {positive}. Benchmark question: {prompt}"


def _choice_criteria(outcomes: list[str], prompt: str) -> dict[str, str]:
    if set(outcomes) == {"A", "not_A"}:
        return {
            "A": "The realized outcome is A, or the queried item belongs to class A.",
            "not_A": "The realized outcome is not A, or the queried item does not belong to class A.",
        }
    if set(outcomes) == {"H", "not_H"}:
        return {
            "H": "The latent hypothesis H is true after conditioning on the supplied evidence.",
            "not_H": "The latent hypothesis H is false after conditioning on the supplied evidence.",
        }
    if set(outcomes) == {"target", "not_target"}:
        return {
            "target": "The compound event named in the question occurs.",
            "not_target": "The compound event named in the question does not occur.",
        }
    return {
        outcome: f"The realized outcome, class, or latent hypothesis is exactly {outcome}."
        for outcome in outcomes
    }


def _load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


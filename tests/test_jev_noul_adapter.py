import json

import pytest

from jev_prob_bench.adapters.jev import JevAdapter, JevNoulAdapter, JevScoreAdapter


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "jev-test",
                "answers": {"positive_probability": {"type": "noul", "noul": 0.23}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode()


def test_jev_noul_maps_probability_to_binary_distribution(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())
    adapter = JevNoulAdapter(name="jev_noul", endpoint="https://example.test", model="jev-test")
    pred = adapter.predict_distribution({"p_A": 0.23}, "What is P(A)?", ["A", "not_A"])
    assert pred == {"A": 0.23, "not_A": 0.77}
    assert adapter.last_metadata["model_version"] == "jev-test"


def test_jev_noul_rejects_multiclass():
    adapter = JevNoulAdapter()
    with pytest.raises(RuntimeError, match="binary"):
        adapter.predict_distribution({}, "question", ["A", "B", "C"])



class ChoiceResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "jev-test",
                "answers": {
                    "probability_distribution": {
                        "type": "choice",
                        "probabilities": {"False": 0.8, "True": 0.2},
                    }
                },
            }
        ).encode()


class ScoreResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "jev-test",
                "answers": {
                    "truth_degree": {
                        "type": "score",
                        "score": 2.25,
                        "probabilities": {"2": 0.75, "3": 0.25},
                        "confidence": 0.75,
                    }
                },
            }
        ).encode()


def _parallel_item():
    return {
        "queries": {
            "noul": {"proposition": "Event A is true."},
            "choice": {
                "question": "What is the truth status of the following proposition?",
                "proposition": "Event A is true.",
                "options": ["False", "True"],
            },
            "score": {
                "question": "To what degree is the following proposition true?",
                "proposition": "Event A is true.",
                "levels": [str(index) for index in range(10)],
                "normalized_level_values": [index / 9 for index in range(10)],
            },
        }
    }


def test_jev_noul_uses_parallel_query_proposition(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = JevNoulAdapter(name="jev_noul", endpoint="https://example.test", model="jev-test")
    pred = adapter.predict_distribution({"p_A": 0.23}, "legacy prompt", ["True", "False"], item=_parallel_item())
    assert pred == {"True": 0.23, "False": 0.77}
    question = captured["payload"]["questions"]["positive_probability"]
    assert question["criteria"]["true"] == "The proposition is true: Event A is true."
    assert "legacy prompt" not in question["instructions"]


def test_jev_choice_uses_parallel_choice_options(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode())
        return ChoiceResponse()

    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = JevAdapter(name="jev_choice", endpoint="https://example.test", model="jev-test")
    pred = adapter.predict_distribution({"p_A": 0.2}, "legacy prompt", ["True", "False"], item=_parallel_item())
    assert pred == {"True": 0.2, "False": 0.8}
    criteria = captured["payload"]["questions"]["probability_distribution"]["criteria"]
    assert list(criteria) == ["False", "True"]
    assert criteria["True"] == "The proposition is true: Event A is true."


def test_jev_score_maps_weighted_score_to_expected_truth_degree(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: ScoreResponse())
    adapter = JevScoreAdapter(name="jev_score", endpoint="https://example.test", model="jev-test")
    pred = adapter.predict_distribution({"p_A": 0.25}, "legacy prompt", ["True", "False"], item=_parallel_item())
    assert pred == {"True": 0.25, "False": 0.75}
    assert adapter.last_metadata["score_distribution"] == {"2": 0.75, "3": 0.25}

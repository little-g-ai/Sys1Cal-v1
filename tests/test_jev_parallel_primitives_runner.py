import json

from jev_prob_bench.evaluation.jev_parallel_primitives import _distribution_for_primitive, _payload


def _item():
    return {
        "state": {"p_A": 0.4},
        "outcomes": ["True", "False"],
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
        },
    }


def test_parallel_payload_can_include_all_three_primitives():
    payload = _payload(_item(), "jev-test", ["noul", "choice", "score"])
    assert payload["model"] == "jev-test"
    assert set(payload["questions"]) == {"positive_probability", "probability_distribution", "truth_degree"}
    assert payload["questions"]["truth_degree"]["criteria"] == [str(index) for index in range(10)]


def test_distribution_parsing_for_combined_response():
    raw = {
        "answers": {
            "positive_probability": {"type": "noul", "noul": 0.4},
            "probability_distribution": {"type": "choice", "probabilities": {"False": 0.7, "True": 0.3}},
            "truth_degree": {
                "type": "score",
                "probabilities": {str(index): 0.0 for index in range(10)},
            },
        }
    }
    raw["answers"]["truth_degree"]["probabilities"]["3"] = 0.25
    raw["answers"]["truth_degree"]["probabilities"]["9"] = 0.75
    assert _distribution_for_primitive(_item(), "noul", raw) == {"True": 0.4, "False": 0.6}
    assert _distribution_for_primitive(_item(), "choice", raw) == {"True": 0.3, "False": 0.7}
    score = _distribution_for_primitive(_item(), "score", raw)
    assert score["True"] == 0.25 * (3 / 9) + 0.75
    assert score["False"] == 1 - score["True"]

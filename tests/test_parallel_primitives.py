import json

from jev_prob_bench.parallel_primitives import (
    SCORE_LEVELS,
    SCORE_NORMALIZED_LEVEL_VALUES,
    make_parallel_primitive_dataset,
    make_parallel_primitive_item,
)


def _source_item():
    return {
        "benchmark_version": "0.1.0",
        "instance_id": "frequency__000124__repr_counts",
        "latent_instance_id": "frequency__000124",
        "family": "frequency",
        "difficulty": 1,
        "seed": 42,
        "representation": "counts",
        "state": {"zor": 37, "nif": 63, "sampling": "uniform"},
        "prompt": "What is the probability that the sampled object is a zor?",
        "outcomes": ["zor", "nif"],
        "gold_distribution": {"zor": 0.37, "nif": 0.63},
        "gold_argmax": "nif",
        "generator_metadata": {},
        "probability_band": "(0.30,0.45]",
    }


def test_parallel_primitive_item_preserves_one_latent_problem_across_queries():
    item = make_parallel_primitive_item(_source_item())
    assert item["latent_instance_id"] == "frequency__000124"
    assert item["representation"] == "counts"
    assert item["state"] == {"zor": 37, "nif": 63, "sampling": "uniform"}
    assert item["proposition"] == "The sampled object is a zor."
    assert item["gold"] == {"probability_true": 0.37, "probability_false": 0.63}
    assert item["queries"]["noul"]["proposition"] == item["proposition"]
    assert item["queries"]["choice"]["proposition"] == item["proposition"]
    assert item["queries"]["choice"]["options"] == ["False", "True"]
    assert item["queries"]["score"]["proposition"] == item["proposition"]
    assert item["queries"]["score"]["levels"] == SCORE_LEVELS
    assert item["queries"]["score"]["normalized_level_values"] == SCORE_NORMALIZED_LEVEL_VALUES


def test_parallel_primitive_item_keeps_legacy_binary_fields_for_runner():
    item = make_parallel_primitive_item(_source_item())
    assert item["prompt"] == item["proposition"]
    assert item["outcomes"] == ["True", "False"]
    assert item["gold_distribution"] == {"True": 0.37, "False": 0.63}
    assert item["gold_argmax"] == "False"


def test_parallel_primitive_dataset_skips_non_binary_items(tmp_path):
    binary = _source_item()
    multiclass = dict(binary)
    multiclass["instance_id"] = "multi"
    multiclass["outcomes"] = ["a", "b", "c"]
    multiclass["gold_distribution"] = {"a": 0.2, "b": 0.3, "c": 0.5}
    source = tmp_path / "source.jsonl"
    target = tmp_path / "parallel.jsonl"
    source.write_text(json.dumps(binary) + "\n" + json.dumps(multiclass) + "\n", encoding="utf-8")
    make_parallel_primitive_dataset(source, target)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["instance_id"].endswith("__parallel_primitives")

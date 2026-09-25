from jev_prob_bench.dataset_variants import make_choice_explicit_item


def test_choice_explicit_variant_preserves_gold_and_adds_semantics():
    item = {
        "instance_id": "x__repr_direct",
        "representation": "direct",
        "state": {"p_A": 0.3},
        "prompt": "What is P(A)?",
        "outcomes": ["A", "not_A"],
        "gold_distribution": {"A": 0.3, "not_A": 0.7},
        "generator_metadata": {},
    }
    explicit = make_choice_explicit_item(item)
    assert explicit["gold_distribution"] == item["gold_distribution"]
    assert explicit["instance_id"].endswith("__choice_explicit")
    assert explicit["state"]["choice_option_semantics"]["mutually_exclusive"] is True
    assert "mutually exclusive and exhaustive" in explicit["prompt"]


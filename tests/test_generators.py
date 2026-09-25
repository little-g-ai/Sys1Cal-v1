from jev_prob_bench.generators import GENERATOR_REGISTRY
from jev_prob_bench.schemas import validate_distribution


def test_all_generators_emit_valid_items():
    for generator_cls in GENERATOR_REGISTRY.values():
        generator = generator_cls(version="0.1.0", seed=7)
        items = list(generator.generate(3))
        assert items
        for item in items:
            validate_distribution(item.gold_distribution, item.outcomes)


def test_representations_share_gold_distribution():
    generator = GENERATOR_REGISTRY["bayes"](version="0.1.0", seed=7)
    items = list(generator.generate(1))
    gold = items[0].gold_distribution
    assert all(item.gold_distribution == gold for item in items)
    assert len({item.representation for item in items}) > 1



def test_explicit_probability_tiny_spans_closed_unit_interval():
    generator = GENERATOR_REGISTRY["explicit_probability"](version="0.1.0", seed=7)
    items = list(generator.generate(20))
    direct = [item for item in items if item.representation == "direct"]
    probs = [item.gold_distribution["A"] for item in direct]
    assert probs[0] == 0.0
    assert probs[-1] == 1.0
    assert min(probs) == 0.0
    assert max(probs) == 1.0


def test_ratio_renderer_preserves_probability_endpoints():
    generator = GENERATOR_REGISTRY["explicit_probability"](version="0.1.0", seed=7)
    items = list(generator.generate(20))
    ratio_zero = next(item for item in items if item.representation == "ratio" and item.gold_distribution["A"] == 0.0)
    ratio_one = next(item for item in items if item.representation == "ratio" and item.gold_distribution["A"] == 1.0)
    assert ratio_zero.state["probability_ratio_A_to_not_A"] == "0:1"
    assert ratio_one.state["probability_ratio_A_to_not_A"] == "1:0"

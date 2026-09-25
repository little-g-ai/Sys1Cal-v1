from jev_prob_bench.evaluation.metrics import argmax_correct, brier_regret, item_metrics, total_variation


def test_oracle_metrics_are_zero():
    gold = {"A": 0.3, "not_A": 0.7}
    metrics = item_metrics(gold, gold, ["A", "not_A"])
    assert metrics["tv"] == 0
    assert metrics["mae_binary"] == 0
    assert metrics["brier_regret"] == 0
    assert metrics["kl"] == 0


def test_total_variation_binary_equals_absolute_positive_error():
    gold = {"A": 0.3, "not_A": 0.7}
    pred = {"A": 0.4, "not_A": 0.6}
    assert round(total_variation(gold, pred, ["A", "not_A"]), 10) == 0.1
    assert round(brier_regret(gold, pred, ["A", "not_A"]), 10) == 0.02
    assert argmax_correct(gold, pred)


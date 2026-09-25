from jev_prob_bench.evaluation.representation import aggregate_representation_sensitivity, representation_sensitivity


def test_representation_sensitivity_for_identical_predictions_is_zero():
    rows = []
    for representation in ["direct", "counts"]:
        rows.append(
            {
                "model": "oracle",
                "latent_instance_id": "z1",
                "representation": representation,
                "gold_distribution": {"A": 0.25, "not_A": 0.75},
                "predicted_distribution": {"A": 0.25, "not_A": 0.75},
                "error": None,
            }
        )
    detail = representation_sensitivity(rows)
    summary = aggregate_representation_sensitivity(detail)
    assert detail[0]["mean_pairwise_tv"] == 0
    assert summary[0]["ris"] == 1


from jev_prob_bench.evaluation.noul_choice_score import (
    build_ncs_records,
    projection_model_summary,
    representation_sensitivity_summary,
    summarize_ncs,
)


def _row(model, instance_id="x__repr_direct", latent="x", representation="direct", pred_true=0.3, p_star=0.25, q=None):
    raw_response = {}
    if model == "jev_score":
        raw_response = {
            "answers": {
                "truth_degree": {
                    "type": "score",
                    "probabilities": {str(i): value for i, value in enumerate(q or [0.0] * 10)},
                }
            }
        }
    return {
        "model": model,
        "instance_id": instance_id,
        "latent_instance_id": latent,
        "family": "explicit_probability",
        "representation": representation,
        "repeat_index": 0,
        "gold_distribution": {"True": p_star, "False": 1 - p_star},
        "predicted_distribution": {"True": pred_true, "False": 1 - pred_true},
        "raw_response": raw_response,
        "metrics": {"tv": abs(pred_true - p_star)},
        "error": None,
    }


def test_build_ncs_records_computes_score_expectation_from_distribution():
    q = [0.0] * 10
    q[2] = 0.5
    q[8] = 0.5
    rows = [
        _row("jev_noul", pred_true=0.2, p_star=0.25),
        _row("jev_choice", pred_true=0.4, p_star=0.25),
        _row("jev_score", pred_true=0.0, p_star=0.25, q=q),
    ]
    records = build_ncs_records(rows)
    assert len(records) == 1
    record = records[0]
    assert record["p_noul"] == 0.2
    assert record["p_choice"] == 0.4
    assert record["mu_score"] == (2 / 9 * 0.5 + 8 / 9 * 0.5)
    assert record["abs_nc"] == 0.2
    assert record["score_threshold"] == 0.5


def test_ncs_summaries_and_projection_models_return_required_outputs():
    rows = []
    for i, p in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.7]):
        q = [0.0] * 10
        q[round(p * 9)] = 1.0
        instance = f"x{i}__repr_direct"
        latent = f"x{i}"
        rows.extend([
            _row("jev_noul", instance, latent, "direct", p, p),
            _row("jev_choice", instance, latent, "direct", min(1.0, max(0.0, p + 0.05)), p),
            _row("jev_score", instance, latent, "direct", p, p, q),
        ])
    records = build_ncs_records(rows)
    summary = summarize_ncs(records)
    projection = projection_model_summary(records)
    assert {row["metric"] for row in summary} >= {"noul_mae_to_p_star", "choice_mae_to_p_star", "abs_noul_choice"}
    assert {row["collapse_rule"] for row in projection} >= {"expected_truth", "hard_threshold", "endpoint_conditioning", "parametric_sharpening", "monotonic_collapse"}
    assert {row["target"] for row in projection} == {"noul", "choice"}


def test_ncs_representation_sensitivity_has_separate_primitives():
    rows = []
    for representation, p in [("direct", 0.2), ("prose", 0.4)]:
        q = [0.0] * 10
        q[round(p * 9)] = 1.0
        instance = f"x__repr_{representation}"
        rows.extend([
            _row("jev_noul", instance, "x", representation, p, 0.3),
            _row("jev_choice", instance, "x", representation, p + 0.1, 0.3),
            _row("jev_score", instance, "x", representation, p, 0.3, q),
        ])
    sensitivity = representation_sensitivity_summary(build_ncs_records(rows))
    assert [row["primitive"] for row in sensitivity] == ["noul", "choice", "score"]
    assert all(row["n_latents"] == 1 for row in sensitivity)

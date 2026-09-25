from jev_prob_bench.evaluation.benchmark_summary import benchmark_summary, mean_tv_error, pointwise_probability_fidelity
from jev_prob_bench.evaluation.noul_choice_score import build_ncs_records


def _row(model, instance_id, pred_true, p_star, q=None):
    raw_response = {}
    if "score" in model:
        raw_response = {
            "answers": {
                "truth_degree": {
                    "type": "score",
                    "probabilities": {str(index): value for index, value in enumerate(q or [0.0] * 10)},
                }
            }
        }
    return {
        "model": model,
        "instance_id": instance_id,
        "latent_instance_id": instance_id,
        "family": "explicit_probability",
        "representation": "direct",
        "repeat_index": "mean",
        "gold_distribution": {"True": p_star, "False": 1.0 - p_star},
        "predicted_distribution": {"True": pred_true, "False": 1.0 - pred_true},
        "raw_response": raw_response,
        "argmax_correct": (pred_true >= 0.5) == (p_star >= 0.5),
        "metrics": {"tv": abs(pred_true - p_star)},
        "error": None,
    }


def test_pointwise_probability_fidelity_is_one_minus_mean_tv():
    rows = [
        _row("jev_noul", "a", 0.2, 0.1),
        _row("jev_noul", "b", 0.8, 0.6),
    ]
    assert round(mean_tv_error(rows), 10) == 0.15
    assert round(pointwise_probability_fidelity(rows), 10) == 0.85


def test_benchmark_summary_reports_primitive_rows_and_pairwise_r2():
    rows = []
    for index, p in enumerate([0.1, 0.3, 0.7, 0.9]):
        q = [0.0] * 10
        q[round(p * 9)] = 1.0
        rows.extend(
            [
                _row("jev_noul", f"x{index}", p, p, q),
                _row("jev_choice", f"x{index}", min(1.0, p + 0.05), p, q),
                _row("jev_score", f"x{index}", p, p, q),
            ]
        )
    summary = benchmark_summary(rows, build_ncs_records(rows))
    response_types = {row["response_type"] for row in summary}
    assert {"noul_like", "choice_like", "score_like", "overall"} <= response_types
    noul_row = next(row for row in summary if row["response_type"] == "noul_like")
    assert noul_row["model"] == "jev"
    assert noul_row["pointwise_probability_fidelity"] == 1.0
    assert noul_row["mean_tv_error"] == 0.0
    assert noul_row["noul_choice_r2"] is not None
    assert noul_row["noul_score_r2"] is not None

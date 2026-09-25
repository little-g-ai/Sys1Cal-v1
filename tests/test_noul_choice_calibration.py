from jev_prob_bench.evaluation.noul_choice_score import (
    noul_choice_calibration_predictions,
    noul_choice_calibration_summary,
)


def _record(i, p):
    q = 1 - (1 - p) ** 2
    return {
        "instance_id": f"x{i}",
        "latent_instance_id": f"latent{i}",
        "family": "explicit_probability",
        "representation": "direct",
        "p_star": p,
        "p_noul": p,
        "p_choice": q,
        "mu_score": p,
        "score_threshold": 1.0 if p >= 0.5 else 0.0,
        "score_endpoint": q,
        "score_endpoint_mass": 1.0,
        **{f"q_{level}": 1.0 if level == round(p * 9) else 0.0 for level in range(10)},
    }


def test_noul_choice_calibration_includes_power_logit_beta_isotonic():
    records = [_record(i, p) for i, p in enumerate([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.35])]
    summary = noul_choice_calibration_summary(records)
    mappings = {row["mapping"] for row in summary}
    assert {"identity", "power", "logit", "beta", "isotonic"} <= mappings
    power = next(row for row in summary if row["mapping"] == "power")
    assert "alpha" in power
    assert power["mae"] < next(row for row in summary if row["mapping"] == "identity")["mae"]


def test_noul_choice_calibration_predictions_have_implied_unknown():
    records = [_record(i, p) for i, p in enumerate([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.35])]
    predictions = noul_choice_calibration_predictions(records)
    assert predictions
    assert "pred_choice_power" in predictions[0]
    assert "implied_unknown_power" in predictions[0]

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from itertools import combinations
from typing import Any

from jev_prob_bench.evaluation.benchmark_summary import base_model_name, primitive_for_model


SCORE_INDICES = [str(index) for index in range(10)]
SCORE_VALUES = [index / 9 for index in range(10)]


def build_ncs_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            continue
        model = row.get("model")
        primitive = primitive_for_model(str(model))
        if primitive is None:
            continue
        grouped[(base_model_name(str(model)), row["instance_id"], row.get("repeat_index", 0))][primitive] = row

    records = []
    for (base_model, _, _), model_rows in sorted(grouped.items()):
        if set(model_rows) != {"noul", "choice", "score"}:
            continue
        noul = model_rows["noul"]
        choice = model_rows["choice"]
        score = model_rows["score"]
        p_star = _positive_probability(noul["gold_distribution"])
        p_n = _positive_probability(noul["predicted_distribution"])
        p_c = _positive_probability(choice["predicted_distribution"])
        q = _score_distribution(score)
        mu_s = sum(q[index] * SCORE_VALUES[index] for index in range(10))
        threshold = sum(q[index] for index in range(5, 10))
        endpoint_mass = q[0] + q[9]
        endpoint = q[9] / endpoint_mass if endpoint_mass > 0 else None
        records.append(
            {
                "base_model": base_model,
                "instance_id": noul["instance_id"],
                "latent_instance_id": noul["latent_instance_id"],
                "family": noul["family"],
                "representation": noul["representation"],
                "probability_band": noul.get("probability_band"),
                "p_star": p_star,
                "p_noul": p_n,
                "p_choice": p_c,
                "mu_score": mu_s,
                "score_threshold": threshold,
                "score_endpoint": endpoint,
                "score_endpoint_mass": endpoint_mass,
                "score_adapter_probability": _positive_probability(score["predicted_distribution"]),
                "error_noul": abs(p_n - p_star),
                "error_choice": abs(p_c - p_star),
                "error_score_expectation": abs(mu_s - p_star),
                "delta_nc": p_c - p_n,
                "abs_nc": abs(p_c - p_n),
                "extremeness_noul": abs(p_n - 0.5),
                "extremeness_choice": abs(p_c - 0.5),
                "extremeness_choice_minus_noul": abs(p_c - 0.5) - abs(p_n - 0.5),
                "abs_score_noul": abs(mu_s - p_n),
                "abs_score_choice": abs(mu_s - p_c),
                **{f"q_{index}": q[index] for index in range(10)},
            }
        )
    return records


def summarize_ncs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    return [
        {
            "section": "external_probability_fidelity",
            "metric": "noul_mae_to_p_star",
            **_error_summary([record["error_noul"] for record in records]),
        },
        {
            "section": "external_probability_fidelity",
            "metric": "choice_mae_to_p_star",
            **_error_summary([record["error_choice"] for record in records]),
        },
        {
            "section": "external_probability_fidelity",
            "metric": "score_expectation_mae_to_p_star",
            **_error_summary([record["error_score_expectation"] for record in records]),
        },
        {
            "section": "cross_primitive_consistency",
            "metric": "abs_noul_choice",
            **_error_summary([record["abs_nc"] for record in records]),
        },
        {
            "section": "cross_primitive_consistency",
            "metric": "abs_noul_score_expectation",
            **_error_summary([record["abs_score_noul"] for record in records]),
        },
        {
            "section": "cross_primitive_consistency",
            "metric": "abs_choice_score_expectation",
            **_error_summary([record["abs_score_choice"] for record in records]),
        },
        {
            "section": "choice_extremeness",
            "metric": "choice_minus_noul_extremeness",
            **_signed_summary([record["extremeness_choice_minus_noul"] for record in records]),
        },
        {
            "section": "noul_choice_bias",
            "metric": "choice_minus_noul_probability",
            **_signed_summary([record["delta_nc"] for record in records]),
        },
    ]


def projection_model_summary(records: list[dict[str, Any]], endpoint_min_mass: float = 0.05) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    rows = []
    for target_name, target_key in [("noul", "p_noul"), ("choice", "p_choice")]:
        rows.append(_projection_row("expected_truth", target_name, test, [record["mu_score"] for record in test], None))
        rows.append(_projection_row("hard_threshold", target_name, test, [record["score_threshold"] for record in test], None))

        endpoint_test = [record for record in test if record["score_endpoint"] is not None and record["score_endpoint_mass"] >= endpoint_min_mass]
        rows.append(
            _projection_row(
                "endpoint_conditioning",
                target_name,
                endpoint_test,
                [record["score_endpoint"] for record in endpoint_test],
                {"endpoint_min_mass": endpoint_min_mass, "n_test_eligible": len(endpoint_test)},
            )
        )

        beta = _fit_beta(train, target_key)
        rows.append(
            _projection_row(
                "parametric_sharpening",
                target_name,
                test,
                [_predict_beta(record, beta) for record in test],
                {"beta": beta},
            )
        )

        alpha = _fit_monotonic_alpha(train, target_key)
        rows.append(
            _projection_row(
                "monotonic_collapse",
                target_name,
                test,
                [_predict_alpha(record, alpha) for record in test],
                {f"alpha_{index}": alpha[index] for index in range(10)},
            )
        )

        weights = _fit_ridge_q(train, target_key)
        rows.append(
            _projection_row(
                "full_q_ridge",
                target_name,
                test,
                [_predict_ridge_q(record, weights) for record in test],
                {"ridge_l2": 0.001, **{f"ridge_w_{index}": weights[index + 1] for index in range(10)}, "ridge_intercept": weights[0]},
            )
        )

        if target_name == "choice":
            false_max, true_min = _fit_crisp_unknown(train, target_key)
            rows.append(
                _projection_row(
                    "crisp_unknown_conditional",
                    target_name,
                    test,
                    [_predict_crisp_unknown(record, false_max, true_min)[0] for record in test],
                    {"false_max_level": false_max, "true_min_level": true_min},
                )
            )
            fuzzy_params = _fit_fuzzy_unknown(train, target_key)
            rows.append(
                _projection_row(
                    "fuzzy_unknown_conditional",
                    target_name,
                    test,
                    [_predict_fuzzy_unknown(record, fuzzy_params)[0] for record in test],
                    {
                        "fuzzy_beta": fuzzy_params[0],
                        "unknown_lambda": fuzzy_params[1],
                        "unknown_gamma": fuzzy_params[2],
                    },
                )
            )
    return rows


def choice_from_score_recovery(records: list[dict[str, Any]], endpoint_min_mass: float = 0.05) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    beta = _fit_beta(train, "p_choice")
    alpha = _fit_monotonic_alpha(train, "p_choice")
    ridge = _fit_ridge_q(train, "p_choice")
    false_max, true_min = _fit_crisp_unknown(train, "p_choice")
    fuzzy_params = _fit_fuzzy_unknown(train, "p_choice")
    rows = []
    for record in test:
        crisp_prediction, crisp_unknown = _predict_crisp_unknown(record, false_max, true_min)
        fuzzy_prediction, fuzzy_unknown = _predict_fuzzy_unknown(record, fuzzy_params)
        predictions = {
            "expected_truth": record["mu_score"],
            "hard_threshold": record["score_threshold"],
            "endpoint_conditioning": record["score_endpoint"] if record["score_endpoint_mass"] >= endpoint_min_mass else None,
            "parametric_sharpening": _predict_beta(record, beta),
            "monotonic_collapse": _predict_alpha(record, alpha),
            "full_q_ridge": _predict_ridge_q(record, ridge),
            "crisp_unknown_conditional": crisp_prediction,
            "fuzzy_unknown_conditional": fuzzy_prediction,
        }
        best_available = min(
            ((name, value) for name, value in predictions.items() if value is not None),
            key=lambda item: abs(float(item[1]) - record["p_choice"]),
        )
        rows.append(
            {
                "instance_id": record["instance_id"],
                "latent_instance_id": record["latent_instance_id"],
                "family": record["family"],
                "representation": record["representation"],
                "p_star": record["p_star"],
                "p_noul": record["p_noul"],
                "p_choice": record["p_choice"],
                "mu_score": record["mu_score"],
                "score_endpoint_mass": record["score_endpoint_mass"],
                "pred_choice_expected_truth": predictions["expected_truth"],
                "pred_choice_hard_threshold": predictions["hard_threshold"],
                "pred_choice_endpoint_conditioning": predictions["endpoint_conditioning"],
                "pred_choice_parametric_sharpening": predictions["parametric_sharpening"],
                "pred_choice_monotonic_collapse": predictions["monotonic_collapse"],
                "pred_choice_full_q_ridge": predictions["full_q_ridge"],
                "pred_choice_crisp_unknown_conditional": predictions["crisp_unknown_conditional"],
                "pred_choice_fuzzy_unknown_conditional": predictions["fuzzy_unknown_conditional"],
                "crisp_unknown_mass": crisp_unknown,
                "fuzzy_unknown_mass": fuzzy_unknown,
                "crisp_false_max_level": false_max,
                "crisp_true_min_level": true_min,
                "fuzzy_beta": fuzzy_params[0],
                "unknown_lambda": fuzzy_params[1],
                "unknown_gamma": fuzzy_params[2],
                "abs_error_expected_truth": abs(predictions["expected_truth"] - record["p_choice"]),
                "abs_error_hard_threshold": abs(predictions["hard_threshold"] - record["p_choice"]),
                "abs_error_endpoint_conditioning": None if predictions["endpoint_conditioning"] is None else abs(predictions["endpoint_conditioning"] - record["p_choice"]),
                "abs_error_parametric_sharpening": abs(predictions["parametric_sharpening"] - record["p_choice"]),
                "abs_error_monotonic_collapse": abs(predictions["monotonic_collapse"] - record["p_choice"]),
                "abs_error_full_q_ridge": abs(predictions["full_q_ridge"] - record["p_choice"]),
                "abs_error_crisp_unknown_conditional": abs(predictions["crisp_unknown_conditional"] - record["p_choice"]),
                "abs_error_fuzzy_unknown_conditional": abs(predictions["fuzzy_unknown_conditional"] - record["p_choice"]),
                "oracle_best_rule": best_available[0],
                "oracle_best_abs_error": abs(float(best_available[1]) - record["p_choice"]),
                **{f"q_{index}": record[f"q_{index}"] for index in range(10)},
            }
        )
    return rows



def score_choice_tuf_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    truth_models = [
        ("score_expectation", None),
        ("score_to_noul_monotonic", _fit_monotonic_alpha(train, "p_noul")),
    ]
    rows = []
    for truth_name, truth_model in truth_models:
        central_lambda = _fit_choice_tuf_central_lambda(train, truth_name, truth_model)
        learned_weights = _fit_choice_tuf_unknown_weights(train, truth_name, truth_model)
        candidates = [
            ("no_unknown", [_predict_choice_tuf(record, truth_name, truth_model, [0.0] * 10) for record in test], {}),
            ("central_2_unknown", [_predict_choice_tuf(record, truth_name, truth_model, _central_2_weights()) for record in test], {"u_profile": _format_weights(_central_2_weights())}),
            (
                "central_decay_unknown",
                [_predict_choice_tuf(record, truth_name, truth_model, _central_decay_weights(central_lambda)) for record in test],
                {"central_lambda": central_lambda, "u_profile": _format_weights(_central_decay_weights(central_lambda))},
            ),
            (
                "learned_symmetric_unknown",
                [_predict_choice_tuf(record, truth_name, truth_model, learned_weights) for record in test],
                {"u_profile": _format_weights(learned_weights)},
            ),
        ]
        y = [record["p_choice"] for record in test]
        for unknown_rule, predictions, extra in candidates:
            metrics = _prediction_metrics(y, predictions)
            predicted_u = [_predict_choice_tuf_parts(record, truth_name, truth_model, extra.get("u_profile"))["U"] if False else None for record in []]
            rows.append(
                {
                    "truth_model": truth_name,
                    "unknown_rule": unknown_rule,
                    **metrics,
                    "corr": _correlation(y, predictions),
                    "mean_observed_choice": _mean(y),
                    "mean_predicted_choice": _mean(predictions),
                    "mean_predicted_unknown": _mean([_predict_score_unknown_weights(record, _choice_tuf_weights_from_extra(unknown_rule, extra, central_lambda, learned_weights)) for record in test]),
                    **extra,
                }
            )
    return rows


def score_choice_tuf_by_family(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)
    rows = []
    for family, family_records in sorted(grouped.items()):
        if len(family_records) < 8:
            continue
        for row in score_choice_tuf_summary(family_records):
            rows.append({"family": family, **row})
    return rows


def score_choice_tuf_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    truth_alpha = _fit_monotonic_alpha(train, "p_noul")
    learned_weights = _fit_choice_tuf_unknown_weights(train, "score_to_noul_monotonic", truth_alpha)
    rows = []
    for record in test:
        parts = _predict_choice_tuf_parts(record, "score_to_noul_monotonic", truth_alpha, learned_weights)
        rows.append(
            {
                "instance_id": record["instance_id"],
                "latent_instance_id": record["latent_instance_id"],
                "family": record["family"],
                "representation": record["representation"],
                "p_noul": record["p_noul"],
                "p_choice": record["p_choice"],
                "t_score": parts["T"],
                "u_score": parts["U"],
                "f_score": parts["F"],
                "pred_choice_tuf": parts["choice"],
                "abs_error_choice": abs(parts["choice"] - record["p_choice"]),
                "u_profile": _format_weights(learned_weights),
                **{f"q_{index}": record[f"q_{index}"] for index in range(10)},
            }
        )
    return rows


def _choice_tuf_weights_from_extra(unknown_rule: str, extra: dict[str, Any], central_lambda: float, learned_weights: list[float]) -> list[float]:
    if unknown_rule == "central_2_unknown":
        return _central_2_weights()
    if unknown_rule == "central_decay_unknown":
        return _central_decay_weights(central_lambda)
    if unknown_rule == "learned_symmetric_unknown":
        return learned_weights
    return [0.0] * 10


def _predict_choice_tuf(record: dict[str, Any], truth_name: str, truth_model: Any, u_weights: list[float]) -> float:
    return _predict_choice_tuf_parts(record, truth_name, truth_model, u_weights)["choice"]


def _predict_choice_tuf_parts(record: dict[str, Any], truth_name: str, truth_model: Any, u_weights: list[float]) -> dict[str, float]:
    t_score = _predict_choice_tuf_truth(record, truth_name, truth_model)
    raw_u = _predict_score_unknown_weights(record, u_weights)
    u_score = min(max(0.0, raw_u), max(0.0, 1.0 - t_score - 1e-6))
    f_score = max(0.0, 1.0 - t_score - u_score)
    choice = t_score / max(1e-6, t_score + f_score)
    return {"T": t_score, "U": u_score, "F": f_score, "choice": min(1.0, max(0.0, choice))}


def _predict_choice_tuf_truth(record: dict[str, Any], truth_name: str, truth_model: Any) -> float:
    if truth_name == "score_to_noul_monotonic":
        return min(1.0 - 1e-6, max(1e-6, _predict_alpha(record, truth_model)))
    return min(1.0 - 1e-6, max(1e-6, record["mu_score"]))


def _fit_choice_tuf_central_lambda(records: list[dict[str, Any]], truth_name: str, truth_model: Any) -> float:
    if not records:
        return 0.0
    best = (float("inf"), 0.0)
    for index in range(101):
        central_lambda = index / 100
        weights = _central_decay_weights(central_lambda)
        mse = _mean([(_predict_choice_tuf(record, truth_name, truth_model, weights) - record["p_choice"]) ** 2 for record in records])
        if mse < best[0]:
            best = (mse, central_lambda)
    return best[1]


def _fit_choice_tuf_unknown_weights(records: list[dict[str, Any]], truth_name: str, truth_model: Any) -> list[float]:
    params = [0.05, 0.15, 0.35, 0.75]
    if not records:
        return _symmetric_weights(params)
    for step in range(1400):
        weights = _symmetric_weights(params)
        gradients = [0.0 for _ in range(4)]
        for record in records:
            t_score = _predict_choice_tuf_truth(record, truth_name, truth_model)
            raw_u = _predict_score_unknown_weights(record, weights)
            cap = max(0.0, 1.0 - t_score - 1e-6)
            u_score = min(raw_u, cap)
            denom = max(1e-6, 1.0 - u_score)
            pred = t_score / denom
            err = pred - record["p_choice"]
            if raw_u >= cap:
                continue
            paired_q = [
                record["q_1"] + record["q_8"],
                record["q_2"] + record["q_7"],
                record["q_3"] + record["q_6"],
                record["q_4"] + record["q_5"],
            ]
            for index in range(4):
                gradients[index] += 2.0 * err * t_score * paired_q[index] / (denom * denom * len(records))
        lr = 0.18 / (1.0 + step / 1800.0)
        params = [params[index] - lr * gradients[index] for index in range(4)]
        params = _project_monotone(params)
    return _symmetric_weights(params)


def _central_2_weights() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def _central_decay_weights(central_lambda: float) -> list[float]:
    value = min(1.0, max(0.0, central_lambda))
    return [0.0, value**3, value**2, value, 1.0, 1.0, value, value**2, value**3, 0.0]


def score_three_state_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    model = _fit_score_three_state_model(train)
    rows = []
    for record in test:
        masses = _predict_score_three_state(record, model)
        implied = _implied_unknown(record["p_noul"], record["p_choice"])
        row = {
            "instance_id": record["instance_id"],
            "latent_instance_id": record["latent_instance_id"],
            "family": record["family"],
            "representation": record["representation"],
            "p_star": record["p_star"],
            "p_noul": record["p_noul"],
            "p_choice": record["p_choice"],
            "implied_unknown": implied,
            "pred_noul_from_score_t": masses["T"],
            "pred_choice_from_score_tf": masses["choice"],
            "pred_unknown_from_score_u": masses["U"],
            "pred_false_from_score_f": masses["F"],
            "abs_error_noul": abs(masses["T"] - record["p_noul"]),
            "abs_error_choice": abs(masses["choice"] - record["p_choice"]),
            "abs_error_unknown": None if implied is None else abs(masses["U"] - implied),
            "u_profile": _format_weights(model["u"]),
            "t_profile": _format_weights(model["t"]),
            "f_profile": _format_weights(model["f"]),
            "truth_share_profile": _format_weights(model["v"]),
            "train_joint_mse": model["train_joint_mse"],
            **{f"q_{index}": record[f"q_{index}"] for index in range(10)},
        }
        rows.append(row)
    return rows


def score_three_state_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = score_three_state_records(records)
    if not rows:
        return []
    model_fields = {
        "u_profile": rows[0]["u_profile"],
        "t_profile": rows[0]["t_profile"],
        "f_profile": rows[0]["f_profile"],
        "truth_share_profile": rows[0]["truth_share_profile"],
        "train_joint_mse": rows[0]["train_joint_mse"],
    }
    outputs = []
    specs = [
        ("noul_from_score_T", "p_noul", "pred_noul_from_score_t"),
        ("choice_from_score_T_over_TF", "p_choice", "pred_choice_from_score_tf"),
        ("unknown_from_score_U", "implied_unknown", "pred_unknown_from_score_u"),
    ]
    for target, observed_key, predicted_key in specs:
        valid = [row for row in rows if row[observed_key] is not None and row[predicted_key] is not None]
        observed = [row[observed_key] for row in valid]
        predicted = [row[predicted_key] for row in valid]
        metrics = _prediction_metrics(observed, predicted)
        outputs.append(
            {
                "target": target,
                **metrics,
                "corr": _correlation(observed, predicted),
                "mean_observed": _mean(observed),
                "mean_predicted": _mean(predicted),
                **model_fields,
            }
        )
    return outputs


def score_three_state_by_family(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)
    rows = []
    for family, family_records in sorted(grouped.items()):
        if len(family_records) < 8:
            continue
        for row in score_three_state_summary(family_records):
            rows.append({"family": family, **row})
    return rows


_PAIR_LEVELS = [(1, 8), (2, 7), (3, 6), (4, 5)]


def _fit_score_three_state_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    inits = [
        ([0.10, 0.20, 0.35, 0.80], [1 / 9, 2 / 9, 3 / 9, 4 / 9]),
        ([0.16, 0.24, 0.24, 1.00], [0.08, 0.20, 0.35, 0.45]),
    ]
    best_model = None
    best_loss = float("inf")
    for u_init, v_init in inits:
        candidate = _optimize_score_three_state_model(records, u_init, v_init)
        loss = candidate["train_joint_mse"]
        if loss < best_loss:
            best_loss = loss
            best_model = candidate
    assert best_model is not None
    return best_model


def _optimize_score_three_state_model(records: list[dict[str, Any]], u_init: list[float], v_init: list[float]) -> dict[str, Any]:
    u_params = _project_monotone(u_init)
    v_params = _project_monotone_half(v_init)
    if not records:
        return _score_three_state_model_from_params(u_params, v_params, 0.0)
    lr = 0.08
    for step in range(1200):
        grad_u = [0.0 for _ in range(4)]
        grad_v = [0.0 for _ in range(4)]
        model = _score_three_state_model_from_params(u_params, v_params, 0.0)
        for record in records:
            q = [record[f"q_{index}"] for index in range(10)]
            masses = _score_three_state_masses(q, model["u"], model["v"])
            t_mass = masses["T"]
            u_mass = masses["U"]
            denom = max(1e-9, 1.0 - u_mass)
            choice = t_mass / denom
            err_t = t_mass - record["p_noul"]
            err_c = choice - record["p_choice"]
            for pair_index, (low, high) in enumerate(_PAIR_LEVELS):
                d_u = q[low] + q[high]
                d_t_u = -q[low] * model["v"][low] - q[high] * model["v"][high]
                d_denom_u = -d_u
                d_choice_u = (d_t_u * denom - t_mass * d_denom_u) / (denom * denom)
                grad_u[pair_index] += (2.0 * err_t * d_t_u + 2.0 * err_c * d_choice_u) / len(records)

                scale_low = 1.0 - model["u"][low]
                scale_high = 1.0 - model["u"][high]
                d_t_v = q[low] * scale_low - q[high] * scale_high
                d_choice_v = d_t_v / denom
                grad_v[pair_index] += (2.0 * err_t * d_t_v + 2.0 * err_c * d_choice_v) / len(records)
        step_lr = lr / (1.0 + step / 1800.0)
        u_params = _project_monotone([u_params[index] - step_lr * grad_u[index] for index in range(4)])
        v_params = _project_monotone_half([v_params[index] - step_lr * grad_v[index] for index in range(4)])
    model = _score_three_state_model_from_params(u_params, v_params, 0.0)
    model["train_joint_mse"] = _score_three_state_joint_mse(records, model)
    return model


def _score_three_state_model_from_params(u_params: list[float], v_params: list[float], train_joint_mse: float) -> dict[str, Any]:
    u = _symmetric_weights(u_params)
    v = _symmetric_truth_share(v_params)
    t = [(1.0 - u[index]) * v[index] for index in range(10)]
    f = [(1.0 - u[index]) * (1.0 - v[index]) for index in range(10)]
    return {"u": u, "v": v, "t": t, "f": f, "train_joint_mse": train_joint_mse}


def _symmetric_truth_share(params: list[float]) -> list[float]:
    p = [min(0.5, max(0.0, value)) for value in params]
    return [0.0, p[0], p[1], p[2], p[3], 1.0 - p[3], 1.0 - p[2], 1.0 - p[1], 1.0 - p[0], 1.0]


def _project_monotone_half(values: list[float]) -> list[float]:
    projected = _project_monotone([2.0 * value for value in values])
    return [0.5 * value for value in projected]


def _score_three_state_joint_mse(records: list[dict[str, Any]], model: dict[str, Any]) -> float:
    if not records:
        return 0.0
    losses = []
    for record in records:
        masses = _predict_score_three_state(record, model)
        losses.append((masses["T"] - record["p_noul"]) ** 2)
        losses.append((masses["choice"] - record["p_choice"]) ** 2)
    return _mean(losses)


def _predict_score_three_state(record: dict[str, Any], model: dict[str, Any]) -> dict[str, float]:
    q = [record[f"q_{index}"] for index in range(10)]
    masses = _score_three_state_masses(q, model["u"], model["v"])
    denom = max(1e-9, masses["T"] + masses["F"])
    masses["choice"] = masses["T"] / denom
    return masses


def _score_three_state_masses(q: list[float], u: list[float], v: list[float]) -> dict[str, float]:
    t_mass = sum(q[index] * (1.0 - u[index]) * v[index] for index in range(10))
    u_mass = sum(q[index] * u[index] for index in range(10))
    f_mass = sum(q[index] * (1.0 - u[index]) * (1.0 - v[index]) for index in range(10))
    total = t_mass + u_mass + f_mass
    if total > 0:
        t_mass /= total
        u_mass /= total
        f_mass /= total
    return {"T": t_mass, "U": u_mass, "F": f_mass}


def score_choice_entropy_scan(records: list[dict[str, Any]], bin_counts: list[int] | None = None) -> list[dict[str, Any]]:
    if not records:
        return []
    bin_counts = bin_counts or list(range(2, 31))
    marginal = [_mean([record[f"q_{level}"] for record in records]) for level in range(10)]
    h_score = _entropy(marginal)
    rows = []
    for bins in bin_counts:
        grouped: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
        for record in records:
            index = min(bins - 1, max(0, int(record["p_choice"] * bins)))
            grouped[index].append(record)
        h_cond = 0.0
        nonempty = 0
        adjacent_wasserstein = []
        previous_q = None
        for group in grouped:
            if not group:
                continue
            nonempty += 1
            q_bar = [_mean([record[f"q_{level}"] for record in group]) for level in range(10)]
            weight = len(group) / len(records)
            h_cond += weight * _entropy(q_bar)
            if previous_q is not None:
                adjacent_wasserstein.append(_wasserstein_ordered(previous_q, q_bar))
            previous_q = q_bar
        mutual_information = max(0.0, h_score - h_cond)
        rows.append(
            {
                "choice_bins": bins,
                "nonempty_bins": nonempty,
                "h_score": h_score,
                "h_score_given_choice_bin": h_cond,
                "mutual_information": mutual_information,
                "normalized_mutual_information": mutual_information / h_score if h_score > 0 else 0.0,
                "mean_adjacent_wasserstein": _mean(adjacent_wasserstein),
                "median_adjacent_wasserstein": _median(adjacent_wasserstein),
                "mean_bin_count": len(records) / nonempty if nonempty else 0.0,
            }
        )
    return rows


def _entropy(distribution: list[float]) -> float:
    total = sum(max(0.0, value) for value in distribution)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in distribution:
        p = max(0.0, value) / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _wasserstein_ordered(left: list[float], right: list[float]) -> float:
    left_total = sum(max(0.0, value) for value in left)
    right_total = sum(max(0.0, value) for value in right)
    if left_total <= 0 or right_total <= 0:
        return 0.0
    cumulative = 0.0
    distance = 0.0
    for index in range(10):
        cumulative += max(0.0, left[index]) / left_total - max(0.0, right[index]) / right_total
        distance += abs(cumulative)
    return distance / 9.0




def score_unknown_mass_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        implied = _implied_unknown(record["p_noul"], record["p_choice"])
        if implied is None:
            continue
        score_fuzzy = _score_fuzzy_mass(record)
        score_entropy = _score_entropy_mass(record)
        score_mid_2_7 = sum(record[f"q_{index}"] for index in range(2, 8))
        score_mid_2_3 = record["q_2"] + record["q_3"]
        score_central_2 = record["q_4"] + record["q_5"]
        score_non_extreme = 1.0 - record["q_0"] - record["q_9"]
        output.append(
            {
                "instance_id": record["instance_id"],
                "latent_instance_id": record["latent_instance_id"],
                "family": record["family"],
                "representation": record["representation"],
                "p_star": record["p_star"],
                "p_noul": record["p_noul"],
                "p_choice": record["p_choice"],
                "implied_unknown": implied,
                "score_fuzzy_mass": score_fuzzy,
                "score_entropy_mass": score_entropy,
                "score_middle_2_7_mass": score_mid_2_7,
                "score_crisp_unknown_2_3_mass": score_mid_2_3,
                "score_central_2_mass": score_central_2,
                "score_non_extreme_mass": score_non_extreme,
                **{f"q_{index}": record[f"q_{index}"] for index in range(10)},
            }
        )
    return output


def score_unknown_mass_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unknown_records = score_unknown_mass_records(records)
    if not unknown_records:
        return []
    train, test = _split_unknown_records(unknown_records)
    fitted_weights = _fit_score_unknown_weights(train)
    gamma = _fit_score_unknown_gamma(train)
    constrained_weights = _fit_constrained_score_unknown_weights(train)
    central_decay_lambda = _fit_central_decay_lambda(train)
    candidates = [
        ("score_fuzzy_mass", [record["score_fuzzy_mass"] for record in test], {}),
        ("score_entropy_mass", [record["score_entropy_mass"] for record in test], {}),
        ("score_gamma_fuzzy_mass", [_score_gamma_fuzzy_mass(record, gamma) for record in test], {"gamma": gamma}),
        ("score_central_2_mass", [record["score_central_2_mass"] for record in test], {}),
        ("score_central_decay_mass", [_score_central_decay_mass(record, central_decay_lambda) for record in test], {"central_lambda": central_decay_lambda}),
        ("score_middle_2_7_mass", [record["score_middle_2_7_mass"] for record in test], {}),
        ("score_crisp_unknown_2_3_mass", [record["score_crisp_unknown_2_3_mass"] for record in test], {}),
        ("score_non_extreme_mass", [record["score_non_extreme_mass"] for record in test], {}),
        (
            "learned_score_unknown_weights",
            [_predict_score_unknown_weights(record, fitted_weights) for record in test],
            {
                **{f"w_{index}": fitted_weights[index] for index in range(10)},
                "weight_profile": _format_weights(fitted_weights),
            },
        ),
        (
            "learned_symmetric_monotone_weights",
            [_predict_score_unknown_weights(record, constrained_weights) for record in test],
            {
                **{f"w_{index}": constrained_weights[index] for index in range(10)},
                "weight_profile": _format_weights(constrained_weights),
            },
        ),
    ]
    y = [record["implied_unknown"] for record in test]
    rows = []
    for measure, predictions, extra in candidates:
        metrics = _prediction_metrics(y, predictions)
        rows.append(
            {
                "measure": measure,
                **metrics,
                "corr": _correlation(y, predictions),
                "mean_predicted_unknown": _mean(predictions),
                "mean_implied_unknown": _mean(y),
                **extra,
            }
        )
    return rows


def score_unknown_mass_by_family(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)
    rows = []
    for family, family_records in sorted(grouped.items()):
        if len(family_records) < 8:
            continue
        for row in score_unknown_mass_summary(family_records):
            rows.append({"family": family, **row})
    return rows


def _score_fuzzy_mass(record: dict[str, Any]) -> float:
    return sum(record[f"q_{index}"] * 4.0 * SCORE_VALUES[index] * (1.0 - SCORE_VALUES[index]) for index in range(10))


def _score_entropy_mass(record: dict[str, Any]) -> float:
    return sum(record[f"q_{index}"] * _binary_entropy_weight(SCORE_VALUES[index]) for index in range(10))


def _binary_entropy_weight(z: float) -> float:
    if z <= 0.0 or z >= 1.0:
        return 0.0
    return (-z * math.log2(z) - (1.0 - z) * math.log2(1.0 - z))


def _score_gamma_fuzzy_mass(record: dict[str, Any], gamma: float) -> float:
    return sum(record[f"q_{index}"] * (4.0 * SCORE_VALUES[index] * (1.0 - SCORE_VALUES[index])) ** gamma for index in range(10))


def _score_central_decay_mass(record: dict[str, Any], central_lambda: float) -> float:
    pairs = [(4, 5, 0), (3, 6, 1), (2, 7, 2), (1, 8, 3)]
    return sum((central_lambda**power) * (record[f"q_{left}"] + record[f"q_{right}"]) for left, right, power in pairs)


def _fit_central_decay_lambda(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    candidates = [index / 100 for index in range(101)]
    best = (float("inf"), 0.0)
    for central_lambda in candidates:
        mse = _mean([(_score_central_decay_mass(record, central_lambda) - record["implied_unknown"]) ** 2 for record in records])
        if mse < best[0]:
            best = (mse, central_lambda)
    return best[1]


def _split_unknown_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    test = []
    cutoff = 250
    for record in records:
        digest = hashlib.sha256(record["latent_instance_id"].encode()).hexdigest()
        bucket = int(digest[:8], 16) % 1000
        if bucket < cutoff:
            test.append(record)
        else:
            train.append(record)
    if not test and records:
        test = records[::4]
        test_ids = {id(record) for record in test}
        train = [record for record in records if id(record) not in test_ids]
    return train, test


def _fit_score_unknown_gamma(records: list[dict[str, Any]]) -> float:
    if not records:
        return 1.0
    candidates = [0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    best = (float("inf"), 1.0)
    for gamma in candidates:
        mse = _mean([(_score_gamma_fuzzy_mass(record, gamma) - record["implied_unknown"]) ** 2 for record in records])
        if mse < best[0]:
            best = (mse, gamma)
    return best[1]


def _fit_constrained_score_unknown_weights(records: list[dict[str, Any]]) -> list[float]:
    # Four free symmetric parameters: levels (1,8), (2,7), (3,6), (4,5).
    params = [0.25, 0.5, 0.75, 1.0]
    if not records:
        return _symmetric_weights(params)
    for _ in range(3000):
        gradients = [0.0 for _ in range(4)]
        weights = _symmetric_weights(params)
        for record in records:
            pred = _predict_score_unknown_weights(record, weights)
            err = pred - record["implied_unknown"]
            paired_q = [
                record["q_1"] + record["q_8"],
                record["q_2"] + record["q_7"],
                record["q_3"] + record["q_6"],
                record["q_4"] + record["q_5"],
            ]
            for index in range(4):
                gradients[index] += 2.0 * err * paired_q[index] / len(records)
        for index in range(4):
            params[index] = min(1.0, max(0.0, params[index] - 0.2 * gradients[index]))
        params = _project_monotone(params)
    return _symmetric_weights(params)


def _symmetric_weights(params: list[float]) -> list[float]:
    p = [min(1.0, max(0.0, value)) for value in params]
    return [0.0, p[0], p[1], p[2], p[3], p[3], p[2], p[1], p[0], 0.0]


def _fit_score_unknown_weights(records: list[dict[str, Any]]) -> list[float]:
    # Projected gradient fit for U ~= sum_j q_j w_j with 0<=w_j<=1 and endpoints fixed at 0.
    weights = [4.0 * value * (1.0 - value) for value in SCORE_VALUES]
    weights[0] = 0.0
    weights[9] = 0.0
    if not records:
        return weights
    for _ in range(3000):
        gradients = [0.0 for _ in range(10)]
        for record in records:
            pred = _predict_score_unknown_weights(record, weights)
            err = pred - record["implied_unknown"]
            for index in range(10):
                gradients[index] += 2.0 * err * record[f"q_{index}"] / len(records)
        for index in range(1, 9):
            weights[index] = min(1.0, max(0.0, weights[index] - 0.2 * gradients[index]))
        weights[0] = 0.0
        weights[9] = 0.0
    return weights


def _predict_score_unknown_weights(record: dict[str, Any], weights: list[float]) -> float:
    return sum(record[f"q_{index}"] * weights[index] for index in range(10))


def _format_weights(weights: list[float]) -> str:
    return ",".join(f"{weight:.3f}" for weight in weights)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    mean_left = _mean(left)
    mean_right = _mean(right)
    num = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    den_left = sum((x - mean_left) ** 2 for x in left)
    den_right = sum((y - mean_right) ** 2 for y in right)
    if den_left <= 0 or den_right <= 0:
        return 0.0
    return num / math.sqrt(den_left * den_right)

def noul_choice_calibration_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    models = _fit_noul_choice_models(train)
    rows = []
    for name, model in models.items():
        predictions = [_predict_noul_choice_model(record["p_noul"], name, model) for record in test]
        metrics = _prediction_metrics([record["p_choice"] for record in test], predictions)
        unknown_values = [_implied_unknown(record["p_noul"], prediction) for record, prediction in zip(test, predictions)]
        row = {
            "mapping": name,
            **metrics,
            "mean_implied_unknown": _mean([value for value in unknown_values if value is not None]),
            "median_implied_unknown": _median([value for value in unknown_values if value is not None]),
        }
        row.update(_model_parameters(name, model))
        rows.append(row)
    return rows


def noul_choice_calibration_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    train, test = _split_by_latent(records)
    models = _fit_noul_choice_models(train)
    rows = []
    for record in test:
        row = {
            "instance_id": record["instance_id"],
            "latent_instance_id": record["latent_instance_id"],
            "family": record["family"],
            "representation": record["representation"],
            "p_star": record["p_star"],
            "p_noul": record["p_noul"],
            "p_choice": record["p_choice"],
        }
        for name, model in models.items():
            prediction = _predict_noul_choice_model(record["p_noul"], name, model)
            row[f"pred_choice_{name}"] = prediction
            row[f"abs_error_{name}"] = abs(prediction - record["p_choice"])
            row[f"implied_unknown_{name}"] = _implied_unknown(record["p_noul"], prediction)
        rows.append(row)
    return rows


def noul_choice_calibration_by_family(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)
    rows = []
    for family, family_records in sorted(grouped.items()):
        if len(family_records) < 8:
            continue
        for row in noul_choice_calibration_summary(family_records):
            row = {"family": family, **row}
            rows.append(row)
    return rows


def _fit_noul_choice_models(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "identity": None,
        "power": _fit_power_mapping(records),
        "logit": _fit_logit_mapping(records),
        "beta": _fit_beta_calibration_mapping(records),
        "isotonic": _fit_isotonic_mapping(records),
    }


def _predict_noul_choice_model(p: float, name: str, model: Any) -> float:
    p = _clip_probability(p)
    if name == "identity":
        return p
    if name == "power":
        alpha = float(model["alpha"])
        return 1.0 - (1.0 - p) ** alpha
    if name == "logit":
        return _sigmoid(model["a"] * _logit(p) + model["b"])
    if name == "beta":
        return _sigmoid(model["a"] * math.log(p) + model["b"] * math.log(1.0 - p) + model["c"])
    if name == "isotonic":
        return _predict_isotonic(p, model)
    raise ValueError(f"unknown Noul-Choice calibration model: {name}")


def _fit_power_mapping(records: list[dict[str, Any]]) -> dict[str, float]:
    candidates = [0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    best = (float("inf"), 1.0)
    for alpha in candidates:
        mse = _mean([(1.0 - (1.0 - _clip_probability(record["p_noul"])) ** alpha - record["p_choice"]) ** 2 for record in records])
        if mse < best[0]:
            best = (mse, alpha)
    return {"alpha": best[1]}


def _fit_logit_mapping(records: list[dict[str, Any]]) -> dict[str, float]:
    xs = [_logit(record["p_noul"]) for record in records]
    ys = [_logit(record["p_choice"]) for record in records]
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 1e-12:
        return {"a": 1.0, "b": mean_y - mean_x}
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    a = cov / var_x
    b = mean_y - a * mean_x
    return {"a": a, "b": b}


def _fit_beta_calibration_mapping(records: list[dict[str, Any]], l2: float = 1e-6) -> dict[str, float]:
    dim = 3
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for record in records:
        p = _clip_probability(record["p_noul"])
        features = [math.log(p), math.log(1.0 - p), 1.0]
        target = _logit(record["p_choice"])
        for i in range(dim):
            xty[i] += features[i] * target
            for j in range(dim):
                xtx[i][j] += features[i] * features[j]
    for i in range(dim - 1):
        xtx[i][i] += l2
    a, b, c = _solve_linear_system(xtx, xty)
    return {"a": a, "b": b, "c": c}


def _fit_isotonic_mapping(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    pairs = sorted((_clip_probability(record["p_noul"]), record["p_choice"]) for record in records)
    blocks: list[dict[str, Any]] = []
    for x, y in pairs:
        blocks.append({"sum_x": x, "sum_y": y, "weight": 1, "value": y})
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            merged = {
                "sum_x": left["sum_x"] + right["sum_x"],
                "sum_y": left["sum_y"] + right["sum_y"],
                "weight": left["weight"] + right["weight"],
            }
            merged["value"] = merged["sum_y"] / merged["weight"]
            blocks.append(merged)
    return [
        {
            "x": block["sum_x"] / block["weight"],
            "value": min(1.0, max(0.0, block["value"])),
            "weight": block["weight"],
        }
        for block in blocks
    ]


def _predict_isotonic(p: float, blocks: list[dict[str, float]]) -> float:
    if not blocks:
        return p
    p = _clip_probability(p)
    if p <= blocks[0]["x"]:
        return blocks[0]["value"]
    if p >= blocks[-1]["x"]:
        return blocks[-1]["value"]
    for left, right in zip(blocks, blocks[1:]):
        if left["x"] <= p <= right["x"]:
            if right["x"] == left["x"]:
                return right["value"]
            weight = (p - left["x"]) / (right["x"] - left["x"])
            return left["value"] * (1.0 - weight) + right["value"] * weight
    return blocks[-1]["value"]


def _model_parameters(name: str, model: Any) -> dict[str, float | int]:
    if name == "identity" or model is None:
        return {}
    if name == "isotonic":
        return {"n_blocks": len(model)}
    return dict(model)


def _implied_unknown(p_noul: float, p_choice: float) -> float | None:
    if p_choice <= 1e-12:
        return None
    return max(0.0, min(1.0, 1.0 - _clip_probability(p_noul) / _clip_probability(p_choice)))


def _clip_probability(value: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, float(value)))


def _logit(value: float) -> float:
    p = _clip_probability(value)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)

def choice_from_score_group_summary(records: list[dict[str, Any]], group_key: str, min_records: int = 8) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_key, "unknown"))].append(record)
    rows = []
    for group, group_records in sorted(grouped.items()):
        if len(group_records) < min_records:
            rows.append(
                {
                    "group_key": group_key,
                    "group": group,
                    "rule": "insufficient_data",
                    "n_total": len(group_records),
                    "n_test": 0,
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                }
            )
            continue
        rows.extend(_choice_from_score_group_rows(group_records, group_key, group))
    return rows


def _choice_from_score_group_rows(records: list[dict[str, Any]], group_key: str, group: str) -> list[dict[str, Any]]:
    train, test = _split_by_latent(records)
    models = []
    models.append(("expected_truth", [record["mu_score"] for record in test], {}))
    models.append(("hard_threshold", [record["score_threshold"] for record in test], {}))
    false_max, true_min = _fit_crisp_unknown(train, "p_choice")
    models.append(
        (
            "crisp_unknown_conditional",
            [_predict_crisp_unknown(record, false_max, true_min)[0] for record in test],
            {"false_max_level": false_max, "true_min_level": true_min},
        )
    )
    alpha = _fit_monotonic_alpha(train, "p_choice")
    models.append(
        (
            "monotonic_collapse",
            [_predict_alpha(record, alpha) for record in test],
            {f"alpha_{index}": alpha[index] for index in range(10)},
        )
    )
    ridge = _fit_ridge_q(train, "p_choice")
    models.append(
        (
            "full_q_ridge",
            [_predict_ridge_q(record, ridge) for record in test],
            {"ridge_intercept": ridge[0], **{f"ridge_w_{index}": ridge[index + 1] for index in range(10)}},
        )
    )

    y = [record["p_choice"] for record in test]
    rows = []
    for rule, predictions, extra in models:
        metrics = _prediction_metrics(y, predictions)
        rows.append(
            {
                "group_key": group_key,
                "group": group,
                "rule": rule,
                "n_total": len(records),
                "n_train": len(train),
                **metrics,
                **extra,
            }
        )
    return rows

def representation_sensitivity_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["latent_instance_id"]].append(record)

    rows = []
    for primitive in ["noul", "choice", "score"]:
        pairwise = []
        max_by_latent = []
        for latent_records in grouped.values():
            if len(latent_records) < 2:
                continue
            distances = []
            for left, right in combinations(latent_records, 2):
                if primitive == "noul":
                    distance = abs(left["p_noul"] - right["p_noul"])
                elif primitive == "choice":
                    distance = abs(left["p_choice"] - right["p_choice"])
                else:
                    distance = 0.5 * sum(abs(left[f"q_{index}"] - right[f"q_{index}"]) for index in range(10))
                distances.append(distance)
                pairwise.append(distance)
            if distances:
                max_by_latent.append(max(distances))
        rows.append(
            {
                "primitive": primitive,
                "n_latents": len(max_by_latent),
                "mean_pairwise_distance": _mean(pairwise),
                "median_pairwise_distance": _median(pairwise),
                "mean_max_distance": _mean(max_by_latent),
                "max_distance": max(max_by_latent) if max_by_latent else 0.0,
            }
        )
    return rows


def _positive_probability(distribution: dict[str, float]) -> float:
    if "True" in distribution:
        return float(distribution["True"])
    return float(distribution[list(distribution)[0]])


def _score_distribution(row: dict[str, Any]) -> list[float]:
    answer = ((row.get("raw_response") or {}).get("answers") or {}).get("truth_degree") or {}
    raw = answer.get("probabilities") or answer.get("distribution") or {}
    values = [max(0.0, float(raw.get(str(index), 0.0))) for index in range(10)]
    total = sum(values)
    if total > 0:
        return [value / total for value in values]
    return [0.0 for _ in range(10)]


def _error_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mae": _mean(values),
        "rmse": math.sqrt(_mean([value * value for value in values])),
        "mean_brier_regret": 2.0 * _mean([value * value for value in values]),
        "mean_tv": _mean(values),
        "median": _median(values),
    }


def _signed_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "mean_abs": _mean([abs(value) for value in values]),
        "rmse": math.sqrt(_mean([value * value for value in values])),
    }


def _projection_row(model: str, target_name: str, records: list[dict[str, Any]], predictions: list[float], extra: dict[str, Any] | None) -> dict[str, Any]:
    target_key = "p_noul" if target_name == "noul" else "p_choice"
    y = [record[target_key] for record in records]
    metrics = _prediction_metrics(y, predictions)
    row = {"collapse_rule": model, "target": target_name, **metrics}
    if extra:
        row.update(extra)
    return row


def _prediction_metrics(y: list[float], y_hat: list[float]) -> dict[str, float | int]:
    if not y:
        return {"n_train": None, "n_test": 0, "mae": 0.0, "rmse": 0.0, "r2": 0.0}
    errors = [pred - true for true, pred in zip(y, y_hat)]
    sse = sum(error * error for error in errors)
    mean_y = _mean(y)
    sst = sum((true - mean_y) ** 2 for true in y)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return {
        "n_test": len(y),
        "mae": _mean([abs(error) for error in errors]),
        "rmse": math.sqrt(_mean([error * error for error in errors])),
        "r2": r2,
    }


def _split_by_latent(records: list[dict[str, Any]], test_fraction: float = 0.25) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    test = []
    cutoff = int(test_fraction * 1000)
    for record in records:
        digest = hashlib.sha256(record["latent_instance_id"].encode()).hexdigest()
        bucket = int(digest[:8], 16) % 1000
        if bucket < cutoff:
            test.append(record)
        else:
            train.append(record)
    if not test and records:
        test = records[::4]
        test_ids = {id(record) for record in test}
        train = [record for record in records if id(record) not in test_ids]
    return train, test


def _fit_beta(records: list[dict[str, Any]], target_key: str) -> float:
    candidates = [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0]
    best = (float("inf"), 1.0)
    for beta in candidates:
        mse = _mean([(_predict_beta(record, beta) - record[target_key]) ** 2 for record in records])
        if mse < best[0]:
            best = (mse, beta)
    return best[1]


def _predict_beta(record: dict[str, Any], beta: float) -> float:
    return sum(record[f"q_{index}"] * _g_beta(SCORE_VALUES[index], beta) for index in range(10))


def _g_beta(z: float, beta: float) -> float:
    if z <= 0:
        return 0.0
    if z >= 1:
        return 1.0
    num = z**beta
    return num / (num + (1.0 - z) ** beta)


def _fit_monotonic_alpha(records: list[dict[str, Any]], target_key: str, iterations: int = 4000, step: float = 0.25) -> list[float]:
    alpha = list(SCORE_VALUES)
    if not records:
        return alpha
    for _ in range(iterations):
        gradients = [0.0 for _ in range(10)]
        for record in records:
            pred = _predict_alpha(record, alpha)
            error = pred - record[target_key]
            for index in range(10):
                gradients[index] += 2.0 * error * record[f"q_{index}"] / len(records)
        updated = alpha[:]
        for index in range(1, 9):
            updated[index] -= step * gradients[index]
        updated[0] = 0.0
        updated[9] = 1.0
        updated[1:9] = _project_monotone(updated[1:9])
        alpha = updated
    return alpha


def _predict_alpha(record: dict[str, Any], alpha: list[float]) -> float:
    return sum(record[f"q_{index}"] * alpha[index] for index in range(10))


def _project_monotone(values: list[float]) -> list[float]:
    blocks = []
    for value in values:
        blocks.append([min(1.0, max(0.0, value)), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            left_value, left_weight = blocks.pop(-2)
            right_value, right_weight = blocks.pop(-1)
            weight = left_weight + right_weight
            blocks.append([(left_value * left_weight + right_value * right_weight) / weight, weight])
    projected = []
    for value, weight in blocks:
        projected.extend([min(1.0, max(0.0, value))] * weight)
    return projected


def _fit_crisp_unknown(records: list[dict[str, Any]], target_key: str) -> tuple[int, int]:
    best = (float("inf"), 1, 2)
    for false_max in range(0, 8):
        for true_min in range(false_max + 1, 10):
            mse = _mean([(_predict_crisp_unknown(record, false_max, true_min)[0] - record[target_key]) ** 2 for record in records])
            if mse < best[0]:
                best = (mse, false_max, true_min)
    return best[1], best[2]


def _predict_crisp_unknown(record: dict[str, Any], false_max: int, true_min: int) -> tuple[float, float]:
    false_mass = sum(record[f"q_{index}"] for index in range(0, false_max + 1))
    true_mass = sum(record[f"q_{index}"] for index in range(true_min, 10))
    unknown_mass = max(0.0, 1.0 - false_mass - true_mass)
    denominator = false_mass + true_mass
    if denominator <= 1e-12:
        return 0.5, unknown_mass
    return true_mass / denominator, unknown_mass


def _fit_fuzzy_unknown(records: list[dict[str, Any]], target_key: str) -> tuple[float, float, float]:
    beta_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    lambda_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    gamma_values = [0.5, 1.0, 2.0, 3.0]
    best = (float("inf"), 1.0, 0.0, 1.0)
    for beta in beta_values:
        for unknown_lambda in lambda_values:
            for gamma in gamma_values:
                params = (beta, unknown_lambda, gamma)
                mse = _mean([(_predict_fuzzy_unknown(record, params)[0] - record[target_key]) ** 2 for record in records])
                if mse < best[0]:
                    best = (mse, beta, unknown_lambda, gamma)
    return best[1], best[2], best[3]


def _predict_fuzzy_unknown(record: dict[str, Any], params: tuple[float, float, float]) -> tuple[float, float]:
    beta, unknown_lambda, gamma = params
    false_mass = 0.0
    true_mass = 0.0
    unknown_mass = 0.0
    for index, z in enumerate(SCORE_VALUES):
        q = record[f"q_{index}"]
        false_raw = (1.0 - z) ** beta
        true_raw = z**beta
        unknown_raw = unknown_lambda * (4.0 * z * (1.0 - z)) ** gamma if 0.0 < z < 1.0 else 0.0
        total = false_raw + true_raw + unknown_raw
        if total <= 0:
            continue
        false_mass += q * false_raw / total
        true_mass += q * true_raw / total
        unknown_mass += q * unknown_raw / total
    denominator = false_mass + true_mass
    if denominator <= 1e-12:
        return 0.5, unknown_mass
    return true_mass / denominator, unknown_mass


def _fit_ridge_q(records: list[dict[str, Any]], target_key: str, l2: float = 0.001) -> list[float]:
    if not records:
        return [0.0] + list(SCORE_VALUES)
    dim = 11
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for record in records:
        features = [1.0] + [record[f"q_{index}"] for index in range(10)]
        target = record[target_key]
        for i in range(dim):
            xty[i] += features[i] * target
            for j in range(dim):
                xtx[i][j] += features[i] * features[j]
    for i in range(1, dim):
        xtx[i][i] += l2
    return _solve_linear_system(xtx, xty)


def _predict_ridge_q(record: dict[str, Any], weights: list[float]) -> float:
    value = weights[0] + sum(weights[index + 1] * record[f"q_{index}"] for index in range(10))
    return min(1.0, max(0.0, value))


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[col][col] += 1e-8
            pivot = col
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[row][n] for row in range(n)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2

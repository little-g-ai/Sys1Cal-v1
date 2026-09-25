from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_prob_bench.evaluation.aggregation import aggregate
from jev_prob_bench.evaluation.benchmark_summary import benchmark_summary
from jev_prob_bench.evaluation.calibration import simulate_calibration
from jev_prob_bench.evaluation.representation import aggregate_representation_sensitivity, representation_sensitivity
from jev_prob_bench.evaluation.repeats import aggregate_repeats, repeat_count_summary
from jev_prob_bench.evaluation.noul_choice_score import (
    build_ncs_records,
    choice_from_score_recovery,
    choice_from_score_group_summary,
    noul_choice_calibration_by_family,
    noul_choice_calibration_predictions,
    noul_choice_calibration_summary,
    projection_model_summary,
    representation_sensitivity_summary as ncs_representation_sensitivity_summary,
    score_choice_entropy_scan,
    score_choice_tuf_by_family,
    score_choice_tuf_predictions,
    score_choice_tuf_summary,
    score_three_state_by_family,
    score_three_state_records,
    score_three_state_summary,
    score_unknown_mass_by_family,
    score_unknown_mass_records,
    score_unknown_mass_summary,
    summarize_ncs,
)
from jev_prob_bench.reporting.plots import build_plots
from jev_prob_bench.reporting.tables import markdown_table, write_csv


def build_report(results_dir: str | Path, output_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _enrich_rows_with_dataset(_load_jsonl(results_dir / "results.jsonl"), results_dir)
    repeat_mean_rows = aggregate_repeats(rows, "mean")
    repeat_median_rows = aggregate_repeats(rows, "median")
    repeat_aggregate_rows = repeat_mean_rows + repeat_median_rows
    repeat_counts = repeat_count_summary(rows)
    overall = aggregate(repeat_mean_rows, ["model"])
    by_family = aggregate(repeat_mean_rows, ["model", "family"])
    by_representation = aggregate(repeat_mean_rows, ["model", "representation"])
    repeat_overall = aggregate(repeat_aggregate_rows, ["repeat_aggregation", "model"])
    repeat_by_family = aggregate(repeat_aggregate_rows, ["repeat_aggregation", "model", "family"])
    repeat_by_representation = aggregate(repeat_aggregate_rows, ["repeat_aggregation", "model", "representation"])
    rep_rows = representation_sensitivity(repeat_mean_rows)
    rep_summary = aggregate_representation_sensitivity(rep_rows)
    calibration = simulate_calibration(repeat_mean_rows)
    powerlaw_calibration = _powerlaw_calibration_summary(repeat_mean_rows)
    ncs_records = build_ncs_records(repeat_mean_rows)
    headline_summary = benchmark_summary(repeat_mean_rows, ncs_records)
    ncs_summary = summarize_ncs(ncs_records)
    ncs_projection = projection_model_summary(ncs_records)
    score_unknown_records = score_unknown_mass_records(ncs_records)
    score_unknown_summary = score_unknown_mass_summary(ncs_records)
    score_unknown_family = score_unknown_mass_by_family(ncs_records)
    score_three_state = score_three_state_summary(ncs_records)
    score_three_state_predictions = score_three_state_records(ncs_records)
    score_three_state_family = score_three_state_by_family(ncs_records)
    score_choice_tuf = score_choice_tuf_summary(ncs_records)
    score_choice_tuf_family = score_choice_tuf_by_family(ncs_records)
    score_choice_tuf_prediction_rows = score_choice_tuf_predictions(ncs_records)
    noul_choice_calibration = noul_choice_calibration_summary(ncs_records)
    noul_choice_predictions = noul_choice_calibration_predictions(ncs_records)
    noul_choice_by_family = noul_choice_calibration_by_family(ncs_records)
    choice_recovery = choice_from_score_recovery(ncs_records)
    score_choice_entropy = score_choice_entropy_scan(ncs_records)
    choice_recovery_by_representation = choice_from_score_group_summary(ncs_records, "representation")
    choice_recovery_by_family = choice_from_score_group_summary(ncs_records, "family")
    ncs_representation = ncs_representation_sensitivity_summary(ncs_records)

    write_csv(output_dir / "overall_model_comparison.csv", overall)
    write_csv(output_dir / "benchmark_summary.csv", headline_summary)
    write_csv(output_dir / "by_family.csv", by_family)
    write_csv(output_dir / "by_representation.csv", by_representation)
    write_csv(output_dir / "repeat_count_summary.csv", repeat_counts)
    write_csv(output_dir / "repeat_aggregated_instance_metrics.csv", repeat_aggregate_rows)
    write_csv(output_dir / "repeat_aggregated_model_comparison.csv", repeat_overall)
    write_csv(output_dir / "repeat_aggregated_by_family.csv", repeat_by_family)
    write_csv(output_dir / "repeat_aggregated_by_representation.csv", repeat_by_representation)
    write_csv(output_dir / "representation_sensitivity.csv", rep_summary)
    write_csv(output_dir / "representation_sensitivity_by_latent.csv", rep_rows)
    write_csv(output_dir / "powerlaw_calibration.csv", powerlaw_calibration)
    write_csv(output_dir / "noul_choice_score_instance_metrics.csv", ncs_records)
    write_csv(output_dir / "noul_choice_score_summary.csv", ncs_summary)
    write_csv(output_dir / "noul_choice_score_projection_models.csv", ncs_projection)
    write_csv(output_dir / "score_unknown_mass_records.csv", score_unknown_records)
    write_csv(output_dir / "score_unknown_mass_summary.csv", score_unknown_summary)
    write_csv(output_dir / "score_unknown_mass_by_family.csv", score_unknown_family)
    write_csv(output_dir / "score_three_state_summary.csv", score_three_state)
    write_csv(output_dir / "score_three_state_predictions.csv", score_three_state_predictions)
    write_csv(output_dir / "score_three_state_by_family.csv", score_three_state_family)
    write_csv(output_dir / "score_choice_tuf_summary.csv", score_choice_tuf)
    write_csv(output_dir / "score_choice_tuf_by_family.csv", score_choice_tuf_family)
    write_csv(output_dir / "score_choice_tuf_predictions.csv", score_choice_tuf_prediction_rows)
    write_csv(output_dir / "noul_choice_calibration.csv", noul_choice_calibration)
    write_csv(output_dir / "noul_choice_calibration_predictions.csv", noul_choice_predictions)
    write_csv(output_dir / "noul_choice_calibration_by_family.csv", noul_choice_by_family)
    write_csv(output_dir / "choice_from_score_recovery.csv", choice_recovery)
    write_csv(output_dir / "score_choice_entropy_scan.csv", score_choice_entropy)
    write_csv(output_dir / "choice_from_score_by_representation.csv", choice_recovery_by_representation)
    write_csv(output_dir / "choice_from_score_by_family.csv", choice_recovery_by_family)
    write_csv(output_dir / "noul_choice_score_representation_sensitivity.csv", ncs_representation)
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    figures = build_plots(repeat_mean_rows, output_dir / "figures", calibration)

    report = "\n\n".join(
        [
            "# Jev Probability Benchmark Report",
            "Log-score regret uses epsilon clipping at 1e-12. Argmax accuracy is secondary; TV distance is the primary recovery metric. For repeated Jev runs, headline tables use the per-question mean prediction over repeats unless stated otherwise.",
            "## Benchmark Summary\n" + markdown_table(headline_summary, ["model", "response_type", "n", "argmax_accuracy", "pointwise_probability_fidelity", "mean_tv_error", "noul_choice_r2", "noul_score_r2", "score_choice_r2"]),
            "## Repeat Coverage\n" + markdown_table(repeat_counts, ["model", "n_items", "min_repeats", "median_repeats", "max_repeats", "mean_repeats"]),
            "## Repeat-Aggregated Model Comparison\n" + markdown_table(repeat_overall, ["repeat_aggregation", "model", "n", "mean_tv", "median_tv", "mean_brier_regret", "mean_kl", "argmax_accuracy"]),
            "## Overall Model Comparison\n" + markdown_table(overall, ["model", "n", "mean_tv", "median_tv", "mean_brier_regret", "mean_kl", "argmax_accuracy"]),
            "## By Family\n" + markdown_table(by_family, ["model", "family", "n", "mean_tv", "mean_mae", "mean_brier_regret", "argmax_accuracy"]),
            "## By Representation\n" + markdown_table(by_representation, ["model", "representation", "n", "mean_tv", "mean_brier_regret"]),
            "## Representation Sensitivity\n" + markdown_table(rep_summary, ["model", "mean_pairwise_tv", "mean_max_tv", "ris", "mean_gold_relative_gap"]),
            "## Power-Law Calibration\n" + markdown_table(powerlaw_calibration, ["model", "n", "alpha", "identity_mae", "power_mae_to_returned", "inverse_calibrated_mae", "identity_rmse", "power_rmse_to_returned", "inverse_calibrated_rmse"]),
            _ncs_report_section(
                ncs_summary,
                ncs_projection,
                score_unknown_summary,
                score_unknown_family,
                score_three_state,
                score_three_state_family,
                score_choice_tuf,
                score_choice_tuf_family,
                noul_choice_calibration,
                noul_choice_by_family,
                choice_recovery,
                score_choice_entropy,
                choice_recovery_by_representation,
                choice_recovery_by_family,
                ncs_representation,
            ),
            f"## Calibration\nECE: {calibration['ece']:.6g}\n\nEmpirical Brier: {calibration['empirical_brier']:.6g}\n\nEmpirical log loss: {calibration['empirical_log_loss']:.6g}",
            "## Figures\n" + "\n".join(f"- {path}" for path in figures),
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    return report_path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def _enrich_rows_with_dataset(rows: list[dict[str, Any]], results_dir: Path) -> list[dict[str, Any]]:
    dataset_pointer = results_dir / "dataset.path"
    if not dataset_pointer.exists():
        return rows
    dataset_path = Path(dataset_pointer.read_text(encoding="utf-8").strip())
    if not dataset_path.exists():
        return rows
    dataset_rows = {row["instance_id"]: row for row in _load_jsonl(dataset_path)}
    for row in rows:
        source = dataset_rows.get(row.get("instance_id"))
        if not source:
            continue
        row.setdefault("generator_metadata", source.get("generator_metadata", {}))
        row.setdefault("state", source.get("state", {}))
        row.setdefault("prompt", source.get("prompt"))
    return rows



def _ncs_report_section(
    summary: list[dict[str, Any]],
    projection: list[dict[str, Any]],
    score_unknown_summary: list[dict[str, Any]],
    score_unknown_family: list[dict[str, Any]],
    score_three_state: list[dict[str, Any]],
    score_three_state_family: list[dict[str, Any]],
    score_choice_tuf: list[dict[str, Any]],
    score_choice_tuf_family: list[dict[str, Any]],
    noul_choice_calibration: list[dict[str, Any]],
    noul_choice_by_family: list[dict[str, Any]],
    choice_recovery: list[dict[str, Any]],
    score_choice_entropy: list[dict[str, Any]],
    choice_by_representation: list[dict[str, Any]],
    choice_by_family: list[dict[str, Any]],
    representation: list[dict[str, Any]],
) -> str:
    if not summary:
        return "## Noul-Choice-Score Evaluation\n_No complete Noul/Choice/Score triples found._"
    projection_fields = ["collapse_rule", "target", "n_test", "mae", "rmse", "r2", "beta", "fuzzy_beta", "unknown_lambda", "unknown_gamma", "false_max_level", "true_min_level", "n_test_eligible"]
    return "\n\n".join(
        [
            "## Noul-Choice-Score Evaluation",
            "### External Fidelity and Cross-Primitive Consistency\n"
            + markdown_table(summary, ["section", "metric", "n", "mae", "rmse", "mean_tv", "mean", "mean_abs", "median"]),
            "### Score Projection Models\n" + markdown_table(projection, projection_fields),
            "### Score-Derived Unknown Mass\n"
            + markdown_table(score_unknown_summary, ["measure", "n_test", "mae", "rmse", "r2", "corr", "mean_predicted_unknown", "mean_implied_unknown", "gamma", "central_lambda", "weight_profile"]),
            "### Score-Derived Unknown Mass by Family\n"
            + markdown_table(score_unknown_family, ["family", "measure", "n_test", "mae", "rmse", "r2", "corr", "mean_predicted_unknown", "mean_implied_unknown", "gamma", "central_lambda", "weight_profile"]),
            "### Score Three-State Decomposition\n"
            + markdown_table(score_three_state, ["target", "n_test", "mae", "rmse", "r2", "corr", "mean_observed", "mean_predicted", "train_joint_mse", "u_profile", "t_profile", "f_profile"]),
            "### Score Three-State Decomposition by Family\n"
            + markdown_table(score_three_state_family, ["family", "target", "n_test", "mae", "rmse", "r2", "corr", "mean_observed", "mean_predicted", "u_profile"]),
            "### Score to Choice via T/U/F\n"
            + markdown_table(score_choice_tuf, ["truth_model", "unknown_rule", "n_test", "mae", "rmse", "r2", "corr", "mean_observed_choice", "mean_predicted_choice", "mean_predicted_unknown", "central_lambda", "u_profile"]),
            "### Score to Choice via T/U/F by Family\n"
            + markdown_table(score_choice_tuf_family, ["family", "truth_model", "unknown_rule", "n_test", "mae", "rmse", "r2", "corr", "mean_observed_choice", "mean_predicted_choice", "mean_predicted_unknown", "central_lambda", "u_profile"]),
            "### Noul to Choice Calibration\n"
            + markdown_table(noul_choice_calibration, ["mapping", "n_test", "mae", "rmse", "r2", "alpha", "a", "b", "c", "n_blocks", "mean_implied_unknown"]),
            "### Noul to Choice Calibration by Family\n"
            + markdown_table(noul_choice_by_family, ["family", "mapping", "n_test", "mae", "rmse", "r2", "alpha", "a", "b", "c", "n_blocks", "mean_implied_unknown"]),
            "### Choice From Score Recovery\n"
            + markdown_table(_choice_recovery_summary(choice_recovery), ["rule", "n", "mae", "rmse", "r2"]),
            "### Multiscale Score-Choice Entropy\n"
            + markdown_table(_selected_entropy_rows(score_choice_entropy), ["choice_bins", "nonempty_bins", "h_score", "h_score_given_choice_bin", "mutual_information", "normalized_mutual_information", "mean_adjacent_wasserstein"]),
            "### Score to Choice by Representation\n"
            + markdown_table(_compact_group_rows(choice_by_representation), ["group", "rule", "n_total", "n_test", "mae", "rmse", "r2", "false_max_level", "true_min_level"]),
            "### Score to Choice by Family\n"
            + markdown_table(_compact_group_rows(choice_by_family), ["group", "rule", "n_total", "n_test", "mae", "rmse", "r2", "false_max_level", "true_min_level"]),
            "### Primitive Representation Sensitivity\n"
            + markdown_table(representation, ["primitive", "n_latents", "mean_pairwise_distance", "median_pairwise_distance", "mean_max_distance", "max_distance"]),
        ]
    )



def _choice_recovery_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = [
        ("expected_truth", "pred_choice_expected_truth"),
        ("hard_threshold", "pred_choice_hard_threshold"),
        ("endpoint_conditioning", "pred_choice_endpoint_conditioning"),
        ("parametric_sharpening", "pred_choice_parametric_sharpening"),
        ("monotonic_collapse", "pred_choice_monotonic_collapse"),
        ("full_q_ridge", "pred_choice_full_q_ridge"),
        ("crisp_unknown_conditional", "pred_choice_crisp_unknown_conditional"),
        ("fuzzy_unknown_conditional", "pred_choice_fuzzy_unknown_conditional"),
    ]
    summary = []
    for name, key in rules:
        pairs = [(row["p_choice"], row[key]) for row in rows if row.get(key) is not None]
        if not pairs:
            summary.append({"rule": name, "n": 0, "mae": 0.0, "rmse": 0.0, "r2": 0.0})
            continue
        y = [pair[0] for pair in pairs]
        errors = [pred - true for true, pred in pairs]
        mean_y = sum(y) / len(y)
        sse = sum(error * error for error in errors)
        sst = sum((true - mean_y) ** 2 for true in y)
        summary.append(
            {
                "rule": name,
                "n": len(pairs),
                "mae": sum(abs(error) for error in errors) / len(errors),
                "rmse": (sum(error * error for error in errors) / len(errors)) ** 0.5,
                "r2": 1.0 - sse / sst if sst > 0 else 0.0,
            }
        )
    return summary



def _selected_entropy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    preferred = {2, 3, 4, 5, 8, 10, 15, 20, 25, 30}
    return [row for row in rows if row.get("choice_bins") in preferred]



def _compact_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"expected_truth", "crisp_unknown_conditional", "monotonic_collapse", "full_q_ridge", "insufficient_data"}
    return [row for row in rows if row.get("rule") in wanted]


def _powerlaw_calibration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("error") or len(row.get("gold_distribution", {})) != 2:
            continue
        grouped.setdefault(row["model"], []).append(row)
    output = []
    for model, model_rows in sorted(grouped.items()):
        pairs = [(_positive_probability(row["gold_distribution"]), _positive_probability(row["predicted_distribution"])) for row in model_rows]
        if not pairs:
            continue
        alpha = _fit_powerlaw_alpha(pairs)
        identity_errors = [pred - gold for gold, pred in pairs]
        power_errors = [_powerlaw_forward(gold, alpha) - pred for gold, pred in pairs]
        inverse_errors = [_powerlaw_inverse(pred, alpha) - gold for gold, pred in pairs]
        output.append(
            {
                "model": model,
                "n": len(pairs),
                "alpha": alpha,
                "identity_mae": _mean_abs(identity_errors),
                "power_mae_to_returned": _mean_abs(power_errors),
                "inverse_calibrated_mae": _mean_abs(inverse_errors),
                "identity_rmse": _rmse(identity_errors),
                "power_rmse_to_returned": _rmse(power_errors),
                "inverse_calibrated_rmse": _rmse(inverse_errors),
            }
        )
    return output


def _fit_powerlaw_alpha(pairs: list[tuple[float, float]]) -> float:
    best = (float("inf"), 1.0)
    candidates = [0.05 + index * 0.01 for index in range(2996)]
    for alpha in candidates:
        mse = sum((_powerlaw_forward(gold, alpha) - pred) ** 2 for gold, pred in pairs) / len(pairs)
        if mse < best[0]:
            best = (mse, alpha)
    return best[1]


def _powerlaw_forward(p: float, alpha: float) -> float:
    p = min(1.0, max(0.0, float(p)))
    return 1.0 - (1.0 - p) ** alpha


def _powerlaw_inverse(q: float, alpha: float) -> float:
    q = min(1.0, max(0.0, float(q)))
    if alpha <= 0:
        return q
    return 1.0 - (1.0 - q) ** (1.0 / alpha)


def _positive_probability(distribution: dict[str, float]) -> float:
    if "True" in distribution:
        return float(distribution["True"])
    return float(distribution[list(distribution)[0]])


def _mean_abs(errors: list[float]) -> float:
    return sum(abs(error) for error in errors) / len(errors) if errors else 0.0


def _rmse(errors: list[float]) -> float:
    return (sum(error * error for error in errors) / len(errors)) ** 0.5 if errors else 0.0

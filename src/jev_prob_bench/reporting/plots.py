from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any

from jev_prob_bench.evaluation.noul_choice_score import build_ncs_records, choice_from_score_recovery, choice_from_score_group_summary, score_choice_entropy_scan, score_choice_tuf_predictions, score_three_state_records, score_three_state_summary, score_unknown_mass_records, score_unknown_mass_summary


PRIMITIVE_STYLES = {
    "jev_noul": {"label": r"Jev-$\mathtt{Noul}$", "color": "tab:blue", "marker": "o", "linestyle": "-"},
    "jev_choice": {"label": r"Jev-$\mathtt{Choice}$", "color": "tab:orange", "marker": "s", "linestyle": "--"},
    "jev_score": {"label": r"Jev-$\mathtt{Score}$", "color": "tab:green", "marker": "^", "linestyle": "-."},
    "semif_qwen35_4b": {"label": r"SemIf-$\mathtt{Choice}$", "color": "tab:purple", "marker": "D", "linestyle": ":"},
}


def build_plots(rows: list[dict[str, Any]], output_dir: str | Path, calibration: dict[str, Any]) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        note = output / "figures_unavailable.txt"
        note.write_text("Install matplotlib to generate PNG figures.\n", encoding="utf-8")
        return [str(note)]
    paths = []
    ok = [row for row in rows if not row.get("error")]
    paths.append(_transfer_curve(plt, ok, output))
    paths.append(_powerlaw_calibration_plot(plt, ok, output))
    paths.append(_powerlaw_calibration_plot(plt, ok, output, family="explicit_probability", filename="figure_16b_powerlaw_calibration_explicit_probability.png"))
    paths.append(_error_by_family(plt, ok, output))
    paths.append(_error_by_representation(plt, ok, output))
    paths.append(_equivalent_spread(plt, ok, output))
    paths.append(_sequential_trajectories(plt, ok, output))
    paths.append(_calibration_curve(plt, ok, calibration, output))
    paths.append(_calibration_by_representation(plt, ok, output))
    ncs_records = build_ncs_records(ok)
    if ncs_records:
        paths.append(_ncs_consistency_scatter(plt, ncs_records, output))
        paths.append(_score_choice_uncertainty_mapping_plot(plt, ncs_records, output))
        paths.append(_ncs_score_expectation_scatter(plt, ncs_records, output))
        paths.append(_choice_from_score_recovery_plot(plt, choice_from_score_recovery(ncs_records), output))
        paths.append(_score_levels_vs_choice_heatmap(plt, ncs_records, output))
        paths.append(_score_choice_entropy_plot(plt, score_choice_entropy_scan(ncs_records), output))
        unknown_records = score_unknown_mass_records(ncs_records)
        unknown_summary = score_unknown_mass_summary(ncs_records)
        paths.append(_score_unknown_scatter(plt, unknown_records, output))
        paths.append(_score_unknown_weights_plot(plt, unknown_summary, output))
        paths.append(_score_choice_tuf_plot(plt, score_choice_tuf_predictions(ncs_records), output))
        grouped_by_representation = _choice_group_summary(ncs_records, "representation")
        grouped_by_family = _choice_group_summary(ncs_records, "family")
        paths.append(_choice_group_mae_plot(plt, grouped_by_representation, output, "representation", "figure_12_choice_from_score_by_representation.png"))
        paths.append(_choice_group_mae_plot(plt, grouped_by_family, output, "family", "figure_13_choice_from_score_by_family.png"))
        paths.append(_crisp_unknown_threshold_plot(plt, grouped_by_representation, output, "figure_14_crisp_unknown_thresholds_by_representation.png"))
    return [str(path) for path in paths if path]


def _primitive_style(model: str) -> dict[str, str]:
    if model in PRIMITIVE_STYLES:
        return PRIMITIVE_STYLES[model]
    return {"label": model, "color": "tab:gray", "marker": "o", "linestyle": ":"}


def _ordered_models(rows) -> list[str]:
    present = {row["model"] for row in rows}
    ordered = [model for model in PRIMITIVE_STYLES if model in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _positive_label(row: dict[str, Any]) -> str:
    outcomes = row.get("outcomes") or list(row["gold_distribution"])
    if "True" in outcomes:
        return "True"
    return outcomes[0]


def _transfer_curve(plt, rows, output: Path):
    data = [r for r in rows if len(r["gold_distribution"]) == 2]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model in _ordered_models(data):
        model_rows = [r for r in data if r["model"] == model]
        if not model_rows:
            continue
        style = _primitive_style(model)
        xs = [r["gold_distribution"][_positive_label(r)] for r in model_rows]
        ys = [r["predicted_distribution"][_positive_label(r)] for r in model_rows]
        mean_tv = sum(abs(y - x) for x, y in zip(xs, ys)) / len(xs)
        label = f"{style['label']} (E[TV]={mean_tv:.3f})"
        ax.scatter(xs, ys, s=14, alpha=0.35, color=style["color"], marker=style["marker"], label=label)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("True P(proposition)")
    ax.set_ylabel("Returned P(proposition)")
    ax.set_title("Probability Transfer: all Sys1Cal families")
    ax.grid(True, alpha=0.2)
    ax.legend(title="Primitive", frameon=False)
    path = output / "figure_1_probability_transfer.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _powerlaw_calibration_plot(plt, rows, output: Path, family: str | None = None, filename: str = "figure_16_powerlaw_calibration.png"):
    data = [row for row in rows if len(row.get("gold_distribution", {})) == 2 and (family is None or row.get("family") == family)]
    if not data:
        return None
    models = _ordered_models(data)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharex=True, sharey=True)
    curve_x = [index / 200 for index in range(201)]
    for model in models:
        model_rows = [row for row in data if row["model"] == model]
        pairs = [
            (row["gold_distribution"][_positive_label(row)], row["predicted_distribution"][_positive_label(row)])
            for row in model_rows
        ]
        if not pairs:
            continue
        alpha = _fit_powerlaw_alpha_for_plot(pairs)
        style = _primitive_style(model)
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        axes[0].scatter(xs, ys, s=18, alpha=0.45, color=style["color"], marker=style["marker"], label=f"{style['label']} data")
        axes[0].plot(curve_x, [_powerlaw_forward_for_plot(x, alpha) for x in curve_x], color=style["color"], linewidth=2, label=f"{style['label']} power alpha={alpha:.2g}")
        calibrated = [_powerlaw_inverse_for_plot(y, alpha) for y in ys]
        axes[1].scatter(xs, calibrated, s=18, alpha=0.45, color=style["color"], marker=style["marker"], label=f"{style['label']} inverse-calibrated")
    for ax in axes:
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="identity")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
    scope = "all binary families" if family is None else f"{family} family"
    axes[0].set_title(f"Returned Probability with Power-Law Fit: {scope}")
    axes[0].set_xlabel("True P(proposition)")
    axes[0].set_ylabel("Returned P(proposition)")
    axes[1].set_title(f"After Inverse Power-Law Calibration: {scope}")
    axes[1].set_xlabel("True P(proposition)")
    axes[1].set_ylabel("Calibrated returned P(proposition)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _fit_powerlaw_alpha_for_plot(pairs):
    best = (float("inf"), 1.0)
    for index in range(2996):
        alpha = 0.05 + index * 0.01
        mse = sum((_powerlaw_forward_for_plot(gold, alpha) - pred) ** 2 for gold, pred in pairs) / len(pairs)
        if mse < best[0]:
            best = (mse, alpha)
    return best[1]


def _powerlaw_forward_for_plot(p, alpha):
    p = min(1.0, max(0.0, float(p)))
    return 1.0 - (1.0 - p) ** alpha


def _powerlaw_inverse_for_plot(q, alpha):
    q = min(1.0, max(0.0, float(q)))
    if alpha <= 0:
        return q
    return 1.0 - (1.0 - q) ** (1.0 / alpha)


def _error_by_family(plt, rows, output: Path):
    families = sorted({row["family"] for row in rows})
    models = _ordered_models(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family"])].append(row["metrics"]["tv"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(1, len(models))
    base = list(range(len(families)))
    for index, model in enumerate(models):
        style = _primitive_style(model)
        offset = (index - (len(models) - 1) / 2) * width
        values = [sum(grouped[(model, family)]) / len(grouped[(model, family)]) if grouped[(model, family)] else 0.0 for family in families]
        ax.bar([x + offset for x in base], values, width=width, color=style["color"], label=style["label"], alpha=0.85)
    ax.set_ylabel("Mean TV error")
    ax.set_title("Error by Family and Jev Primitive")
    ax.set_xticks(base)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Primitive", frameon=False)
    path = output / "figure_2_error_by_family.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _error_by_representation(plt, rows, output: Path):
    representations = sorted({row["representation"] for row in rows})
    models = _ordered_models(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["representation"])].append(row["metrics"]["tv"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(1, len(models))
    base = list(range(len(representations)))
    for index, model in enumerate(models):
        style = _primitive_style(model)
        offset = (index - (len(models) - 1) / 2) * width
        values = [sum(grouped[(model, rep)]) / len(grouped[(model, rep)]) if grouped[(model, rep)] else 0.0 for rep in representations]
        ax.bar([x + offset for x in base], values, width=width, color=style["color"], label=style["label"], alpha=0.85)
    ax.set_ylabel("Mean TV error")
    ax.set_title("Representation Sensitivity by Model-Primitive")
    ax.set_xticks(base)
    ax.set_xticklabels(representations, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.1)
    ax.legend(frameon=True)
    path = output / "figure_3_representation_sensitivity.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _equivalent_spread(plt, rows, output: Path):
    by_latent = defaultdict(list)
    for row in rows:
        if len(row["gold_distribution"]) == 2:
            by_latent[row["latent_instance_id"]].append(row)
    chosen = next((items for items in by_latent.values() if len({r["representation"] for r in items}) > 2 and len({r["model"] for r in items}) > 1), [])
    if not chosen:
        return None
    models = _ordered_models(chosen)
    representations = sorted({row["representation"] for row in chosen})
    by_key = {(row["model"], row["representation"]): row for row in chosen}
    gold = chosen[0]["gold_distribution"][_positive_label(chosen[0])]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(1, len(models))
    base = list(range(len(representations)))
    for index, model in enumerate(models):
        style = _primitive_style(model)
        offset = (index - (len(models) - 1) / 2) * width
        values = []
        for rep in representations:
            row = by_key.get((model, rep))
            values.append(row["predicted_distribution"][_positive_label(row)] if row else 0.0)
        ax.bar([x + offset for x in base], values, width=width, color=style["color"], label=style["label"], alpha=0.85)
    ax.axhline(gold, color="black", linewidth=1.2, label="gold P(proposition)")
    ax.set_ylabel("Returned P(proposition)")
    ax.set_title(f"Equivalent Renderings for {chosen[0]['latent_instance_id']}")
    ax.set_xticks(base)
    ax.set_xticklabels(representations, rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Primitive", frameon=False)
    path = output / "figure_4_equivalent_state_spread.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _sequential_trajectories(plt, rows, output: Path):
    data = [r for r in rows if r["family"] == "sequential_bayes" and len(r["gold_distribution"]) == 2]
    grouped = defaultdict(list)
    for row in data:
        trajectory = _posterior_trajectory(row)
        if trajectory:
            grouped[row["latent_instance_id"]].append(row)
    if not grouped:
        return _sequential_final_scatter(plt, data, output)

    selected = list(grouped.items())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    axes_flat = list(axes.flat)
    color_by_representation = {
        "direct": "#4c78a8",
        "table": "#f58518",
        "prose": "#54a24b",
        "counts": "#e45756",
        "distractor": "#b279a2",
        "nested": "#9d755d",
    }
    primitive_handles = {}
    representation_handles = {}
    gold_handle = None
    for ax, (latent_id, items) in zip(axes_flat, selected):
        trajectory = _posterior_trajectory(items[0]) or []
        x_values = list(range(len(trajectory)))
        gold_handle, = ax.plot(x_values, trajectory, color="black", linewidth=2, marker="o", label="gold posterior")
        final_step = len(trajectory) - 1
        for item in sorted(items, key=lambda row: (row["model"], row["representation"])):
            representation = item["representation"]
            primitive = _primitive_style(item["model"])
            marker = ax.scatter(
                [final_step],
                [item["predicted_distribution"][_positive_label(item)]],
                color=color_by_representation.get(representation, "tab:gray"),
                marker=primitive["marker"],
                edgecolor="black",
                linewidth=0.4,
                s=62,
                zorder=3,
            )
            primitive_handles.setdefault(primitive["label"], marker)
            representation_handles.setdefault(
                representation,
                ax.scatter([], [], color=color_by_representation.get(representation, "tab:gray"), marker="o", s=48),
            )
        ax.set_title(latent_id.replace("sequential_bayes__", ""), fontsize=9)
        ax.set_xlabel("Evidence step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(selected):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("P(proposition)")
    axes_flat[2].set_ylabel("P(proposition)")
    fig.suptitle("Sequential Bayes: Gold Trajectories with Final Predictions by Primitive")
    handles = ([gold_handle] if gold_handle else []) + list(primitive_handles.values()) + list(representation_handles.values())
    labels = (["gold posterior"] if gold_handle else []) + list(primitive_handles.keys()) + list(representation_handles.keys())
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    path = output / "figure_5_sequential_bayes_trajectories.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _posterior_trajectory(row):
    metadata = row.get("generator_metadata") or {}
    trajectory = metadata.get("posterior_trajectory")
    if trajectory is None:
        trajectory = (metadata.get("parameters") or {}).get("posterior_trajectory")
    if not trajectory:
        return None
    return [float(value) for value in trajectory]


def _sequential_final_scatter(plt, data, output: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model in _ordered_models(data):
        model_rows = [r for r in data if r["model"] == model]
        style = _primitive_style(model)
        xs = [r["gold_distribution"][_positive_label(r)] for r in model_rows]
        ys = [r["predicted_distribution"][_positive_label(r)] for r in model_rows]
        ax.scatter(xs, ys, s=24, color=style["color"], marker=style["marker"], label=style["label"])
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="identity")
    ax.set_xlabel("Final gold posterior")
    ax.set_ylabel("Final returned posterior")
    ax.legend(title="Primitive", frameon=False)
    path = output / "figure_5_sequential_bayes_final.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _calibration_curve(plt, rows, calibration, output: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="ideal")
    for model in _ordered_models(rows):
        model_rows = [row for row in rows if row["model"] == model]
        model_calibration = _calibration_bins(model_rows)
        style = _primitive_style(model)
        bins = model_calibration.get("bins", [])
        if not bins:
            continue
        ax.plot(
            [b["mean_predicted"] for b in bins],
            [b["empirical_frequency"] for b in bins],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=f"{style['label']} (ECE {model_calibration['ece']:.3f})",
        )
    ax.set_xlabel("Mean predicted P(proposition)")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Calibration by Jev Primitive")
    ax.grid(True, alpha=0.2)
    ax.legend(title="Primitive", frameon=False)
    path = output / "figure_6_calibration_curve.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _calibration_bins(rows, bins: int = 10) -> dict[str, Any]:
    records = []
    for row in rows:
        if len(row["gold_distribution"]) != 2:
            continue
        positive = _positive_label(row)
        # Deterministic expected calibration against the known Bernoulli gold,
        # not a simulated draw. This makes primitive curves stable across runs.
        records.append((row["predicted_distribution"][positive], row["gold_distribution"][positive]))
    if not records:
        return {"bins": [], "ece": 0.0}
    bucketed = []
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [(q, p) for q, p in records if lo <= q < hi or (i == bins - 1 and q == 1.0)]
        if not bucket:
            continue
        mean_q = sum(q for q, _ in bucket) / len(bucket)
        mean_p = sum(p for _, p in bucket) / len(bucket)
        ece += len(bucket) / len(records) * abs(mean_p - mean_q)
        bucketed.append({"lo": lo, "hi": hi, "count": len(bucket), "mean_predicted": mean_q, "empirical_frequency": mean_p})
    return {"bins": bucketed, "ece": ece}



def _ncs_consistency_scatter(plt, records, output: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    by_family = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
    for family, family_records in sorted(by_family.items()):
        ax.scatter(
            [record["p_noul"] for record in family_records],
            [record["p_choice"] for record in family_records],
            s=20,
            alpha=0.65,
            label=family,
        )
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="Noul = Choice")
    ax.set_xlabel("Noul P(proposition)")
    ax.set_ylabel("Choice P(True)")
    ax.set_title("Noul-Choice Consistency")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    path = output / "figure_7_noul_choice_consistency.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _ncs_score_expectation_scatter(plt, records, output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True, sharey=True)
    comparisons = [
        ("p_star", "Ground truth P(A)"),
        ("p_noul", "Noul P(A)"),
        ("p_choice", "Choice P(A)"),
    ]
    for ax, (key, label) in zip(axes, comparisons):
        ax.scatter([record[key] for record in records], [record["mu_score"] for record in records], s=18, alpha=0.65, color="tab:green")
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(r"Score expectation $\mu_s$")
    fig.tight_layout()
    path = output / "figure_8_score_expectation_connections.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _choice_from_score_recovery_plot(plt, rows, output: Path):
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True)
    panels = [
        ("pred_choice_fuzzy_unknown_conditional", "Fuzzy unknown conditional"),
        ("pred_choice_monotonic_collapse", "Monotonic collapse"),
        ("pred_choice_full_q_ridge", "Full-q ridge"),
    ]
    for ax, (key, title) in zip(axes, panels):
        ax.scatter([row["p_choice"] for row in rows], [row[key] for row in rows], s=24, alpha=0.7, color="tab:orange")
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Observed Choice P(True)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Predicted Choice from Score")
    fig.suptitle("Recovering Choice From Score Distribution")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output / "figure_9_choice_from_score_recovery.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _score_levels_vs_choice_heatmap(plt, records, output: Path):
    if not records:
        return None
    bins = 20
    matrix = [[0.0 for _ in range(bins)] for _ in range(10)]
    counts = [0 for _ in range(bins)]
    for record in records:
        bin_index = min(bins - 1, max(0, int(record["p_choice"] * bins)))
        counts[bin_index] += 1
        for level in range(10):
            matrix[level][bin_index] += record[f"q_{level}"]
    for bin_index, count in enumerate(counts):
        if count <= 0:
            continue
        for level in range(10):
            matrix[level][bin_index] /= count

    fig, ax = plt.subplots(figsize=(10, 5.8))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[0, 1, -0.5, 9.5],
        cmap="viridis",
        vmin=0,
        vmax=max(max(row) for row in matrix) if matrix else 1,
    )
    ax.set_xlabel("Observed Choice P(True)")
    ax.set_ylabel("Score level")
    ax.set_yticks(range(10))
    ax.set_yticklabels([
        "0 false",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9 true",
    ])
    ax.set_title("Score Level Mass Conditional on Choice Probability")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean Score probability mass")

    top_levels = []
    for bin_index, count in enumerate(counts):
        if count <= 0:
            continue
        column = [matrix[level][bin_index] for level in range(10)]
        top_level = max(range(10), key=lambda level: column[level])
        top_levels.append(((bin_index + 0.5) / bins, top_level))
    if top_levels:
        ax.plot([x for x, _ in top_levels], [level for _, level in top_levels], color="white", linewidth=1.4, marker="o", markersize=3, label="modal Score level")
        ax.legend(frameon=False, loc="upper left")

    path = output / "figure_10_score_levels_vs_choice_heatmap.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _score_choice_entropy_plot(plt, rows, output: Path):
    if not rows:
        return None
    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    xs = [row["choice_bins"] for row in rows]
    nmi = [row["normalized_mutual_information"] for row in rows]
    cond = [row["h_score_given_choice_bin"] for row in rows]
    wasserstein = [row["mean_adjacent_wasserstein"] for row in rows]

    line1, = ax1.plot(xs, nmi, color="tab:purple", marker="o", linewidth=2, label="normalized MI")
    line2, = ax1.plot(xs, wasserstein, color="tab:orange", marker="s", linewidth=1.5, label="adjacent W1")
    ax1.set_xlabel("Number of Choice probability bins")
    ax1.set_ylabel("Normalized information / ordered shift")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    line3, = ax2.plot(xs, cond, color="tab:green", marker="^", linewidth=1.5, linestyle="--", label="H(Score | Choice bin)")
    ax2.set_ylabel("Conditional Score entropy (bits)")
    ax2.set_ylim(bottom=0)

    ax1.legend([line1, line2, line3], [line1.get_label(), line2.get_label(), line3.get_label()], frameon=False, loc="best")
    ax1.set_title("Multiscale Entropy: Score-Choice Relationship")
    path = output / "figure_11_score_choice_multiscale_entropy.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _choice_group_summary(records, group_key: str):
    return choice_from_score_group_summary(records, group_key)


def _choice_group_mae_plot(plt, rows, output: Path, group_key: str, filename: str):
    wanted_rules = [
        ("expected_truth", "Expected truth", "#8da0cb"),
        ("crisp_unknown_conditional", "Unknown conditional", "#fc8d62"),
        ("monotonic_collapse", "Monotonic", "#66c2a5"),
        ("full_q_ridge", "Full-q ridge", "#e78ac3"),
    ]
    groups = sorted({row["group"] for row in rows if row.get("rule") != "insufficient_data"})
    if not groups:
        return None
    by_key = {(row["group"], row["rule"]): row for row in rows}
    fig_width = max(9, 1.1 * len(groups) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    width = 0.8 / len(wanted_rules)
    base = list(range(len(groups)))
    for index, (rule, label, color) in enumerate(wanted_rules):
        offset = (index - (len(wanted_rules) - 1) / 2) * width
        values = []
        for group in groups:
            row = by_key.get((group, rule))
            values.append(float(row["mae"]) if row and row.get("mae") not in {None, ""} else 0.0)
        ax.bar([x + offset for x in base], values, width=width, color=color, label=label, alpha=0.9)
    ax.set_ylabel("Held-out MAE to Choice P(True)")
    ax.set_title(f"Score to Choice Recovery by {group_key.title()}")
    ax.set_xticks(base)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    path = output / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _crisp_unknown_threshold_plot(plt, rows, output: Path, filename: str):
    crisp = [row for row in rows if row.get("rule") == "crisp_unknown_conditional" and row.get("false_max_level") is not None]
    if not crisp:
        return None
    groups = [row["group"] for row in crisp]
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(groups) + 3), 4.8))
    y_positions = list(range(len(groups)))
    for y, row in zip(y_positions, crisp):
        false_max = int(row["false_max_level"])
        true_min = int(row["true_min_level"])
        ax.barh(y, false_max + 1, left=0, color="#4c78a8", alpha=0.9, label="false" if y == 0 else None)
        unknown_width = max(0, true_min - false_max - 1)
        if unknown_width:
            ax.barh(y, unknown_width, left=false_max + 1, color="#f58518", alpha=0.9, label="unknown" if y == 0 else None)
        ax.barh(y, 10 - true_min, left=true_min, color="#54a24b", alpha=0.9, label="true" if y == 0 else None)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(groups)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(11))
    ax.set_xlabel("Score level interval")
    ax.set_title("Crisp Unknown Model: False / Unknown / True Bands")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    path = output / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _calibration_by_representation(plt, rows, output: Path):
    binary = [row for row in rows if len(row["gold_distribution"]) == 2]
    representations = sorted({row["representation"] for row in binary})
    if not representations:
        return None
    cols = 3
    rows_count = (len(representations) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 4.7, rows_count * 3.8), sharex=True, sharey=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    legend_handles = {}
    for ax, representation in zip(axes_flat, representations):
        rep_rows = [row for row in binary if row["representation"] == representation]
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.8)
        ece_lines = []
        for model in _ordered_models(rep_rows):
            model_rows = [row for row in rep_rows if row["model"] == model]
            calibration = _calibration_bins(model_rows)
            bins = calibration.get("bins", [])
            if not bins:
                continue
            style = _primitive_style(model)
            line, = ax.plot(
                [b["mean_predicted"] for b in bins],
                [b["empirical_frequency"] for b in bins],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.4,
                markersize=4,
                label=style["label"],
            )
            legend_handles.setdefault(style["label"], line)
            ece_lines.append(f"{style['label'][0]} {calibration['ece']:.3f}")
        ax.set_title(representation)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.03,
            0.97,
            "ECE: " + "  ".join(ece_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    for ax in axes_flat[len(representations):]:
        ax.axis("off")
    for ax in axes_flat[-cols:]:
        ax.set_xlabel("Mean predicted P(proposition)")
    for ax in axes_flat[::cols]:
        ax.set_ylabel("Mean gold P(proposition)")
    fig.suptitle("Calibration by Representation and Jev Primitive")
    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(), loc="lower center", ncol=len(legend_handles), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = output / "figure_15_calibration_by_representation.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path



def _noul_choice_calibration_plot(plt, rows, output: Path):
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family, family_rows in sorted(by_family.items()):
        ax.scatter(
            [row["p_noul"] for row in family_rows],
            [row["p_choice"] for row in family_rows],
            s=22,
            alpha=0.6,
            label=family,
        )
    xs = [index / 200 for index in range(201)]
    curve_specs = [
        ("identity", "identity", "black", ":"),
        ("power", "power", "tab:red", "-"),
        ("logit", "logit", "tab:purple", "--"),
        ("beta", "beta", "tab:green", "-."),
        ("isotonic", "isotonic", "tab:orange", "-"),
    ]
    for key, label, color, linestyle in curve_specs:
        pred_key = f"pred_choice_{key}"
        points = sorted((row["p_noul"], row[pred_key]) for row in rows if row.get(pred_key) is not None)
        if key == "identity":
            ys = xs
        elif not points:
            continue
        else:
            ys = [_interpolate_curve(points, x) for x in xs]
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=2, label=label)
    ax.set_xlabel("Noul P(True)")
    ax.set_ylabel("Choice P(True)")
    ax.set_title("Noul to Choice Calibration Mappings")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    path = output / "figure_16_noul_choice_calibration.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _score_choice_uncertainty_mapping_plot(plt, records, output: Path):
    if not records:
        return None
    params = _fit_score_choice_uncertainty_params(records)
    lam, alpha, beta = params
    predictions = [_score_choice_uncertainty_prediction(record["mu_score"], params) for record in records]
    targets = [record["p_choice"] for record in records]
    mae = sum(abs(prediction - target) for prediction, target in zip(predictions, targets)) / len(records)
    mean_target = sum(targets) / len(targets)
    sse = sum((prediction - target) ** 2 for prediction, target in zip(predictions, targets))
    sst = sum((target - mean_target) ** 2 for target in targets)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.scatter(
        [record["mu_score"] for record in records],
        targets,
        s=42,
        alpha=0.45,
        color="#6f8fb8",
        edgecolors="none",
    )
    xs = [index / 400 for index in range(401)]
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.6, linestyle="--")
    ax.plot(
        xs,
        [_score_choice_uncertainty_prediction(x, params) for x in xs],
        color="#df3f5b",
        linewidth=4,
        label=rf"uncertainty fit ($\lambda={lam:.3f}$)",
    )
    ax.text(0.03, 0.94, rf"MAE={mae:.3f}  $R^2$={r2:.3f}", transform=ax.transAxes, fontsize=16, va="top")
    ax.set_title("Score Expectation to Choice Mapping", fontsize=18, pad=10)
    ax.set_xlabel(r"Score expectation $\mu_S$", fontsize=14)
    ax.set_ylabel("Choice P(True)", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    fig.tight_layout()
    path = output / "figure_16_noul_choice_calibration.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _fit_score_choice_uncertainty_params(records) -> tuple[float, float, float]:
    params = [1.0, 0.5, 1.0]
    steps = [0.1, 0.1, 0.1]
    bounds = [(0.0, 1.0), (0.05, 3.0), (1.0, 4.0)]

    def objective(values: list[float]) -> float:
        errors = [
            (_score_choice_uncertainty_prediction(record["mu_score"], values) - record["p_choice"]) ** 2
            for record in records
        ]
        return sum(errors) / len(errors)

    best = objective(params)
    for _ in range(80):
        improved = False
        for index in range(3):
            for sign in (-1, 1):
                candidate = params[:]
                lo, hi = bounds[index]
                candidate[index] = min(hi, max(lo, candidate[index] + sign * steps[index]))
                value = objective(candidate)
                if value < best:
                    params = candidate
                    best = value
                    improved = True
        if not improved:
            steps = [step * 0.5 for step in steps]
            if max(steps) < 1e-4:
                break
    return (params[0], params[1], params[2])


def _score_choice_uncertainty_prediction(mu: float, params) -> float:
    lam, alpha, beta = params
    if mu <= 0.0 or mu >= 1.0:
        return min(1.0, max(0.0, mu))
    uncertainty = lam * (mu**alpha) * ((1.0 - mu) ** beta)
    return min(1.0, max(0.0, mu / max(1e-12, 1.0 - uncertainty)))


def _interpolate_curve(points, x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            weight = (x - x0) / (x1 - x0)
            return y0 * (1.0 - weight) + y1 * weight
    return points[-1][1]



def _score_unknown_scatter(plt, rows, output: Path):
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    panels = [
        ("score_fuzzy_mass", "Quadratic 4z(1-z)"),
        ("score_entropy_mass", "Normalized binary entropy"),
        ("score_crisp_unknown_2_3_mass", "Score mass levels 2-3"),
    ]
    for ax, (key, title) in zip(axes, panels):
        ax.scatter([row[key] for row in rows], [row["implied_unknown"] for row in rows], s=22, alpha=0.65, color="tab:purple")
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Score-derived unknown")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Implied unknown from Noul/Choice")
    fig.suptitle("Does Score Mass Predict Implied Unknown?")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output / "figure_17_score_unknown_vs_implied_unknown.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _score_unknown_weights_plot(plt, summary_rows, output: Path):
    learned = next((row for row in summary_rows if row.get("measure") == "learned_score_unknown_weights"), None)
    constrained = next((row for row in summary_rows if row.get("measure") == "learned_symmetric_monotone_weights"), None)
    gamma_row = next((row for row in summary_rows if row.get("measure") == "score_gamma_fuzzy_mass"), None)
    if not learned and not constrained:
        return None
    levels = list(range(10))
    z_values = [level / 9 for level in levels]
    fuzzy = [4.0 * z * (1.0 - z) for z in z_values]
    entropy = [_binary_entropy_weight_for_plot(z) for z in z_values]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    if learned:
        weights = [float(learned.get(f"w_{level}", 0.0)) for level in levels]
        ax.plot(levels, weights, marker="o", linewidth=2, label="learned unconstrained")
    if constrained:
        constrained_weights = [float(constrained.get(f"w_{level}", 0.0)) for level in levels]
        ax.plot(levels, constrained_weights, marker="D", linewidth=2, label="learned symmetric monotone")
    ax.plot(levels, fuzzy, marker="s", linestyle="--", linewidth=1.5, label="quadratic 4z(1-z)")
    ax.plot(levels, entropy, marker="^", linestyle=":", linewidth=1.8, label="binary entropy")
    if gamma_row and gamma_row.get("gamma") not in (None, ""):
        gamma = float(gamma_row["gamma"])
        gamma_fuzzy = [value**gamma for value in fuzzy]
        ax.plot(levels, gamma_fuzzy, marker="x", linestyle="-.", linewidth=1.5, label=f"fitted [4z(1-z)]^{gamma:g}")
    ax.set_xlabel("Score level")
    ax.set_ylabel("Unknown weight")
    ax.set_title("Score-Level Weights for Implied Unknown Mass")
    ax.set_xticks(levels)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    path = output / "figure_18_score_unknown_level_weights.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _binary_entropy_weight_for_plot(z: float) -> float:
    if z <= 0.0 or z >= 1.0:
        return 0.0
    return -z * math.log2(z) - (1.0 - z) * math.log2(1.0 - z)


def _score_three_state_scatter(plt, rows, output: Path):
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    panels = [
        ("pred_noul_from_score_t", "p_noul", "Noul: Score T", "Observed Noul"),
        ("pred_choice_from_score_tf", "p_choice", "Choice: T/(T+F)", "Observed Choice"),
        ("pred_unknown_from_score_u", "implied_unknown", "Score U", "Implied Unknown"),
    ]
    for ax, (x_key, y_key, x_label, y_label) in zip(axes, panels):
        valid = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
        ax.scatter([row[x_key] for row in valid], [row[y_key] for row in valid], s=22, alpha=0.65, color="tab:cyan")
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Direct Score to Three-State Decomposition")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output / "figure_19_score_three_state_decomposition.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _score_three_state_memberships_plot(plt, summary_rows, output: Path):
    if not summary_rows:
        return None
    first = summary_rows[0]
    levels = list(range(10))
    u = _parse_weight_profile(first.get("u_profile"))
    t = _parse_weight_profile(first.get("t_profile"))
    f = _parse_weight_profile(first.get("f_profile"))
    if not (u and t and f):
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.plot(levels, f, marker="s", linewidth=2, label="False membership")
    ax.plot(levels, u, marker="o", linewidth=2, label="Unknown membership")
    ax.plot(levels, t, marker="^", linewidth=2, label="True membership")
    ax.set_xlabel("Score level")
    ax.set_ylabel("Membership weight")
    ax.set_title("Learned Score to T/U/F Memberships")
    ax.set_xticks(levels)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    path = output / "figure_20_score_three_state_memberships.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _parse_weight_profile(value):
    if not value:
        return []
    if isinstance(value, str):
        return [float(part) for part in value.split(",") if part]
    return [float(part) for part in value]


def _score_choice_tuf_plot(plt, rows, output: Path):
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    families = sorted({row.get("family", "unknown") for row in rows})
    cmap = plt.get_cmap("tab10")
    colors = {family: cmap(index % 10) for index, family in enumerate(families)}
    for family in families:
        group = [row for row in rows if row.get("family") == family]
        axes[0].scatter([row["pred_choice_tuf"] for row in group], [row["p_choice"] for row in group], s=24, alpha=0.68, color=colors[family], label=family)
        axes[1].scatter([row["u_score"] for row in group], [row["p_choice"] - row["t_score"] for row in group], s=24, alpha=0.68, color=colors[family])
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set_xlabel("Predicted Choice from Score T/U/F")
    axes[0].set_ylabel("Observed Choice")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.2)
    axes[1].set_xlabel("Score-derived U")
    axes[1].set_ylabel("Choice - Score-derived T")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].grid(True, alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Choice Explained Through Score-Derived T/U/F")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output / "figure_21_score_choice_tuf.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

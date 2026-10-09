#!/usr/bin/env python3
"""How well do pre-proposal features predict whether a prefetch edge takes the acceptance branch?

Every scored edge of the prefetching tree is one example (see ``acceptance_features``): the target is its acceptance
probability A, and the inputs are what is known before the proposal is generated -- the cut, the tree position, and
the subtree root's likelihood and entropy around the cut. Three outputs share one prefix:

- ``.acceptance-predictors.pdf``: one panel per feature, mean A and the three classes (A >= 1 - epsilon,
  A <= epsilon, and the uncertain rest; epsilon defaults to 0.01) in quantile bins of that feature.
- ``.acceptance-predictors.features.csv``: per feature, Spearman correlation with A and, in separate columns, its AUC
  as a one-feature classifier of each class against all other edges.
- ``.acceptance-predictors.model.csv`` / ``.coefficients.csv``: three separate ridge logistic classifiers on nested
  feature sets -- ``A >= 1 - epsilon``, ``A <= epsilon``, and ``epsilon < A < 1 - epsilon``, each vs. rest --
  cross-validated with whole questions held out, each against its base-rate baseline, in separate columns.
- ``.acceptance-predictors.selection.csv``: forward feature selection for each classifier -- starting from no feature,
  repeatedly add the feature that most lowers the held-out log loss, until no feature lowers it by ``--min-gain``. The
  order features enter is their usefulness *given the ones already chosen*, so a near-duplicate of a chosen feature
  (``cut_frac`` after ``cut``) enters late or not at all.

Run from the repository root:

    uv run --project case_studies/draw \
        python case_studies/draw/draw_acceptance_predictors.py \
        --log /absolute/path/to/run.subtreePrefetch.vllm.log \
        --epsilon 0.01

Pass several ``--log`` files (with ``--output-prefix``) to pool runs, or ``--edges`` to reuse a CSV written by
``extract/acceptance_features.py``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.common.plot_helpers import average_ranks, rank_correlation
from case_studies.extract.analysis.acceptance_features import (
    CUT_FEATURES,
    DEFAULT_EPSILON,
    ENTROPY_FEATURES,
    FEATURES,
    LIKELIHOOD_FEATURES,
    certainty_classes,
    edge_table,
)
from case_studies.extract.run_naming import config_output_prefix, latex_text, output_path
from case_studies.plot_config import BLUE, DARK_GRAY, ORANGE, RED, apply_plot_style

apply_plot_style()

OUTPUT_TAG = ".acceptance-predictors"
NUM_BINS = 20
# Features with at most this many distinct values are binned by value instead of by quantile.
MAX_DISCRETE_VALUES = 12
NUM_FOLDS = 5
RIDGE = 1.0
NEWTON_STEPS = 50
PANEL_SIZE = (3.2, 2.5)
PANEL_COLUMNS = 4
SEED = 0
# Forward selection stops once the best remaining feature lowers the held-out log loss by less than this.
DEFAULT_MIN_GAIN = 1e-3
# Column prefix -> edge flag each classifier predicts against all other edges.
CLASSIFIERS = {"accept": "certain_accept", "reject": "certain_reject", "uncertain": "uncertain"}


# ---------------------------------------------------------------------------
# Univariate scores
# ---------------------------------------------------------------------------

def class_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC of ``scores`` ranking positives above negatives; ties count one half."""
    num_pos = int(positive.sum())
    num_neg = positive.size - num_pos
    if num_pos == 0 or num_neg == 0:
        return math.nan
    ranks = average_ranks(scores) + 1.0
    return float((ranks[positive].sum() - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg))


def _either_sign(auc: float) -> float:
    """A feature that ranks a class backwards is as informative as one that ranks it forwards."""
    return max(auc, 1.0 - auc) if not math.isnan(auc) else math.nan


def feature_scores(edges: pd.DataFrame, features: list[str]) -> list[dict[str, object]]:
    """Spearman correlation with A, and each feature's AUC as a one-feature classifier of each easy case vs. rest.

    AUC is invariant to monotone transforms, so it is exactly the best one-feature threshold classifier's ranking
    quality; ``*_auc_either_sign`` also credits a feature that ranks the class backwards.
    """
    rows = []
    for feature in features:
        data = edges[[feature, "accept_prob", *CLASSIFIERS.values()]].dropna()
        values = data[feature].to_numpy(dtype=float)
        row = {
            "feature": feature,
            "num_edges": len(data),
            "spearman_with_accept_prob": rank_correlation(values, data["accept_prob"].to_numpy(dtype=float)),
        }
        for prefix, flag in CLASSIFIERS.items():
            auc = class_auc(values, data[flag].to_numpy(dtype=bool))
            row[f"{prefix}_auc"] = auc
            row[f"{prefix}_auc_either_sign"] = _either_sign(auc)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Binned curves
# ---------------------------------------------------------------------------

def binned_curves(values: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    """Mean A, its standard error, and the three class rates in bins of ``values``."""
    distinct = np.unique(values)
    if distinct.size <= MAX_DISCRETE_VALUES:
        bin_ids = np.searchsorted(distinct, values)
        centers = distinct
    else:
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, NUM_BINS + 1)))
        bin_ids = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)
        centers = None
    frame = pd.DataFrame(
        {
            "bin": bin_ids,
            "value": values,
            "accept_prob": data["accept_prob"].to_numpy(dtype=float),
            "certain_accept": data["certain_accept"].to_numpy(dtype=float),
            "certain_reject": data["certain_reject"].to_numpy(dtype=float),
            "uncertain": data["uncertain"].to_numpy(dtype=float),
        }
    )
    grouped = frame.groupby("bin")
    curves = grouped.agg(
        center=("value", "median"),
        count=("value", "size"),
        mean_accept_prob=("accept_prob", "mean"),
        std_accept_prob=("accept_prob", "std"),
        certain_accept_rate=("certain_accept", "mean"),
        certain_reject_rate=("certain_reject", "mean"),
        uncertain_rate=("uncertain", "mean"),
    ).reset_index(drop=True)
    if centers is not None:
        curves["center"] = centers[grouped.size().index.to_numpy()]
    curves["se_accept_prob"] = curves["std_accept_prob"].fillna(0.0) / np.sqrt(curves["count"])
    return curves


def plot_feature_panel(
    ax: Axes, feature: str, edges: pd.DataFrame, score: dict[str, object], epsilon: float
) -> None:
    data = edges[[feature, "accept_prob", *CLASSIFIERS.values()]].dropna()
    curves = binned_curves(data[feature].to_numpy(dtype=float), data)
    ax.errorbar(
        curves["center"], curves["mean_accept_prob"], yerr=curves["se_accept_prob"],
        color=DARK_GRAY, marker="o", ms=2.5, lw=1.0, capsize=1.5, label="mean $A$",
    )
    ax.plot(curves["center"], curves["certain_accept_rate"], color=ORANGE, marker="^", ms=2.5, lw=1.0,
            label=f"$P(A \\geq {1.0 - epsilon:g})$")
    ax.plot(curves["center"], curves["certain_reject_rate"], color=BLUE, marker="v", ms=2.5, lw=1.0,
            label=f"$P(A \\leq {epsilon:g})$")
    ax.plot(curves["center"], curves["uncertain_rate"], color=RED, marker="s", ms=2.5, lw=1.0,
            label=f"$P({epsilon:g} < A < {1.0 - epsilon:g})$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(latex_text(feature))
    ax.set_title(
        f"$\\rho={score['spearman_with_accept_prob']:+.2f}$, AUC acc$={score['accept_auc']:.2f}$, "
        f"rej$={score['reject_auc']:.2f}$, unc$={score['uncertain_auc']:.2f}$",
        fontsize="small",
    )


def draw_binned_figure(edges: pd.DataFrame, scores: list[dict[str, object]], path: Path, epsilon: float) -> Path:
    num_panels = len(scores)
    num_rows = math.ceil(num_panels / PANEL_COLUMNS)
    num_cols = min(PANEL_COLUMNS, num_panels)
    figure, axes = plt.subplots(
        num_rows, num_cols, figsize=(PANEL_SIZE[0] * num_cols, PANEL_SIZE[1] * num_rows), squeeze=False
    )
    for ax, score in zip(axes.flat, scores):
        plot_feature_panel(ax, str(score["feature"]), edges, score, epsilon)
    for ax in list(axes.flat)[num_panels:]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("rate")
    axes[0, 0].legend(fontsize="x-small", loc="best")
    figure.tight_layout()
    return save_figure(figure, path)


# ---------------------------------------------------------------------------
# Logistic model with question-held-out cross-validation
# ---------------------------------------------------------------------------

def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Ridge logistic regression of a 0/1 target ``y`` by Newton's method; column 0 is the unpenalized intercept."""
    weights = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], ridge)
    penalty[0] = 0.0
    for _ in range(NEWTON_STEPS):
        prob = 1.0 / (1.0 + np.exp(-(x @ weights)))
        gradient = x.T @ (prob - y) + penalty * weights
        hessian = (x * (prob * (1.0 - prob))[:, None]).T @ x + np.diag(penalty) + 1e-9 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return weights


def _design(frame: pd.DataFrame, features: list[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (frame[features].to_numpy(dtype=float) - mean) / std
    return np.column_stack([np.ones(len(frame)), standardized])


def log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))


def question_folds(edges: pd.DataFrame, num_folds: int, seed: int) -> np.ndarray:
    """Assign whole (run, question) groups to folds so no question is in both train and test."""
    groups = edges["run"].astype(str) + "#" + edges["question"].astype(str)
    unique = groups.unique()
    order = np.random.default_rng(seed).permutation(unique.size)
    fold_of_group = dict(zip(unique[order], np.arange(unique.size) % num_folds))
    return groups.map(fold_of_group).to_numpy()


def held_out_predictions(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold logistic predictions of ``y`` from ``x``, and the training base rate for the same test rows.

    Standardization is fit on each training fold only. An ``x`` with no columns fits the intercept alone, which
    reproduces the base rate.
    """
    predicted = np.empty(len(y))
    baseline = np.empty(len(y))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        mean, std = x[train].mean(axis=0), x[train].std(axis=0)
        std[std == 0] = 1.0
        design_train = np.column_stack([np.ones(train.sum()), (x[train] - mean) / std])
        design_test = np.column_stack([np.ones(test.sum()), (x[test] - mean) / std])
        weights = fit_logistic(design_train, y[train])
        predicted[test] = 1.0 / (1.0 + np.exp(-(design_test @ weights)))
        baseline[test] = y[train].mean()
    return predicted, baseline


def cross_validate(edges: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    """Fit the three classifiers separately on each feature set; one wide row per set, a column group each.

    Each classifier's held-out AUC, log loss, and Brier score sit beside the same scores of a constant prediction at
    the training base rate, which is what a classifier must beat to carry any information.
    """
    metrics, coefficients = [], []
    for name, features in feature_sets.items():
        data = edges.dropna(subset=features).reset_index(drop=True)
        num_groups = data.groupby(["run", "question"]).ngroups
        if num_groups < 2:
            continue
        folds = question_folds(data, min(NUM_FOLDS, num_groups), SEED)
        x = data[features].to_numpy(dtype=float)
        row: dict[str, object] = {"feature_set": name, "num_edges": len(data), "num_questions": num_groups}
        coefficient_rows = {feature: {"feature_set": name, "feature": feature} for feature in ["intercept", *features]}

        for prefix, flag in CLASSIFIERS.items():
            y = data[flag].to_numpy(dtype=float)
            predicted, baseline = held_out_predictions(x, y, folds)
            row.update(
                {
                    f"{prefix}_positive_rate": float(y.mean()),
                    f"{prefix}_held_out_auc": class_auc(predicted, y.astype(bool)),
                    f"{prefix}_held_out_log_loss": log_loss(y, predicted),
                    f"{prefix}_baseline_log_loss": log_loss(y, baseline),
                    f"{prefix}_held_out_brier": float(np.mean((predicted - y) ** 2)),
                    f"{prefix}_baseline_brier": float(np.mean((baseline - y) ** 2)),
                }
            )

            mean, std = x.mean(axis=0), x.std(axis=0)
            std[std == 0] = 1.0
            weights = fit_logistic(_design(data, features, mean, std), y)
            for feature, weight in zip(["intercept", *features], weights):
                coefficient_rows[feature][f"{prefix}_standardized_coefficient"] = float(weight)

        metrics.append(row)
        coefficients.extend(coefficient_rows.values())
    return metrics, coefficients


def forward_selection(
    edges: pd.DataFrame,
    features: list[str],
    *,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_features: int | None = None,
) -> list[dict[str, object]]:
    """Greedy forward selection per classifier by held-out log loss, on the edges where every feature is defined.

    Step 0 is the base rate. Each later step records the feature added, the held-out log loss and AUC of the model with
    everything chosen so far, and the drop in log loss the feature bought. Selection for a classifier stops when the
    best remaining feature buys less than ``min_gain`` (that feature is recorded with ``selected=False``) or when
    ``max_features`` features are chosen.
    """
    data = edges.dropna(subset=features).reset_index(drop=True)
    num_groups = data.groupby(["run", "question"]).ngroups
    if num_groups < 2:
        return []
    folds = question_folds(data, min(NUM_FOLDS, num_groups), SEED)
    columns = {feature: data[feature].to_numpy(dtype=float) for feature in features}
    limit = len(features) if max_features is None else min(max_features, len(features))

    rows: list[dict[str, object]] = []
    for prefix, flag in CLASSIFIERS.items():
        y = data[flag].to_numpy(dtype=float)
        _, baseline = held_out_predictions(np.empty((len(y), 0)), y, folds)
        current_loss = log_loss(y, baseline)
        rows.append({"classifier": prefix, "step": 0, "feature": "(base rate)", "selected": True,
                     "held_out_log_loss": current_loss, "log_loss_gain": 0.0, "held_out_auc": 0.5,
                     "num_edges": len(data)})
        chosen: list[str] = []
        while len(chosen) < limit:
            best = None
            for candidate in features:
                if candidate in chosen:
                    continue
                x = np.column_stack([columns[feature] for feature in [*chosen, candidate]])
                predicted, _ = held_out_predictions(x, y, folds)
                loss = log_loss(y, predicted)
                if best is None or loss < best[1]:
                    best = (candidate, loss, predicted)
            candidate, loss, predicted = best
            gain = current_loss - loss
            selected = gain >= min_gain
            rows.append({"classifier": prefix, "step": len(chosen) + 1, "feature": candidate, "selected": selected,
                         "held_out_log_loss": loss, "log_loss_gain": gain,
                         "held_out_auc": class_auc(predicted, y.astype(bool)), "num_edges": len(data)})
            if not selected:
                break
            chosen.append(candidate)
            current_loss = loss
    return rows


def nested_feature_sets(available: list[str]) -> dict[str, list[str]]:
    """Cut and tree features alone, then adding likelihood, then adding entropy, as far as the log supports."""
    sets: dict[str, list[str]] = {}
    current: list[str] = []
    for name, group in (("cut", CUT_FEATURES), ("cut+likelihood", LIKELIHOOD_FEATURES),
                        ("cut+likelihood+entropy", ENTROPY_FEATURES)):
        added = [feature for feature in group if feature in available]
        if not added:
            break
        current = current + added
        sets[name] = list(current)
    return sets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score pre-proposal features as predictors of the prefetch acceptance branch."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, nargs="+", help="Sampler logs to parse.")
    source.add_argument("--edges", type=Path, help="CSV written by extract/acceptance_features.py.")
    parser.add_argument(
        "--output-prefix", type=Path, default=None,
        help="Path prefix for the outputs; defaults to the single log's run configuration, beside the log.",
    )
    parser.add_argument(
        "--held-root-only", action="store_true",
        help="Keep only edges whose decision node still holds the root state (no accepts above it).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help="Easy-case threshold: certain accept A >= 1 - epsilon, certain reject A <= epsilon.",
    )
    parser.add_argument(
        "--min-gain", type=float, default=DEFAULT_MIN_GAIN,
        help="Forward selection stops when the best remaining feature lowers held-out log loss by less than this.",
    )
    parser.add_argument(
        "--max-selected-features", type=int, default=None,
        help="Cap on the features forward selection may choose per classifier (default: no cap).",
    )
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf", help="PDF export needs qpdf.")
    parser.add_argument(
        "--no-usetex", action="store_true",
        help="Render figure text with Matplotlib's own mathtext, for machines without a LaTeX install.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.no_usetex:
        plt.rcParams["text.usetex"] = False
    if args.edges is not None:
        edges = pd.read_csv(args.edges)
        default_prefix = args.edges.with_suffix("")
    else:
        for log in args.log:
            if not log.is_file():
                raise SystemExit(f"missing log {log}")
        edges = pd.concat([edge_table(log, epsilon=args.epsilon) for log in args.log], ignore_index=True)
        default_prefix = config_output_prefix(args.log[0]) if len(args.log) == 1 else None
    output_prefix = args.output_prefix or default_prefix
    if output_prefix is None:
        raise SystemExit("--output-prefix is required when pooling several logs")
    # Recomputed here so an --edges CSV written at another epsilon is relabeled consistently.
    edges["certain_accept"], edges["certain_reject"], edges["uncertain"] = certainty_classes(
        edges["accept_prob"], args.epsilon
    )
    if args.held_root_only:
        edges = edges[edges["held_is_root"].astype(bool)]
    if edges.empty:
        raise SystemExit("no scored edges with a cut index found")

    available = [feature for feature in FEATURES if edges[feature].notna().any()]
    if not any(feature in available for feature in LIKELIHOOD_FEATURES):
        print("no root log_p lines found: only cut and tree features are scored (rerun with print_tree=true)")

    scores = feature_scores(edges, available)
    metrics, coefficients = cross_validate(edges, nested_feature_sets(available))
    selection = forward_selection(
        edges, available, min_gain=args.min_gain, max_features=args.max_selected_features
    )

    # Tables first: they do not depend on a LaTeX install, the figure does.
    outputs = []
    for suffix, rows in (
        (".features.csv", scores),
        (".model.csv", metrics),
        (".coefficients.csv", coefficients),
        (".selection.csv", selection),
    ):
        path = output_path(output_prefix, OUTPUT_TAG + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        outputs.append(path)
    class_labels = {
        "accept": f"A >= {1.0 - args.epsilon:g}",
        "reject": f"A <= {args.epsilon:g}",
        "uncertain": f"{args.epsilon:g} < A < {1.0 - args.epsilon:g}",
    }
    table = pd.DataFrame(scores)
    shared = ["feature", "num_edges", "spearman_with_accept_prob"]
    for prefix, label in class_labels.items():
        columns = shared + [f"{prefix}_auc", f"{prefix}_auc_either_sign"]
        print(f"one-feature classifiers for {label} vs. rest, best first:")
        print(
            table[columns].sort_values(f"{prefix}_auc_either_sign", ascending=False)
            .to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
    if metrics:
        model = pd.DataFrame(metrics)
        for prefix, label in class_labels.items():
            columns = ["feature_set", "num_edges"] + [column for column in model.columns if column.startswith(prefix)]
            print(f"logistic classifier for {label} vs. rest (held-out questions):")
            print(model[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if selection:
        chosen = pd.DataFrame(selection)
        for prefix, label in class_labels.items():
            steps = chosen[chosen["classifier"] == prefix].drop(columns="classifier")
            print(f"forward feature selection for {label} vs. rest (held-out log loss; selected=False stops it):")
            print(steps.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    outputs.append(
        draw_binned_figure(
            edges, scores, output_path(output_prefix, f"{OUTPUT_TAG}.{args.figure_format}"), args.epsilon
        )
    )

    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

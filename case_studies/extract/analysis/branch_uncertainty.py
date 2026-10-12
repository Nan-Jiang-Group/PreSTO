#!/usr/bin/env python3
"""Branch-uncertainty metrics over the realized MH walks in a prefetch log.

Each ``sampled path:`` line in the log is one MH call's realized walk: path i with K_i transitions, each scored by an
untruncated acceptance ratio R and resolved into an accept/reject outcome Y. This module turns those walks into the
node-, path-, and run-level metrics the prefetching argument rests on.

Uncertainty is read off the acceptance probability A, never off the realized label Y -- Y is a single Bernoulli draw
from A, so it carries far less information than A does about how predictable a branch was.

The run-length metrics are the ones that bear directly on subtree prefetching: a long predictable run means a narrow,
deep subtree resolves several consecutive transitions, which is exactly what a prefetcher is built to exploit.

Run from the repository root:

    uv run --project src \
        python case_studies/extract/branch_uncertainty.py \
        --log case_studies/logs/math500/<date>/<run>.subtreePrefetch.hf.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from case_studies.extract.logs.proposal_log import ParsedLog, parse_log
from case_studies.extract.run_naming import config_output_prefix

# The threshold the headline numbers use; the others are sensitivity analysis.
MAIN_EPSILON = 0.10
EPSILONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
SURVIVAL_DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 8, 10)


def acceptance_probability_from_log_ratio(log_ratio: np.ndarray) -> np.ndarray:
    """Convert log MH ratios into ``A = min(1, R) = exp(min(0, log R))``.

    Taking the min in log space before exponentiating avoids overflowing on the large positive ratios that a forced
    accept produces.
    """
    log_ratio = np.asarray(log_ratio, dtype=float)
    if np.isnan(log_ratio).any():
        raise ValueError("log acceptance ratios contain NaN values")
    return np.exp(np.minimum(log_ratio, 0.0))


def acceptance_probability_from_ratio(ratio: np.ndarray) -> np.ndarray:
    """Convert raw MH ratios into ``A = min(1, R)``."""
    ratio = np.asarray(ratio, dtype=float)
    if np.isnan(ratio).any():
        raise ValueError("acceptance ratios contain NaN values")
    if (ratio < 0).any():
        raise ValueError("MH acceptance ratios must be nonnegative")
    return np.minimum(ratio, 1.0)


def normalized_binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in bits, so 0 at A in {0, 1} and 1 at A = 0.5.

    Uses the convention ``0 log 0 = 0``, which also keeps the endpoints out of ``log`` entirely rather than relying on a
    nan-to-zero cleanup.
    """
    p = np.asarray(p, dtype=float)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    result = np.zeros_like(p)
    interior = (p > 0.0) & (p < 1.0)
    q = p[interior]
    result[interior] = (-q * np.log(q) - (1.0 - q) * np.log(1.0 - q)) / np.log(2.0)
    return result


def consecutive_true_run_lengths(mask: Iterable[bool]) -> list[int]:
    """Lengths of the maximal runs of consecutive True values."""
    values = np.asarray(list(mask), dtype=bool)
    if values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False]))
    starts = np.flatnonzero((~padded[:-1]) & padded[1:])
    ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
    return (ends - starts).astype(int).tolist()


def path_transitions(parsed: ParsedLog) -> pd.DataFrame:
    """One row per realized MH transition, in walk order.

    A walk's terminal leaf decides nothing, so only ``visited_internal_ids`` contribute. Each of those is looked up
    among the scored nodes to recover the ratio the decision was made against; a visited node the log never scored is
    dropped and counted, since a transition with no ratio has no A to measure.
    """
    ratios = {(n.tree_index, n.node_id): n.log_ratio for n in parsed.nodes}
    rows: list[dict[str, object]] = []
    unscored = 0
    for walk in parsed.paths:
        step = 0
        for node_id, decision in zip(walk.visited_internal_ids, walk.decisions):
            log_ratio = ratios.get((walk.tree_index, node_id))
            if log_ratio is None:
                unscored += 1
                continue
            rows.append(
                {
                    "path_id": walk.tree_index,
                    "block_index": walk.block_index,
                    "step": step,
                    "node_id": node_id,
                    "log_accept_ratio": log_ratio,
                    "accepted": int(decision == "accept"),
                }
            )
            step += 1
    frame = pd.DataFrame(
        rows,
        columns=[
            "path_id",
            "block_index",
            "step",
            "node_id",
            "log_accept_ratio",
            "accepted",
        ],
    )
    frame.attrs["unscored_transitions"] = unscored
    return frame


# Emitted per MH step by backends/hf/samplers/power_samp_MH.py under verbose.
BASELINE_STEP_PATTERN = re.compile(
    r"MHStep\(block=(?P<block>\d+), step=(?P<step>\d+), "
    r"log_acc_ratio=(?P<ratio>-?[\d.]+|-?inf|nan), "
    r"accepted=(?P<accepted>[01]), cut_idx=(?P<cut>-?\d+), seq_len=(?P<seq>\d+)\)"
)
# The runner opens each question with this, which is what separates one chain
# from the next; block and step counters restart within a sample.
BASELINE_SAMPLE_PATTERN = re.compile(r"^IDX (?P<idx>\d+)\s")


def baseline_transitions(path: Path) -> pd.DataFrame:
    """Realized MH transitions from a baseline power-MH log.

    The baseline proposes once per step and accepts or rejects it, so there is no tree and every scored proposal is also
    a realized transition -- the walk and tree populations that differ for a prefetching run coincide here. One sample's
    whole chain is one path; ``step`` runs across blocks so a run of consecutive predictable decisions can span a block
    boundary, which is what the run-length metrics are about.

    Non-finite ratios are dropped rather than clamped: they mean the proposal scored outside floating-point range, and A
    is not recoverable from them.
    """
    rows: list[dict[str, object]] = []
    sample = -1
    step_in_sample = 0
    dropped = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            opened = BASELINE_SAMPLE_PATTERN.match(line)
            if opened:
                sample = int(opened.group("idx"))
                step_in_sample = 0
                continue
            match = BASELINE_STEP_PATTERN.search(line)
            if not match or sample < 0:
                continue
            ratio = float(match.group("ratio"))
            if not np.isfinite(ratio):
                dropped += 1
                continue
            rows.append(
                {
                    "path_id": sample,
                    "block_index": int(match.group("block")),
                    "step": step_in_sample,
                    "node_id": -1,  # no tree: kept so the columns line up
                    "log_accept_ratio": ratio,
                    "accepted": int(match.group("accepted")),
                }
            )
            step_in_sample += 1
    frame = pd.DataFrame(
        rows,
        columns=[
            "path_id",
            "block_index",
            "step",
            "node_id",
            "log_accept_ratio",
            "accepted",
        ],
    )
    frame.attrs["unscored_transitions"] = dropped
    return frame


def compute_branch_uncertainty_metrics(
    df: pd.DataFrame,
    *,
    path_col: str = "path_id",
    step_col: str = "step",
    ratio_col: str = "log_accept_ratio",
    ratio_kind: str = "log_ratio",
    accepted_col: str | None = "accepted",
    epsilons: tuple[float, ...] = EPSILONS,
) -> dict[str, object]:
    """Node-level, path-level, and run-length branch-uncertainty metrics.

    ``ratio_kind`` selects how ``ratio_col`` is read: ``"log_ratio"`` for log R, ``"ratio"`` for raw R,
    ``"probability"`` for A itself.
    """
    required = {path_col, step_col, ratio_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = df.copy().sort_values([path_col, step_col]).reset_index(drop=True)
    values = data[ratio_col].to_numpy(dtype=float)

    if ratio_kind == "log_ratio":
        acceptance = acceptance_probability_from_log_ratio(values)
    elif ratio_kind == "ratio":
        acceptance = acceptance_probability_from_ratio(values)
    elif ratio_kind == "probability":
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Acceptance probabilities must be in [0, 1]")
        acceptance = values
    else:
        raise ValueError("ratio_kind must be 'log_ratio', 'ratio', or 'probability'")

    data["acceptance_probability"] = acceptance
    data["branch_entropy"] = normalized_binary_entropy(acceptance)
    data["dominant_branch_probability"] = np.maximum(acceptance, 1.0 - acceptance)
    data["minority_branch_probability"] = np.minimum(acceptance, 1.0 - acceptance)

    mean_acceptance = float(data["acceptance_probability"].mean())
    entropy_of_mean = float(normalized_binary_entropy(np.array([mean_acceptance]))[0])
    mean_entropy = float(data["branch_entropy"].mean())

    global_summary = {
        "num_paths": int(data[path_col].nunique()),
        "num_transitions": int(len(data)),
        "mean_acceptance_probability": mean_acceptance,
        "mean_branch_entropy": mean_entropy,
        "mean_dominant_branch_probability": float(
            data["dominant_branch_probability"].mean()
        ),
        "mean_minority_branch_probability": float(
            data["minority_branch_probability"].mean()
        ),
        # Concavity of h makes this non-negative; a large value means the nodes are individually far more decided than
        # the pooled accept rate implies.
        "polarization_gap": entropy_of_mean - mean_entropy,
        "entropy_of_mean_acceptance": entropy_of_mean,
    }

    threshold_results: dict[float, dict[str, object]] = {}
    for epsilon in epsilons:
        if not 0.0 < epsilon < 0.5:
            raise ValueError("Each epsilon must be in (0, 0.5)")

        uncertain_col = f"uncertain_eps_{epsilon:g}"
        predictable_col = f"predictable_eps_{epsilon:g}"
        data[uncertain_col] = (data["acceptance_probability"] > epsilon) & (
            data["acceptance_probability"] < 1.0 - epsilon
        )
        data[predictable_col] = ~data[uncertain_col]

        per_path = (
            data.groupby(path_col, sort=False)
            .agg(
                path_length=(step_col, "size"),
                uncertain_count=(uncertain_col, "sum"),
                uncertain_fraction=(uncertain_col, "mean"),
                mean_entropy=("branch_entropy", "mean"),
                mean_dominant_probability=("dominant_branch_probability", "mean"),
            )
            .reset_index()
        )

        predictable_runs: list[int] = []
        longest_run_per_path: list[int] = []
        for _, path_data in data.groupby(path_col, sort=False):
            runs = consecutive_true_run_lengths(path_data[predictable_col].to_numpy())
            predictable_runs.extend(runs)
            longest_run_per_path.append(max(runs, default=0))
        run_array = np.asarray(predictable_runs, dtype=float)

        run_survival = {
            depth: (float(np.mean(run_array >= depth)) if run_array.size else 0.0)
            for depth in SURVIVAL_DEPTHS
        }
        per_path["longest_predictable_run"] = longest_run_per_path

        threshold_results[epsilon] = {
            "global_uncertain_fraction": float(data[uncertain_col].mean()),
            "global_predictable_fraction": float(data[predictable_col].mean()),
            "median_path_uncertain_fraction": float(
                per_path["uncertain_fraction"].median()
            ),
            "mean_path_uncertain_fraction": float(
                per_path["uncertain_fraction"].mean()
            ),
            "path_uncertain_fraction_q25": float(
                per_path["uncertain_fraction"].quantile(0.25)
            ),
            "path_uncertain_fraction_q75": float(
                per_path["uncertain_fraction"].quantile(0.75)
            ),
            "path_uncertain_fraction_q90": float(
                per_path["uncertain_fraction"].quantile(0.90)
            ),
            "fraction_paths_with_zero_uncertain_nodes": float(
                np.mean(per_path["uncertain_count"] == 0)
            ),
            "mean_predictable_run_length": (
                float(run_array.mean()) if run_array.size else 0.0
            ),
            "median_predictable_run_length": (
                float(np.median(run_array)) if run_array.size else 0.0
            ),
            "mean_longest_run_per_path": float(np.mean(longest_run_per_path)),
            "median_longest_run_per_path": float(np.median(longest_run_per_path)),
            "num_predictable_runs": int(run_array.size),
            "predictable_run_survival": run_survival,
            "per_path": per_path,
        }

    outcome_summary = None
    if accepted_col is not None and accepted_col in data.columns:
        accepted = data[accepted_col].to_numpy(dtype=int)
        if not np.isin(accepted, [0, 1]).all():
            raise ValueError("accepted column must contain only 0 and 1")

        data["dominant_outcome"] = (data["acceptance_probability"] >= 0.5).astype(int)
        data["followed_dominant_branch"] = (
            accepted == data["dominant_outcome"].to_numpy()
        )

        dominant_runs: list[int] = []
        longest_dominant_run_per_path: list[int] = []
        for _, path_data in data.groupby(path_col, sort=False):
            runs = consecutive_true_run_lengths(
                path_data["followed_dominant_branch"].to_numpy()
            )
            dominant_runs.extend(runs)
            longest_dominant_run_per_path.append(max(runs, default=0))
        dominant_run_array = np.asarray(dominant_runs, dtype=float)

        outcome_summary = {
            "realized_dominant_branch_accuracy": float(
                data["followed_dominant_branch"].mean()
            ),
            # The theoretical value the realized accuracy should sit near.
            "expected_dominant_branch_accuracy": float(
                data["dominant_branch_probability"].mean()
            ),
            "mean_dominant_outcome_run_length": (
                float(dominant_run_array.mean()) if dominant_run_array.size else 0.0
            ),
            "median_dominant_outcome_run_length": (
                float(np.median(dominant_run_array)) if dominant_run_array.size else 0.0
            ),
            "median_longest_dominant_run_per_path": float(
                np.median(longest_dominant_run_per_path)
            ),
            "dominant_outcome_run_survival": {
                depth: (
                    float(np.mean(dominant_run_array >= depth))
                    if dominant_run_array.size
                    else 0.0
                )
                for depth in SURVIVAL_DEPTHS
            },
        }

    return {
        "node_metrics": data,
        "global_summary": global_summary,
        "threshold_results": threshold_results,
        "outcome_summary": outcome_summary,
    }


def print_branch_uncertainty(results: dict[str, object]) -> None:
    """Print the numbers section 8 asks a paper to report."""
    summary = results["global_summary"]
    thresholds = results["threshold_results"]
    outcome = results["outcome_summary"]

    print(
        f"\nbranch uncertainty over {summary['num_transitions']} realized "
        f"transitions on {summary['num_paths']} walks"
    )
    print(
        f"  mean acceptance A:            {summary['mean_acceptance_probability']:.3f}"
    )
    print(
        f"  normalized Bernoulli branch entropy: "
        f"{summary['mean_branch_entropy']:.3f} bits"
    )
    print(
        f"  mean dominant probability c:  "
        f"{summary['mean_dominant_branch_probability']:.3f}"
    )
    print(
        f"  polarization gap:             {summary['polarization_gap']:+.3f} "
        f"(h of mean A = {summary['entropy_of_mean_acceptance']:.3f})"
    )

    print("\n  eps   uncertain  median path  zero-unc.  median  mean   Pr(R>=4)")
    print("        (global)   fraction     paths      run     run")
    for epsilon, threshold in thresholds.items():
        marker = "*" if epsilon == MAIN_EPSILON else " "
        print(
            f"  {epsilon:<4g}{marker} {threshold['global_uncertain_fraction']:8.3f}   "
            f"{threshold['median_path_uncertain_fraction']:8.3f}     "
            f"{threshold['fraction_paths_with_zero_uncertain_nodes']:6.3f}   "
            f"{threshold['median_predictable_run_length']:5.1f}  "
            f"{threshold['mean_predictable_run_length']:5.2f}  "
            f"{threshold['predictable_run_survival'][4]:8.3f}"
        )
    print("  (* = main threshold; the rest are sensitivity analysis)")

    main = thresholds.get(MAIN_EPSILON)
    if main is not None:
        survival = main["predictable_run_survival"]
        chain = ", ".join(f"P(R>={d}) {survival[d]:.3f}" for d in SURVIVAL_DEPTHS)
        print(f"  predictable-run survival at eps={MAIN_EPSILON:g}: {chain}")
        print(
            f"    {main['num_predictable_runs']} runs; longest per path: median "
            f"{main['median_longest_run_per_path']:.1f}, mean "
            f"{main['mean_longest_run_per_path']:.2f}"
        )
        print(
            f"    path uncertain fraction IQR "
            f"[{main['path_uncertain_fraction_q25']:.3f}, "
            f"{main['path_uncertain_fraction_q75']:.3f}], "
            f"q90 {main['path_uncertain_fraction_q90']:.3f}"
        )

    if outcome is not None:
        print(
            f"  dominant-branch accuracy:     "
            f"{outcome['realized_dominant_branch_accuracy']:.3f} realized vs "
            f"{outcome['expected_dominant_branch_accuracy']:.3f} expected"
        )
        print(
            f"    dominant-outcome runs: median "
            f"{outcome['median_dominant_outcome_run_length']:.1f}, mean "
            f"{outcome['mean_dominant_outcome_run_length']:.2f}, "
            f"P(R>=4) {outcome['dominant_outcome_run_survival'][4]:.3f}"
        )


def write_metric_csvs(results: dict[str, object], prefix: Path) -> list[Path]:
    """Write the transition-level rows and the per-path rows beside the log."""
    written: list[Path] = []

    node_path = Path(f"{prefix}.transition-metrics.csv")
    results["node_metrics"].to_csv(node_path, index=False)
    written.append(node_path)

    # One per-path table, with a column per threshold rather than a table each.
    merged = None
    for epsilon, threshold in results["threshold_results"].items():
        columns = threshold["per_path"][
            ["path_id", "path_length", "uncertain_fraction", "longest_predictable_run"]
        ].rename(
            columns={
                "uncertain_fraction": f"uncertain_fraction_eps_{epsilon:g}",
                "longest_predictable_run": f"longest_predictable_run_eps_{epsilon:g}",
            }
        )
        merged = (
            columns
            if merged is None
            else merged.merge(columns, on=["path_id", "path_length"])
        )
    if merged is not None:
        path_path = Path(f"{prefix}.path-metrics.csv")
        merged.to_csv(path_path, index=False)
        written.append(path_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Branch-uncertainty metrics over the realized MH walks."
    )
    parser.add_argument(
        "--log", type=Path, required=True, help="Sampler log to parse."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Read a baseline power-MH log (MHStep(...) lines) instead of a "
            "subtree-prefetch log's proposal trees."
        ),
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Print the summary without writing CSVs."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.log.is_file():
        raise SystemExit(f"missing log {args.log}")

    if args.baseline:
        frame = baseline_transitions(args.log)
    else:
        frame = path_transitions(parse_log(args.log))
    if frame.empty:
        raise SystemExit(f"no realized transitions found in {args.log}")
    unscored = frame.attrs.get("unscored_transitions", 0)
    if unscored:
        reason = (
            "step(s) had a non-finite ratio"
            if args.baseline
            else "walked node(s) had no scored ratio"
        )
        print(f"  note: {unscored} {reason} and were dropped")

    results = compute_branch_uncertainty_metrics(frame)
    print_branch_uncertainty(results)

    if not args.no_csv:
        for written in write_metric_csvs(results, config_output_prefix(args.log)):
            print(f"wrote {written}")


if __name__ == "__main__":
    main()

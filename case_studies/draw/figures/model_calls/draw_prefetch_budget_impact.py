#!/usr/bin/env python3
"""Plot the teaser's efficiency metrics against the prefetch budget N.

Run with:

    PATH=/Library/TeX/texbin:/opt/homebrew/bin:$PATH MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_prefetch_budget_impact.py

The teaser figure (``exps/model-call-time-vs-mh-transitions.pdf``) places each sampler at one point: model-call
time per call against MH transitions per model call. This figure spreads the PreSTO points over every recorded
prefetch budget N, for both PowerMH (uniform cut, ``logs/``) and EntropyCut (``logs-entropycut/``):

    (a) MH transitions per model call,  (b) model-call time per call,  (c) MH transitions per second
        relative to the sequential sampler.

Each complete ``--rank`` PreSTO log of ``--model`` is paired with the sequential (``powerMH``) log of identical
settings in the same directory; a budget recorded on several dates uses its latest run. The sequential samplers
are drawn as dashed horizontal lines (interleaved dashes where they coincide), pooled over the baselines of the
plotted runs. One trace is one benchmark sample. Each marker is the pooled ratio over a run's traces, and each error
bar or band is +/-1 SD of the per-trace ratios:

    transitions per call = sum MH transitions / sum model calls,
    time per call        = sum call seconds / sum model calls,
    speedup              = (sum MH transitions / sum call seconds) / the same ratio for the sequential baseline.

At a fixed K, the pooled speedup equals the ratio of total model-call times. ``--aggregate mean`` plots the mean of
the per-trace ratios instead.

A sibling ``.csv`` records every plotted value.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from case_studies import paths
from case_studies.extract import run_naming

run_naming.EXCLUDED_RANKS = frozenset()

from case_studies.draw.common.figure_io import save_figure, write_dict_rows  # noqa: E402
from case_studies.draw.figures.model_calls import (
    draw_accept_first_power_mh_model_calls_and_empirical_time as combined,  # noqa: E402
)
from case_studies.extract.logs.model_call_traces import NoModelCallTracesError, extract_model_call_traces  # noqa: E402
from case_studies.plot_config import BLUE, DARK_GRAY, GRAY, GREEN  # noqa: E402

LOG_ROOT = paths.CASE_STUDIES_DIR
DEFAULT_OUTPUT = LOG_ROOT / "logs" / "lcb_v6" / "prefetch-budget-impact.model-qwen3.5-9b.rank-bfs_accept_first.pdf"

FONT_SIZE = 10.0
PANEL_WIDTH = 2.75
FIGURE_HEIGHT = 2.75
MARKER_SIZE = 5.0
LEGEND_HEIGHT = 0.36
LINE_WIDTH = 1.4
CAP_SIZE = 2.0
BAND_ALPHA = 0.12
# Half the spacing between families, so their error bars at one budget do not overlap.
X_OFFSET = 0.25


@dataclass(frozen=True)
class Family:
    """One sequential sampler and its PreSTO counterpart."""

    baseline_label: str
    presto_label: str
    root: Path
    entropy_cut: bool
    color: str
    marker: str
    offset: float
    # Dash phase: the two baselines coincide at 1.0 in (a) and (c), so their dashes interleave.
    dashes: tuple[float, tuple[float, float]]


FAMILIES = (
    Family("PowerMH", "PreSTO-PowerMH", LOG_ROOT / "logs" / "lcb_v6", False, BLUE, "o", -X_OFFSET, (0.0, (3.0, 3.0))),
    Family("EntropyCut", "PreSTO-EntropyCut", LOG_ROOT / "logs-entropycut" / "lcb_v6", True, GREEN, "D", X_OFFSET, (3.0, (3.0, 3.0))),
)


@dataclass(frozen=True)
class TraceMetrics:
    """Per-trace totals of one run."""

    transitions: np.ndarray
    calls: np.ndarray
    seconds: np.ndarray


def trace_metrics(log_path: Path, complete_only: bool = False) -> TraceMetrics | None:
    """Return per-trace metrics, or None when any trace is missing or incomplete.

    With ``complete_only``, unfinished traces are dropped instead of rejecting the whole log.
    """
    try:
        traces = extract_model_call_traces(log_path)
    except NoModelCallTracesError:
        return None
    if complete_only:
        traces = [trace for trace in traces if trace.complete and trace.calls]
    if not traces or not all(trace.complete and trace.calls for trace in traces):
        return None
    return TraceMetrics(
        transitions=np.array([trace.calls[-1].total_mh_steps for trace in traces], dtype=float),
        calls=np.array([len(trace.calls) for trace in traces], dtype=float),
        seconds=np.array([sum(call.duration_seconds for call in trace.calls) for trace in traces]),
    )


@dataclass(frozen=True)
class BudgetRun:
    """The PreSTO run plotted for one budget, with its same-directory sequential baseline."""

    budget: int
    log_path: Path
    baseline_log: Path
    metrics: TraceMetrics


def latest_runs(family: Family, model: str, steps: int, rank: str) -> list[BudgetRun]:
    """Pick the latest complete PreSTO run per budget that has a matching baseline."""
    chosen: dict[int, BudgetRun] = {}
    for log_path in sorted(family.root.glob(f"*/vllm/*.model-{model}.alpha*.steps{steps}.*.subtreePrefetch.vllm.log")):
        if (".cut-entropy" in log_path.name) != family.entropy_cut:
            continue
        if run_naming.canonical_rank(run_naming.rank_of(log_path)) != rank:
            continue
        metrics = trace_metrics(log_path)
        if metrics is None:
            print(f"skip incomplete {log_path}")
            continue
        try:
            baseline_log = combined.find_power_mh_log(log_path.parent, log_path, None)
        except ValueError:
            print(f"skip without baseline {log_path}")
            continue
        budget = run_naming.prefetch_budget_of(log_path)
        # Sorted paths put later dates last, so the latest run of a budget wins.
        chosen[budget] = BudgetRun(budget, log_path, baseline_log, metrics)
    return [chosen[budget] for budget in sorted(chosen)]


def pooled_baseline(runs: Sequence[BudgetRun]) -> TraceMetrics:
    """Pool the finished traces of the distinct baselines used by the plotted runs."""
    metrics = [trace_metrics(path, complete_only=True) for path in sorted({run.baseline_log for run in runs})]
    if any(item is None for item in metrics):
        raise SystemExit("A sequential baseline log has no finished trace")
    return TraceMetrics(*(np.concatenate([getattr(item, name) for item in metrics])
                          for name in ("transitions", "calls", "seconds")))


def pooled_rate(metrics: TraceMetrics) -> float:
    return float(metrics.transitions.sum() / metrics.seconds.sum())


def panel_values(metrics: TraceMetrics, baseline_rate: float, aggregate: str) -> dict[str, tuple[float, float, int]]:
    """(centre, per-trace SD, trace count) of each panel; the centre is pooled or the per-trace mean."""
    per_trace = {
        "a": metrics.transitions / metrics.calls,
        "b": metrics.seconds / metrics.calls,
        "c": metrics.transitions / metrics.seconds / baseline_rate,
    }
    pooled = {
        "a": metrics.transitions.sum() / metrics.calls.sum(),
        "b": metrics.seconds.sum() / metrics.calls.sum(),
        "c": pooled_rate(metrics) / baseline_rate,
    }
    return {
        panel: (float(pooled[panel] if aggregate == "pooled" else np.mean(values)),
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values))
        for panel, values in per_trace.items()
    }


PANELS = (
    ("a", "transitions / call"),
    ("b", "sec / call"),
    ("c", r"speedup ($\times$)"),
)


def plot(model: str, steps: int, rank: str, output: Path, aggregate: str = "pooled") -> tuple[Path, list[dict]]:
    figure, axes = plt.subplots(1, 3, figsize=(PANEL_WIDTH * 3, FIGURE_HEIGHT))
    rows: list[dict] = []
    all_budgets: set[int] = set()
    handles: list[Line2D] = []
    for family in FAMILIES:
        runs = latest_runs(family, model, steps, rank)
        if not runs:
            print(f"no {family.presto_label} runs under {family.root}")
            continue
        baseline = pooled_baseline(runs)
        baseline_rate = (pooled_rate(baseline) if aggregate == "pooled"
                         else float(np.mean(baseline.transitions / baseline.seconds)))
        budgets = [run.budget for run in runs]
        all_budgets.update(budgets)
        baseline_values = panel_values(baseline, baseline_rate, aggregate)
        run_values = [panel_values(run.metrics, baseline_rate, aggregate) for run in runs]
        for (panel, label), ax in zip(PANELS, axes):
            base_mean, base_sd, base_count = baseline_values[panel]
            ax.axhline(base_mean, color=family.color, linestyle=family.dashes, linewidth=LINE_WIDTH, zorder=2)
            ax.axhspan(base_mean - base_sd, base_mean + base_sd, color=family.color, alpha=BAND_ALPHA,
                       linewidth=0, zorder=1)
            rows.append({"panel": panel, "metric": label, "method": family.baseline_label, "prefetch_budget": 1,
                         "trace_count": base_count, "aggregate": aggregate, "value": base_mean, "sd": base_sd,
                         "log": ";".join(sorted({str(run.baseline_log) for run in runs}))})
            stats = [values[panel] for values in run_values]
            ax.errorbar(
                [budget + family.offset for budget in budgets], [m for m, _, _ in stats], yerr=[s for _, s, _ in stats],
                color=family.color, marker=family.marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                capsize=CAP_SIZE, elinewidth=0.9, markeredgecolor=family.color, zorder=3,
            )
            rows += [{"panel": panel, "metric": label, "method": family.presto_label, "prefetch_budget": run.budget,
                      "trace_count": count, "aggregate": aggregate, "value": m, "sd": sd, "log": str(run.log_path)}
                     for run, (m, sd, count) in zip(runs, stats)]
        handles += [
            Line2D([], [], color=family.color, linestyle=(0.0, (3.0, 1.5)), linewidth=LINE_WIDTH,
                   label=family.baseline_label),
            Line2D([], [], color=family.color, marker=family.marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                   label=family.presto_label),
        ]

    ticks = sorted(all_budgets)
    for (panel, label), ax in zip(PANELS, axes):
        ax.set_xticks(ticks)
        ax.set_xlim(min(ticks) - 1, max(ticks) + 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"prefetch budget $N_{\mathrm{pf}}$", fontsize=FONT_SIZE, labelpad=1)
        ax.set_ylabel(label, fontsize=FONT_SIZE)
        ax.minorticks_off()
        ax.tick_params(axis="both", which="both", top=False, right=False, labelsize=FONT_SIZE - 1)
        ax.text(-0.02, 1.02, rf"\textbf{{({panel})}}", transform=ax.transAxes, fontsize=FONT_SIZE,
                ha="right", va="bottom")
    # Reserve a fixed height in inches for the legend, so the layout holds at any --figsize.
    figure.tight_layout(pad=0.2, w_pad=1.0, rect=(0.0, 0.0, 1.0, 1.0 - LEGEND_HEIGHT / figure.get_figheight()))
    # Order the legend by column: sequential samplers first, then their PreSTO versions.
    figure.legend(handles=handles[0::2] + handles[1::2], loc="upper center", ncol=4, fontsize=FONT_SIZE - 1,
                  frameon=True, edgecolor=GRAY, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4)
    return save_figure(figure, output, pad_inches=0.01), rows


def main(argv: Sequence[str] | None = None) -> None:
    global PANEL_WIDTH, FIGURE_HEIGHT, FONT_SIZE, MARKER_SIZE, LEGEND_HEIGHT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3.5-9b", help="Model tag in the log filenames.")
    parser.add_argument("--steps", type=int, default=100, help="MH steps of the runs to plot.")
    parser.add_argument("--rank", default="bfs_accept_first", help="Traversal rank policy to plot.")
    parser.add_argument("--aggregate", choices=("pooled", "mean"), default="pooled",
                        help="Marker value: pooled ratio over traces (default) or mean of per-trace ratios.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF.")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                        help=f"Figure size in inches (default {PANEL_WIDTH * 3:g} {FIGURE_HEIGHT:g}).")
    parser.add_argument("--font-size", type=float, default=None, help=f"Font size (default {FONT_SIZE:g}).")
    args = parser.parse_args(argv)
    if args.figsize is not None:
        PANEL_WIDTH, FIGURE_HEIGHT = args.figsize[0] / 3, args.figsize[1]
    if args.font_size is not None:
        scale = args.font_size / FONT_SIZE
        FONT_SIZE, MARKER_SIZE, LEGEND_HEIGHT = args.font_size, MARKER_SIZE * scale, LEGEND_HEIGHT * scale
    written, rows = plot(args.model, args.steps, args.rank, args.output.resolve(), args.aggregate)
    write_dict_rows(written.with_suffix(".csv"), rows, fieldnames=list(rows[0]))
    for row in rows:
        if row["panel"] == "a":
            print(f"{row['method']:>18} N={row['prefetch_budget']:>2}: {row['log']}")
    print(f"wrote {written}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Draw a sequential sampler versus its PreSTO version across every recorded prefetch budget.

Run with:

    MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_entropycut_prefetch_budget_sweep.py \
    --root /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-entropycut/lcb_v6 \
    --model qwen3.5-9b

Pass ``--family uniform`` (with ``--root .../case_studies/logs/lcb_v6``) to draw PowerMH versus PreSTO-PowerMH
instead of the default EntropyCut versus PreSTO-EntropyCut; only logs of the chosen cut family and ``--steps``
MH steps are used.

The script scans every ``<root>/<date>/vllm`` directory for complete ``--rank`` (default ``accept_first``)
PreSTO-EntropyCut logs of one model, and pairs each with the EntropyCut (``powerMH``) log of identical settings in
the same directory. Panels (a)-(c) overlay one mean trajectory per prefetch budget against the pooled EntropyCut
traces; a budget recorded on several dates uses its latest run there. One legend entry covers every budget, its
handle showing the budget colours from light (smallest N) to dark (largest). A sibling ``.summary.csv`` records
every plotted value.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from case_studies import paths, plot_config
from case_studies.extract import run_naming

plot_config.POWER_METHOD_LABEL = "EntropyCut"
plot_config.SUBTREE_METHOD_LABEL = "PreSTO-EntropyCut"
run_naming.EXCLUDED_RANKS = frozenset()

from case_studies.draw.common.figure_io import save_figure, write_dict_rows  # noqa: E402
from case_studies.draw.figures.model_calls import (
    draw_accept_first_power_mh_model_calls_and_empirical_time as combined,  # noqa: E402
)
from case_studies.extract.logs.model_call_traces import (  # noqa: E402
    NoModelCallTracesError,
    build_power_mh_time_per_call_summary_rows,
    build_power_mh_time_summary_rows,
    extract_model_call_traces,
    power_mh_time_per_call_trajectories,
)
from case_studies.plot_config import BLUE, DARK_GRAY, GRAY  # noqa: E402

DEFAULT_ROOT = paths.CASE_STUDIES_DIR / "logs-entropycut" / "lcb_v6"
DEFAULT_MODEL = "qwen3.5-9b"
BASELINE_LABEL = "EntropyCut"
SUBTREE_LABEL = "PreSTO-EntropyCut"
FIGURE_SUFFIX = ".prefetch-budget-sweep.entropycut-vs-presto.pdf"
# Labels, output suffix and log filter of each sampler family; main() applies the chosen one.
FAMILIES = {
    "entropy": ("EntropyCut", "PreSTO-EntropyCut", ".prefetch-budget-sweep.entropycut-vs-presto.pdf", True),
    "uniform": ("PowerMH", "PreSTO-PowerMH", ".prefetch-budget-sweep.powermh-vs-presto.pdf", False),
}

FIGURE_FONT_SIZE = 11.0
PANEL_WIDTH = 3.25
FIGURE_HEIGHT = 3.55
MEAN_LINE_WIDTH = 1.8
BASELINE_LINE_WIDTH = 2.0
SD_BAND_ALPHA = 0.10
# Light-to-dark green ramp: larger budgets read darker, and every shade stays PreSTO's hue.
BUDGET_CMAP = LinearSegmentedColormap.from_list(
    "presto_budget", ["#A6DD6F", plot_config.GREEN, "#2E7D13", "#0E3B06"]
)


@dataclass(frozen=True)
class SubtreeRun:
    """One PreSTO-EntropyCut run with its same-directory EntropyCut baseline."""

    date: str
    budget: int
    log_path: Path
    analysis: combined.AnalysisRows
    baseline: combined.PowerMHTimeData


def discover_runs(root: Path, model: str, rank: str, steps: int, entropy_cut: bool) -> list[SubtreeRun]:
    """Parse every complete matching subtree log and pair it with its baseline."""
    runs: list[SubtreeRun] = []
    baselines: dict[Path, combined.PowerMHTimeData] = {}
    for log_path in sorted(root.glob(f"*/vllm/*.model-{model}.alpha*.steps{steps}.*.subtreePrefetch.vllm.log")):
        if run_naming.canonical_rank(run_naming.rank_of(log_path)) != rank:
            continue
        if (".cut-entropy" in log_path.name) != entropy_cut:
            continue
        try:
            traces = extract_model_call_traces(log_path)
        except NoModelCallTracesError:
            # A run that stopped before its first model call leaves a log with no traces at all.
            traces = []
        if not traces or not all(trace.complete for trace in traces):
            print(f"skip incomplete {log_path}")
            continue
        try:
            baseline_log = combined.find_power_mh_log(log_path.parent, log_path, None)
        except ValueError:
            print(f"skip without baseline {log_path}")
            continue
        comparison = combined.single_budget_comparison(log_path, traces_per_rank=len(traces))
        if baseline_log not in baselines:
            baselines[baseline_log] = combined.load_power_mh_time_data(baseline_log)
        runs.append(SubtreeRun(
            date=log_path.parent.parent.name,
            budget=run_naming.prefetch_budget_of(log_path),
            log_path=log_path,
            analysis=comparison.batch_analyses[0][1],
            baseline=baselines[baseline_log],
        ))
    if not runs:
        raise SystemExit(f"No complete {rank!r} {model} subtree logs under {root}")
    return runs


def pooled_baseline(runs: Sequence[SubtreeRun]) -> combined.PowerMHTimeData:
    """Pool the distinct EntropyCut baseline traces used by the plotted runs."""
    distinct = {run.baseline.log_path: run.baseline for run in runs}
    trajectories = [
        (f"{path.parent.parent.name}:{trace_id}", values)
        for path, data in sorted(distinct.items())
        for trace_id, values in data.trajectories
    ]
    return combined.PowerMHTimeData(
        log_path=Path(";".join(str(path) for path in sorted(distinct))),
        trajectories=trajectories,
        summary=build_power_mh_time_summary_rows(trajectories),
        time_per_call_trajectories=power_mh_time_per_call_trajectories(trajectories),
        time_per_call_summary=build_power_mh_time_per_call_summary_rows(trajectories),
    )


def draw_mean_band(
    ax: Axes, rows: Sequence[dict], mean_key: str, sd_key: str, scale: float, **line_kwargs: object
) -> None:
    """Draw one step mean with a light sample-SD band."""
    iterations = [int(row["mh_iteration"]) for row in rows]
    means = [float(row[mean_key]) * scale for row in rows]
    sds = [float(row[sd_key]) * scale for row in rows]
    ax.fill_between(
        iterations, [max(m - s, 0.0) for m, s in zip(means, sds)], [m + s for m, s in zip(means, sds)],
        step="post", color=line_kwargs["color"], alpha=SD_BAND_ALPHA, linewidth=0, zorder=1,
    )
    ax.step(iterations, means, where="post", zorder=3, **line_kwargs)


def plot(runs: list[SubtreeRun], rank: str, output: Path) -> tuple[Path, list[dict]]:
    """Render the three-panel sweep and return the rows it plotted."""
    budgets = sorted({run.budget for run in runs})
    # One trajectory per budget: the latest recorded run of that budget.
    latest = {budget: max((run for run in runs if run.budget == budget), key=lambda run: run.date) for budget in budgets}
    baseline = pooled_baseline(list(latest.values()))
    colors = {
        budget: BUDGET_CMAP(index / max(len(budgets) - 1, 1)) for index, budget in enumerate(budgets)
    }

    figure, (time_ax, efficiency_ax, per_call_ax) = plt.subplots(1, 3, figsize=(PANEL_WIDTH * 3, FIGURE_HEIGHT))
    rows: list[dict] = []
    panels = (
        ("a", time_ax, "empirical_time_summary", "mean_cumulative_seconds", "std_cumulative_seconds",
         1 / combined.SECONDS_PER_MINUTE, combined.TIME_AXIS_LABEL),
        ("b", efficiency_ax, "transitions_per_call_summary", "mean_mh_transitions_per_call",
         "std_mh_transitions_per_call", 1.0, combined.TRANSITIONS_PER_CALL_AXIS_LABEL),
        ("c", per_call_ax, "time_per_call_summary", "mean_empirical_seconds_per_call",
         "std_empirical_seconds_per_call", 1.0, combined.TIME_PER_CALL_AXIS_LABEL),
    )
    for panel, ax, attribute, mean_key, sd_key, scale, label in panels:
        final_iteration = 0
        for budget in budgets:
            run = latest[budget]
            summary = getattr(run.analysis, attribute)
            final_iteration = max(final_iteration, max(int(row["mh_iteration"]) for row in summary))
            draw_mean_band(ax, summary, mean_key, sd_key, scale, color=colors[budget], linewidth=MEAN_LINE_WIDTH)
            rows += [{
                "panel": panel, "metric": label, "method": SUBTREE_LABEL, "run_date": run.date,
                "prefetch_budget": budget, "mh_iteration": row["mh_iteration"], "trace_count": row["trace_count"],
                "mean": float(row[mean_key]) * scale, "sample_sd": float(row[sd_key]) * scale,
            } for row in summary]
        if panel == "b":
            ax.axhline(1.0, color=BLUE, linestyle="--", linewidth=BASELINE_LINE_WIDTH, zorder=4)
            baseline_rows = [{"mh_iteration": step, "trace_count": len(baseline.trajectories), "mean": 1.0,
                              "sample_sd": 0.0} for step in range(1, final_iteration + 1)]
        else:
            source, base_mean, base_sd = (
                (baseline.summary, "mean_cumulative_seconds", "std_cumulative_seconds") if panel == "a"
                else (baseline.time_per_call_summary, "mean_empirical_seconds_per_call",
                      "std_empirical_seconds_per_call")
            )
            draw_mean_band(ax, source, base_mean, base_sd, scale, color=BLUE, linewidth=BASELINE_LINE_WIDTH,
                           linestyle="--")
            baseline_rows = [{"mh_iteration": row["mh_iteration"], "trace_count": row["trace_count"],
                              "mean": float(row[base_mean]) * scale, "sample_sd": float(row[base_sd]) * scale}
                             for row in source]
        rows += [{"panel": panel, "metric": label, "method": BASELINE_LABEL, "run_date": "pooled",
                  "prefetch_budget": 1, **row} for row in baseline_rows]
        ax.set_xlim(0, final_iteration)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(combined.TRANSITION_AXIS_LABEL, fontsize=FIGURE_FONT_SIZE, labelpad=0)
        ax.set_ylabel(label, fontsize=FIGURE_FONT_SIZE)

    for ax, letter in zip((time_ax, efficiency_ax, per_call_ax), "abc"):
        ax.text(-0.02, 1.02, rf"\textbf{{({letter})}}", transform=ax.transAxes, fontsize=FIGURE_FONT_SIZE,
                ha="right", va="bottom", clip_on=False)
        ax.minorticks_off()
        ax.tick_params(axis="both", which="both", top=False, right=False, labelsize=FIGURE_FONT_SIZE)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in (time_ax, efficiency_ax, per_call_ax):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.15, top=0.76, wspace=0.30)
    first, last = time_ax.get_position(), per_call_ax.get_position()
    source_log = runs[0].log_path
    dataset = "LCB V6" if run_naming.dataset_of(source_log) == "lcb_v6" else run_naming.dataset_of(source_log)
    model = run_naming.model_display_name(
        source_log, combined.experiment_figure_title(source_log).partition(", ")[2]
    )
    figure.text(
        (first.x0 + last.x1) / 2, 0.985,
        f"dataset: {dataset}, base LLM: {model}, inference engine: {run_naming.backend_of(source_log)}, "
        f"rank: {rank.replace('_', '-')}",
        color=DARK_GRAY, fontsize=FIGURE_FONT_SIZE, fontweight="semibold", ha="center", va="top",
    )
    handles = [Line2D([], [], color=BLUE, linestyle="--", linewidth=BASELINE_LINE_WIDTH,
                      label=f"{BASELINE_LABEL} ({len(baseline.trajectories)} traces)")]
    # One entry for every budget: its handle shows the budget colours side by side, light (small N) to dark.
    budget_lines = tuple(Line2D([], [], color=colors[budget], linewidth=MEAN_LINE_WIDTH + 0.6) for budget in budgets)
    handles.append(Rectangle((0, 0), 1, 1, facecolor=GRAY, alpha=0.35, edgecolor="none"))
    labels = [handles[0].get_label(), rf"{SUBTREE_LABEL} ($N_{{\mathrm{{pf}}}}={', '.join(str(b) for b in budgets)}$)", r"$\pm 1$ SD"]
    figure.legend(handles=[handles[0], budget_lines, handles[1]], labels=labels, loc="upper center",
                  bbox_to_anchor=((first.x0 + last.x1) / 2, 0.915), ncol=3, fontsize=FIGURE_FONT_SIZE - 1,
                  frameon=True, handlelength=1.8 * len(budgets) / 2, columnspacing=1.2, edgecolor=GRAY,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    return save_figure(figure, output, pad_inches=0.01), rows


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, collect runs, and write the PDF and its source CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset log root holding <date>/vllm.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model tag in the log filenames.")
    parser.add_argument("--rank", default="accept_first", help="Traversal rank policy to plot.")
    parser.add_argument("--family", choices=sorted(FAMILIES), default="entropy",
                        help="Sampler family: EntropyCut (entropy) or PowerMH with uniform cuts (uniform).")
    parser.add_argument("--steps", type=int, default=100, help="MH steps of the runs to plot.")
    parser.add_argument("--output", type=Path, default=None, help="Output PDF; defaults beside --root.")
    parser.add_argument("--time-unit", choices=("min", "sec"), default="min",
                        help="Unit of cumulative model-call time in panel (a).")
    args = parser.parse_args(argv)
    combined.set_time_unit(args.time_unit)

    global BASELINE_LABEL, SUBTREE_LABEL, FIGURE_SUFFIX
    BASELINE_LABEL, SUBTREE_LABEL, FIGURE_SUFFIX, entropy_cut = FAMILIES[args.family]
    runs = discover_runs(args.root.resolve(), args.model, args.rank, args.steps, entropy_cut)
    if args.output is None:
        prefix = run_naming.shared_run_config_of(runs[0].log_path)
        output = args.root.resolve() / f"{prefix}.rank-{args.rank}{FIGURE_SUFFIX}"
    else:
        output = args.output.resolve()
    written, rows = plot(runs, args.rank, output)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    write_dict_rows(written.with_suffix(".summary.csv"), [{key: row.get(key, "") for key in fieldnames}
                                                          for row in rows], fieldnames=fieldnames)
    for run in runs:
        print(f"{run.date} N={run.budget}: {run.log_path.name}")
    print(f"wrote {written}")


if __name__ == "__main__":
    main()

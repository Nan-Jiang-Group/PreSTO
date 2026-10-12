#!/usr/bin/env python3
"""Draw LCB V6 call/time panels for every model with usable recorded traces.

Run with:
    MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    XDG_CACHE_HOME=/private/tmp/power-sharpening-xdg-cache \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_lcb_v6_available_model_call_time.py

Use matched PowerMH/PreSTO pairs when available. Otherwise plot only the recorded
method and label the missing counterpart. Never substitute another token limit,
dataset version, model, or cut distribution for a missing timing baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.figures.model_calls import draw_accept_first_power_mh_model_calls_and_empirical_time as combined
from case_studies.extract.logs.model_call_traces import (
    experiment_figure_title,
    extract_model_call_traces,
    group_rows_by_rank,
    group_rows_by_trace,
    map_trace_ranks,
)
from case_studies.extract.run_naming import (
    dataset_of,
    model_display_name,
    model_of,
    prefetch_budget_of,
    rank_of,
)

ROOT = paths.REPO_ROOT
DEFAULT_OUTPUT = ROOT / "case_studies/lcb-v6-model-call-time-figures"
KINDS = (
    combined.EMPIRICAL_TIME,
    combined.TRANSITIONS_PER_CALL,
    combined.TIME_PER_CALL,
)


def complete_traces(path: Path) -> list:
    """Keep complete, 100-transition traces from a true LCB V6 run."""
    try:
        if not experiment_figure_title(path).startswith("LiveCodeBench-release_v6, "):
            return []
        return [
            trace for trace in extract_model_call_traces(path)
            if trace.complete and trace.total_mh_steps == 100
        ]
    except ValueError:
        return []


def single_method_figure(
    comparison, baseline, source: Path, output: Path, *, note: str | None = None,
) -> str:
    """Reuse the reference styles and statistics for an explicitly unpaired run."""
    is_subtree = comparison is not None
    method = combined.SUBTREE_METHOD_LABEL if is_subtree else combined.POWER_METHOD_LABEL
    color = combined.SUBTREE_COLOR if is_subtree else combined.POWER_COLOR
    if note is None:
        note = (
            "PreSTO-PowerMH only: matching PowerMH timing unavailable."
            if is_subtree else "PowerMH only: matching subtree timing unavailable."
        )
    figure, axes = plt.subplots(
        1, 3, figsize=(combined.PANEL_WIDTH * 3, combined.FIGURE_HEIGHT), sharex=True,
    )
    for ax, kind, label in zip(axes, KINDS, ("(a)", "(b)", "(c)"), strict=True):
        if is_subtree:
            analysis = comparison.batch_analyses[0][1]
            rows = kind.trajectory_rows(analysis)
            summary = kind.summary_rows(analysis)
            if kind is combined.EMPIRICAL_TIME:
                rows = combined.convert_time_rows_to_minutes(rows, kind.value_key)
                summary = combined.convert_time_rows_to_minutes(summary, kind.mean_key, kind.sd_key)
            trace_rows = group_rows_by_trace(rows)
            _, means, deviations = combined.draw_subtree_trajectories(
                ax, trace_rows, map_trace_ranks(trace_rows), group_rows_by_rank(summary),
                kind, rank_name=rank_of(source),
            )
            limits, iteration_limits, _ = combined.trajectory_limits(rows, summary, kind)
            upper = limits[1]
        elif kind is combined.TRANSITIONS_PER_CALL:
            means = combined.draw_power_mh_transitions_per_call_reference(ax, 100)
            deviations = [0.0] * len(means)
            upper = 1.4
            iteration_limits = (0.0, 100.0)
        else:
            if kind is combined.EMPIRICAL_TIME:
                _, means, deviations = combined.draw_power_mh_time_trajectories(ax, baseline)
                traces = [[value / 60 for value in values] for _, values in baseline.trajectories]
            else:
                _, means, deviations = combined.draw_power_mh_time_per_call_trajectories(ax, baseline)
                traces = [values for _, values in baseline.time_per_call_trajectories]
            upper = max(
                max(max(values) for values in traces),
                max(mean + sd for mean, sd in zip(means, deviations, strict=True)),
            )
            iteration_limits = (0.0, 100.0)
        # Reserve space above measured curves for the terminal statistic.
        upper *= 1.18
        combined.style_panel(ax, kind, cost_limits=(0.0, upper), iteration_limits=iteration_limits)
        combined.statistic_text(
            ax, 0.5, 0.95, method, means[-1], deviations[-1],
            color=color, ha="center", va="top",
        )
        ax.text(
            -0.12, combined.PANEL_LABEL_Y, rf"\textbf{{{label}}}",
            transform=ax.transAxes, fontsize=combined.FIGURE_FONT_SIZE,
            ha="right", va="top", clip_on=False,
        )
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=combined.FIGURE_FONT_SIZE)
    figure.subplots_adjust(
        left=combined.LEFT_MARGIN_INCHES / (combined.PANEL_WIDTH * 3),
        right=0.995, bottom=combined.BOTTOM_MARGIN_INCHES / combined.FIGURE_HEIGHT,
        top=combined.PLOT_TOP, wspace=combined.PANEL_WSPACE,
    )
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    dataset, model = experiment_figure_title(source).split(", ", 1)
    model = model_display_name(source, model)
    if dataset_of(source) == "lcb_v6":
        dataset = "LCB V6"
    token_limit = re.search(r"\.maxnew(\d+)\.", source.name).group(1)
    rank_label = rank_of(source).replace("bfs_", "BFS ").replace("_", "-") if is_subtree else ""
    settings = (
        f"{rank_label}, $B={prefetch_budget_of(source)}$, " if is_subtree else ""
    ) + f"max new tokens: {token_limit}"
    figure.text(
        (left + right) / 2, combined.TITLE_Y,
        f"dataset: {dataset}, base LLM: {model}, inference engine: vLLM, {settings}",
        color=combined.DARK_GRAY, fontsize=combined.FIGURE_FONT_SIZE, ha="center", va="top",
    )
    combined.add_method_legends(figure, left=left, right=right, methods=(method,))
    figure.text(
        (left + right) / 2, -0.015, note,
        color=combined.DARK_GRAY, fontsize=10, ha="center", va="top",
    )
    save_figure(figure, output, pad_inches=0.02)
    return note


def main() -> None:
    """Discover real LCB V6 inputs and write model-specific PDFs and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "case_studies/logs/lcb_v6"
    subtree_logs: dict[str, list[Path]] = {}
    power_logs: dict[str, list[Path]] = {}
    for path in sorted(logs.rglob("*.log")):
        if dataset_of(path) != "lcb_v6" or ".cut-entropy" in path.name:
            continue
        if ".subtreePrefetch." in path.name:
            if rank_of(path) != "bfs_accept_first" or prefetch_budget_of(path) != 20:
                continue
            if ".maxnew1024." not in path.name or not complete_traces(path):
                continue
            subtree_logs.setdefault(model_of(path), []).append(path)
        elif ".powerMH." in path.name and complete_traces(path):
            power_logs.setdefault(model_of(path), []).append(path)
    records = []
    for model in sorted(subtree_logs.keys() | power_logs.keys()):
        candidates = subtree_logs.get(model, [])
        subtree = max(candidates) if candidates else None
        baselines = power_logs.get(model, [])
        comparison = combined.single_budget_comparison(subtree, traces_per_rank=10) if subtree else None
        matching = [path for path in baselines if combined._power_mh_config_matches(subtree, path)] if subtree else baselines
        power = max(matching) if matching else None
        baseline = combined.load_power_mh_time_data(power) if power else None
        source = subtree or power
        mode = "matched-power-mh" if subtree and power else "subtree-only" if subtree else "power-mh-only"
        output = output_directory / f"dataset-lcb_v6.model-{model}.{mode}.model-calls-and-empirical-time.pdf"
        if subtree and power:
            combined.find_power_mh_log(logs, subtree, power)
            combined.plot_figure(
                comparison, baseline, output, rank_name="bfs_accept_first",
                run_label="BFS accept-first, $B=20$",
            )
            note = "Full configuration-matched PowerMH and PreSTO-PowerMH comparison."
        else:
            note = single_method_figure(comparison, baseline, source, output)
        used = [path for path in (subtree, power) if path]
        record = {
            "model": model, "mode": mode, "pdf": str(output), "note": note,
            "subtree_log": str(subtree) if subtree else None,
            "power_mh_log": str(power) if power else None,
            "unmatched_power_mh_logs": [str(path) for path in baselines if path != power],
            "subtree_traces_plotted": comparison.batch_analyses[0][1].traces_per_rank if comparison else 0,
            "power_mh_traces_plotted": len(baseline.trajectories) if baseline else 0,
            "mh_transitions": 100,
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in used},
        }
        records.append(record)
        print(f"{model}: {mode}: {output}", flush=True)
    (output_directory / "manifest.json").write_text(json.dumps({
        "description": "LCB V6 measured call/time panels for every available base model.",
        "timing_scope": "Only configuration-matched runs are overlaid. Matching hardware is not independently verified.",
        "figures": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

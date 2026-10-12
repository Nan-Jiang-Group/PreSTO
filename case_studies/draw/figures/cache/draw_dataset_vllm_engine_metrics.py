#!/usr/bin/env python3
"""Combine all logs for one dataset and model into one engine-metric figure.

Run with the repository environment; the PDF gets sample and run-inventory CSVs:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_dataset_vllm_engine_metrics.py \
        --directory /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/lcb_v6 \
        --dataset lcb_v6 --model qwen3.5-9b \
        --exclude-log /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/lcb_v6/2026-09-07/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps10.blocks-1.samples20.maxnew1024.seed10086.powerMH.vllm.log \
        --exclude-log /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/lcb_v6/2026-08-28/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew3072.seed10086.powerMH.vllm.log \
        --output /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/output/pdf/lcb-v6-qwen3.5-9b-all-log-engine-metrics.pdf

Each log with metric records remains a separate row, including incomplete logs. Logs without metric records or selected
with --exclude-log are omitted from the figure. Row labels show the method, traversal rule, and budget; the
run-inventory CSV preserves all source logs, dates, settings, and completion status. Elapsed time starts at each log's
first metric sample; time ranges differ by row, while each metric column shares a y scale. Values are not smoothed or
replaced by resource-JSON aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.figures.cache.draw_vllm_engine_metrics import PANELS
from case_studies.extract.logs.vllm_engine_metrics import EngineMetricSample, elapsed_seconds, parse_engine_metrics
from case_studies.extract.run_naming import (
    RANK_ORDER,
    dataset_of,
    latex_text,
    model_of,
    prefetch_budget_of,
    rank_label,
    rank_of,
)
from case_studies.plot_config import POWER_METHOD_LABEL, SUBTREE_METHOD_LABEL


@dataclass(frozen=True)
class RunLog:
    """One input log, its run identity, and all available logger samples."""

    path: Path
    samples: tuple[EngineMetricSample, ...]
    method: str
    rank: str
    budget: int | None
    steps: int
    maxnew: int
    date: str
    dataset_size: int | None
    completed: bool
    sha256: str

    @classmethod
    def read(cls, path: Path, dataset: str) -> RunLog:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if dataset.startswith("lcb_v"):
            expected = "release_" + dataset.removeprefix("lcb_")
            releases = set(re.findall(r"LiveCodeBench-(release_v\d+)", raw))
            if releases != {expected}:
                raise ValueError(f"Dataset release mismatch in {path}: {releases}")
        engines = set(re.findall(r"\[loggers\.py:\d+\]\s+Engine\s+([^:]+):", raw))
        if len(engines) > 1:
            raise ValueError(f"Multiple engines in {path}; cannot merge their samples")
        # An absent logger record stays recorded in the inventory. Parsing errors in logs that do contain records remain
        # errors rather than becoming empty rows.
        samples = tuple(parse_engine_metrics(path)) if "Avg generation throughput:" in raw else ()
        is_baseline = path.name.endswith(".powerMH.vllm.log")
        size_match = re.search(r"LiveCodeBench-release_v\d+ of length (\d+)", raw)

        def setting(name: str) -> int:
            match = re.search(rf"\.{name}(\d+)\.", path.name)
            if match is None:
                raise ValueError(f"Missing {name} setting in {path}")
            return int(match.group(1))

        return cls(
            path=path,
            samples=samples,
            method=POWER_METHOD_LABEL if is_baseline else SUBTREE_METHOD_LABEL,
            rank="" if is_baseline else rank_of(path),
            budget=None if is_baseline else prefetch_budget_of(path),
            steps=setting("steps"), maxnew=setting("maxnew"),
            date=path.parent.parent.name,
            dataset_size=int(size_match.group(1)) if size_match else None,
            completed="INFO: output saved to" in raw,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    @property
    def status(self) -> str:
        if not self.samples:
            return "no metric records"
        return "completed" if self.completed else "partial log"

    def label(self) -> str:
        lines = [self.method]
        if self.rank:
            lines.append(f"{rank_label(self.rank)}, budget {self.budget}")
        return "\n".join(lines)


def discover_logs(directory: Path, dataset: str, model: str) -> list[RunLog]:
    """Select exact filename dataset/model identities across all run dates."""
    paths = [
        path for path in sorted(directory.rglob("*.vllm.log"))
        if path.name.endswith((".powerMH.vllm.log", ".subtreePrefetch.vllm.log"))
        and dataset_of(path) == dataset and model_of(path) == model
    ]
    if not paths:
        raise ValueError(f"No logs for {dataset}/{model} under {directory}")
    runs = [RunLog.read(path, dataset) for path in paths]
    # Newest sweep first, then baseline and budgets within each dated setup.
    runs.sort(key=lambda run: (
        -run.steps, run.maxnew, run.method != POWER_METHOD_LABEL,
        RANK_ORDER.get(run.rank, -1), run.budget or 0, str(run.path),
    ))
    runs.sort(key=lambda run: run.date, reverse=True)
    return runs


def write_source_data(
    runs: list[RunLog], plotted_runs: list[RunLog], output: Path
) -> None:
    """Save plotted samples and an inventory that also covers omitted logs."""
    inventory, rows = [], []
    figure_rows = {run.path: index for index, run in enumerate(plotted_runs, start=1)}
    for run in runs:
        row_id = figure_rows.get(run.path)
        times = elapsed_seconds(list(run.samples)) if run.samples else []
        identity = {
            "figure_row": row_id, "source_log": str(run.path),
            "method": run.method, "rank": run.rank,
            "prefetch_budget": run.budget, "date": run.date,
            "mh_steps": run.steps, "max_new_tokens": run.maxnew,
            "dataset_size": run.dataset_size, "status": run.status,
        }
        inventory.append({
            **identity, "included_in_figure": row_id is not None,
            "omission_reason": (
                "" if row_id is not None else
                "no metric records" if not run.samples else "excluded by --exclude-log"
            ),
            "metric_samples": len(run.samples),
            "elapsed_minutes": times[-1] / 60 if times else None,
            "source_sha256": run.sha256,
        })
        if row_id is None:
            continue
        for sample, seconds in zip(run.samples, times, strict=True):
            rows.append({
                **identity, **asdict(sample),
                "timestamp": sample.timestamp.isoformat(sep=" "),
                "elapsed_seconds": seconds,
            })
    write_dict_rows(output.with_suffix(".runs.csv"), inventory)
    if rows:
        write_dict_rows(output.with_suffix(".csv"), rows)


def draw_all_logs(runs: list[RunLog], output: Path, title: str) -> Path:
    """Draw four metric columns with paired throughput and request series."""
    all_samples = [sample for run in runs for sample in run.samples]
    if not all_samples:
        raise ValueError("The selected logs contain no engine-metric samples")
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10}):
        figure, axes_grid = plt.subplots(
            len(runs), len(PANELS), figsize=(16.0, 1.75 * len(runs) + 1.2),
            sharey="col", squeeze=False,
        )
        limits = [panel.limits(panel.values(all_samples)) for panel in PANELS]
        column_titles = [
            "(a) Token throughput (tokens/s)", "(b) Request count",
            r"(c) GPU KV cache usage (\%)", r"(d) Prefix-cache hit rate (\%)",
        ]
        for row_index, run in enumerate(runs):
            axes = axes_grid[row_index]
            times = [seconds / 60 for seconds in elapsed_seconds(list(run.samples))]
            for column, (ax, panel) in enumerate(zip(axes, PANELS, strict=True)):
                ax.set_ylim(*limits[column])
                ax.grid(False)
                ax.minorticks_off()
                ax.tick_params(top=False, right=False)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=panel.integer_y))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                panel.draw(ax, times, run.samples, show_legend=row_index == 0)
                ax.set_xlim(0, max(times[-1], 1 / 60))
                if row_index == 0:
                    ax.set_title(column_titles[column], fontsize=13, pad=12)
            axes[0].text(
                -0.24, 0.5, run.label(), transform=axes[0].transAxes,
                ha="right", va="center", fontsize=10, linespacing=1.4,
            )
        figure.suptitle(title, fontsize=18, y=0.992)
        figure.supxlabel("Elapsed time since first logger sample (min)", fontsize=15, y=0.019)
        figure.subplots_adjust(
            left=0.21, right=0.99, bottom=0.043, top=0.963,
            hspace=0.55, wspace=0.26,
        )
        return save_figure(figure, output, pad_inches=0.06)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-log", type=Path, action="append", default=[],
        help="Omit this source log from the figure; repeat to omit multiple logs.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    dataset_label = "LCB V6" if args.dataset == "lcb_v6" else latex_text(args.dataset)
    model_label = "Qwen3.5-9B" if args.model == "qwen3.5-9b" else latex_text(args.model)
    runs = discover_logs(args.directory.resolve(), args.dataset, args.model)
    excluded_paths = {path.resolve() for path in args.exclude_log}
    unknown_paths = excluded_paths - {run.path for run in runs}
    if unknown_paths:
        parser.error("Excluded logs are not in the selected dataset/model: " +
                     ", ".join(str(path) for path in sorted(unknown_paths)))
    plotted_runs = [run for run in runs if run.samples and run.path not in excluded_paths]
    draw_all_logs(
        plotted_runs, output,
        f"{dataset_label} / {model_label}: {len(plotted_runs)} logs with metrics",
    )
    write_source_data(runs, plotted_runs, output)
    print(f"Wrote {output}")
    print(f"{len(runs)} source logs; {len(plotted_runs)} plotted logs; "
          f"{sum(len(run.samples) for run in plotted_runs)} plotted samples")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot base-model likelihood and confidence distributions from sampler logs.

Extracts the per-response diagnostics printed by the subtree-prefetching MH
runner, i.e. the length-normalized log-likelihood (1/n_K) log p_0(x^(K) | x_0)
and confidence (1/n_K) sum_t sum_u p_0 log p_0. The histogram grid uses one column per prefetch count and compares
traversal rank functions within each column, with log-likelihood and confidence on separate rows.

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/plot_likelihood_and_confidence.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_studies import paths
from case_studies.plot_config import (
    BLACK,
    GRAY,
    PANEL_LABEL_FONT_SIZE,
    PANEL_LABEL_FONT_WEIGHT,
    TRANSITION_RANK_COLORS,
    apply_plot_style,
)
from case_studies.plot_config import (
    COLOR_PALETTE as GROUP_COLORS,
)
from case_studies.plot_config import (
    DISTRIBUTION_HISTOGRAM_ALPHA as HIST_ALPHA,
)
from case_studies.plot_config import (
    DISTRIBUTION_HISTOGRAM_BAR_WIDTH as BAR_WIDTH_FRACTION,
)
from case_studies.plot_config import (
    DISTRIBUTION_MEAN_LINE_WIDTH as MEAN_LINE_WIDTH,
)

# Render without a display: this is a batch tool and may run over SSH.
matplotlib.use("Agg")
apply_plot_style()

SCRIPT_DIR = paths.DRAW_DIR
DEFAULT_LOG_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_STEM = "likelihood-and-confidence"
DEFAULT_BINS = 10

# Deliberately far finer than plot_config's shared histogram edge (0.55): these panels overlay up to six rank
# distributions per axis, where a heavier edge reads as fill and hides which bars belong to which rank. Set here rather
# than imported and then reassigned, which is what this file used to do.
HIST_LINE_WIDTH = 0.1
PANEL_WIDTH = 3.2
PANEL_HEIGHT = 2.4
PANEL_SPECS = (
    ("log_likelihood", "Token log-likelihood under $p_0$"),
    ("confidence", "Token confidence under $p_0$"),
)
RANK_ORDER = {
    "accept_first": 0,
    "reject_first": 1,
    "longest_first": 2,
    "smallest_cut_diff": 3,
    "longest_path_first": 4,
    "bfs_reject_first": 5,
}

SAMPLE_PATTERN = re.compile(
    r"^IDX\s+(?P<sample>\d+)\b(?:\s+QUESTION:\s*(?P<question>.*))?$"
)
CONFIG_HEADER_PATTERN = re.compile(r"resolved configuration:\s*$")
CONFIG_ENTRY_PATTERN = re.compile(r"^  (?P<key>[A-Za-z_]\w*): (?P<value>.*)$")
DIAGNOSTICS_PATTERN = re.compile(
    r"^response length: (?P<chars>\d+), "
    r"response tokens: (?P<tokens>\d+), "
    r"log-likelihood: (?P<log_likelihood>-?\d+(?:\.\d+)?), "
    r"confidence: (?P<confidence>-?\d+(?:\.\d+)?)\s*$"
)
PREDICTION_PATTERN = re.compile(
    r"pred length: \d+, pred: (?P<prediction>.*), "
    r"is_correct: (?P<is_correct>True|False)\s*$"
)
EMPTY_PREDICTION_PATTERN = re.compile(r"^\s*pred is empty\s*$")

# Runs logged before the sampler's ``subtree_batch_size`` became the prefetch budget echo the old key in their resolved
# configuration. Both spellings are read so those logs still plot; only the first is written.
PREFETCH_BUDGET_CONFIG_KEYS = ("prefetch_budget", "subtree_batch_size")


def prefetch_budget_in(config: dict[str, object], default: object) -> object:
    """Read the prefetch budget from a logged configuration, either spelling."""
    for key in PREFETCH_BUDGET_CONFIG_KEYS:
        value = config.get(key)
        if value not in (None, ""):
            return value
    return default

Row = dict[str, object]


@dataclass
class ResponseDiagnostics:
    """Base-model diagnostics for one sampled response."""

    sample_idx: int
    question: str
    response_chars: int
    response_tokens: int
    log_likelihood: float
    confidence: float
    prediction: str | None = None
    is_correct: bool | None = None

    @property
    def logprob_sum(self) -> float:
        """Un-normalized log p_0(x^(K) | x_0) implied by the logged mean."""
        return self.log_likelihood * self.response_tokens

    @property
    def neg_entropy_sum(self) -> float:
        """Un-normalized negative entropy implied by the logged mean."""
        return self.confidence * self.response_tokens


@dataclass
class LogRun:
    """One sampler log: its resolved configuration and scored responses."""

    source_log: str
    config: dict[str, object] = field(default_factory=dict)
    samples: list[ResponseDiagnostics] = field(default_factory=list)
    unscored_samples: int = 0

    def label(self, group_key: str | None) -> str:
        """Return the comparison group this run belongs to."""
        if group_key is not None:
            if group_key in self.config:
                return str(self.config[group_key])
            available = ", ".join(sorted(self.config)) or "none"
            raise ValueError(
                f"{self.source_log} has no resolved configuration entry "
                f"{group_key!r}; available keys: {available}"
            )

        transition = self.config.get("tree_transition_type")
        if transition is None:
            cut_type = self.config.get("cut_dist_type")
            transition = f"{cut_type} cut" if cut_type is not None else "unspecified"
        rank = str(self.config.get("rank_fn", "unspecified")).replace("_", " ")
        batch_size = prefetch_budget_in(self.config, "?")
        return f"{transition} / {rank} / batch {batch_size}"


def parse_config_value(raw_value: str) -> object:
    """Interpret a logged configuration value, keeping unparsable text as-is."""
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def parse_log(log_path: Path) -> LogRun:
    """Parse the resolved configuration and response diagnostics from one log."""
    run = LogRun(source_log=log_path.name)
    in_config = False
    sample_idx: int | None = None
    question = ""
    pending: ResponseDiagnostics | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            line = line.rstrip("\n")

            if CONFIG_HEADER_PATTERN.search(line):
                in_config = True
                continue
            if in_config:
                config_match = CONFIG_ENTRY_PATTERN.match(line)
                if config_match:
                    run.config[config_match.group("key")] = parse_config_value(
                        config_match.group("value")
                    )
                    continue
                in_config = False

            sample_match = SAMPLE_PATTERN.match(line)
            if sample_match:
                if sample_idx is not None and pending is None:
                    run.unscored_samples += 1
                sample_idx = int(sample_match.group("sample"))
                question = sample_match.group("question") or ""
                pending = None
                continue

            diagnostics_match = DIAGNOSTICS_PATTERN.match(line)
            if diagnostics_match:
                if sample_idx is None:
                    raise ValueError(
                        f"{log_path.name}:{line_number}: diagnostics before IDX"
                    )
                pending = ResponseDiagnostics(
                    sample_idx=sample_idx,
                    question=question,
                    response_chars=int(diagnostics_match.group("chars")),
                    response_tokens=int(diagnostics_match.group("tokens")),
                    log_likelihood=float(diagnostics_match.group("log_likelihood")),
                    confidence=float(diagnostics_match.group("confidence")),
                )
                run.samples.append(pending)
                continue

            prediction_match = PREDICTION_PATTERN.search(line)
            if prediction_match and pending is not None:
                pending.prediction = prediction_match.group("prediction")
                pending.is_correct = prediction_match.group("is_correct") == "True"
                continue

            # An unparsable answer is graded incorrect rather than skipped, so it still counts against accuracy
            # alongside its diagnostics.
            if EMPTY_PREDICTION_PATTERN.match(line) and pending is not None:
                pending.prediction = ""
                pending.is_correct = False

    if sample_idx is not None and pending is None:
        run.unscored_samples += 1
    return run


def discover_logs(log_dir: Path) -> list[Path]:
    """Return all sampler logs contained in a directory tree."""
    if not log_dir.is_dir():
        raise ValueError(f"Log directory does not exist: {log_dir}")
    logs = sorted(path for path in log_dir.rglob("*.log") if path.is_file())
    if not logs:
        raise ValueError(f"No .log files found in {log_dir}")
    return logs


def build_sample_rows(runs: Sequence[LogRun], group_key: str | None) -> list[Row]:
    """Build one source-data row per scored response."""
    rows: list[Row] = []
    for run in runs:
        label = run.label(group_key)
        for sample in run.samples:
            rows.append(
                {
                    "source_log": run.source_log,
                    "group": label,
                    "benchmark": run.config.get("benchmark_name", ""),
                    "model": run.config.get("model_str", ""),
                    "alpha": run.config.get("alpha", ""),
                    "mcmc_steps": run.config.get("mcmc_steps", ""),
                    "num_blocks": run.config.get("num_blocks", ""),
                    "rank_fn": run.config.get("rank_fn", ""),
                    "tree_transition_type": run.config.get("tree_transition_type", ""),
                    "cut_dist_type": run.config.get("cut_dist_type", ""),
                    "prefetch_budget": prefetch_budget_in(run.config, ""),
                    "seed": run.config.get("seed", ""),
                    "sample_idx": sample.sample_idx,
                    "response_chars": sample.response_chars,
                    "response_tokens": sample.response_tokens,
                    "log_likelihood": sample.log_likelihood,
                    "confidence": sample.confidence,
                    "logprob_sum": sample.logprob_sum,
                    "neg_entropy_sum": sample.neg_entropy_sum,
                    "is_correct": sample.is_correct,
                    "question": sample.question,
                }
            )
    return rows


def mean_and_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return the arithmetic mean and sample standard deviation."""
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def group_rows(rows: Sequence[Row], key: str) -> dict[str, list[Row]]:
    """Group rows by one string-valued column."""
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def group_rows_by_n_pf_and_rank(
    rows: Sequence[Row],
) -> dict[int, dict[str, list[Row]]]:
    """Group diagnostics by prefetch count, then traversal rank function."""
    grouped: dict[int, dict[str, list[Row]]] = {}
    for row in rows:
        n_pf = row["prefetch_budget"]
        rank_fn = str(row["rank_fn"])
        if n_pf in (None, "") or not rank_fn:
            raise ValueError("Plot rows require prefetch_budget and rank_fn metadata")
        grouped.setdefault(int(n_pf), {}).setdefault(rank_fn, []).append(row)
    return grouped


def build_summary_rows(sample_rows: Sequence[Row]) -> list[Row]:
    """Summarize both diagnostics and sample quality for each group."""
    rows: list[Row] = []
    for label, group in sorted(group_rows(sample_rows, "group").items()):
        mean_ll, sd_ll = mean_and_sd([float(row["log_likelihood"]) for row in group])
        mean_conf, sd_conf = mean_and_sd([float(row["confidence"]) for row in group])
        mean_tokens, sd_tokens = mean_and_sd(
            [float(row["response_tokens"]) for row in group]
        )
        graded = [row for row in group if row["is_correct"] is not None]
        correct = [row for row in graded if row["is_correct"] is True]
        rows.append(
            {
                "group": label,
                "response_count": len(group),
                "seeds": ",".join(sorted({str(row["seed"]) for row in group})),
                "mean_log_likelihood": mean_ll,
                "sd_log_likelihood": sd_ll,
                "min_log_likelihood": min(
                    float(row["log_likelihood"]) for row in group
                ),
                "max_log_likelihood": max(
                    float(row["log_likelihood"]) for row in group
                ),
                "mean_confidence": mean_conf,
                "sd_confidence": sd_conf,
                "min_confidence": min(float(row["confidence"]) for row in group),
                "max_confidence": max(float(row["confidence"]) for row in group),
                "mean_response_tokens": mean_tokens,
                "sd_response_tokens": sd_tokens,
                "graded_count": len(graded),
                "accuracy": len(correct) / len(graded) if graded else 0.0,
            }
        )
    return rows


def write_csv(rows: Sequence[Row], output_path: Path) -> Path:
    """Write insertion-ordered records to one CSV file."""
    if not rows:
        raise ValueError(f"Cannot write empty data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def output_path(output_prefix: Path, suffix: str) -> Path:
    """Append a suffix without treating the output prefix as an extension."""
    return output_prefix.parent / f"{output_prefix.name}{suffix}"


def rank_sort_key(rank_fn: str) -> tuple[int, str]:
    """Return the stable traversal-rank order shared by case-study figures."""
    return RANK_ORDER.get(rank_fn, len(RANK_ORDER)), rank_fn


def assign_rank_colors(rank_fns: Sequence[str]) -> dict[str, str]:
    """Assign stable traversal-rank colors with a fallback for new methods."""
    return {
        rank_fn: TRANSITION_RANK_COLORS.get(
            rank_fn,
            GROUP_COLORS[index % len(GROUP_COLORS)],
        )
        for index, rank_fn in enumerate(rank_fns)
    }


def add_panel_label(ax: Axes, label: str) -> None:
    """Add a compact Nature-style panel label."""
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight=PANEL_LABEL_FONT_WEIGHT,
        ha="left",
        va="bottom",
    )


def display_rank_label(rank_fn: str) -> str:
    """Render one traversal-rank identifier compactly."""
    return f"rank_fn={rank_fn}"


def style_quantitative_axis(ax: Axes) -> None:
    """Apply consistent quantitative-axis styling."""
    ax.tick_params(
        axis="both",
        which="both",
        length=3,
        width=0.8,
        top=False,
        right=False,
    )


def shared_bin_edges(values: Sequence[float], bins: int) -> list[float]:
    """Build one pooled bin grid so all groups use comparable intervals."""
    low, high = min(values), max(values)
    if high <= low:
        low, high = low - 0.5, high + 0.5
    width = (high - low) / bins
    return [low + index * width for index in range(bins + 1)]


def histogram_legend_handles(
    rank_fns: Sequence[str],
    colors: dict[str, str],
) -> list[Line2D | Patch]:
    """Build shared rank-color and group-mean legend entries."""
    handles: list[Line2D | Patch] = [
        Patch(
            facecolor=colors[rank_fn],
            edgecolor=BLACK,
            linewidth=HIST_LINE_WIDTH,
            alpha=HIST_ALPHA,
            label=display_rank_label(rank_fn),
        )
        for rank_fn in rank_fns
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=GRAY,
            linestyle="--",
            linewidth=MEAN_LINE_WIDTH,
            label="Group mean",
        )
    )
    return handles


def draw_histogram(
    ax: Axes,
    grouped: dict[str, list[Row]],
    rank_fns: Sequence[str],
    colors: dict[str, str],
    *,
    value_key: str,
    x_label: str,
    edges: Sequence[float],
    show_y_label: bool,
) -> None:
    """Draw one diagnostic as density-normalized, side-by-side histograms."""
    ax.hist(
        [
            [float(row[value_key]) for row in grouped[rank_fn]]
            for rank_fn in rank_fns
        ],
        bins=edges,
        density=True,
        histtype="bar",
        rwidth=BAR_WIDTH_FRACTION,
        color=[colors[rank_fn] for rank_fn in rank_fns],
        edgecolor=BLACK,
        linewidth=HIST_LINE_WIDTH,
        alpha=HIST_ALPHA,
    )
    for rank_fn in rank_fns:
        ax.axvline(
            fmean(float(row[value_key]) for row in grouped[rank_fn]),
            color=colors[rank_fn],
            linestyle="--",
            linewidth=MEAN_LINE_WIDTH,
            zorder=4,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density" if show_y_label else "")
    if not show_y_label:
        ax.tick_params(axis="y", labelleft=False)
    style_quantitative_axis(ax)


def save_figure(figure: Figure, output_prefix: Path) -> Path:
    """Save the figure as a PDF."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = output_path(output_prefix, ".pdf")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return path


def plot_panels(
    sample_rows: Sequence[Row],
    output_prefix: Path,
    *,
    bins: int,
) -> Path:
    """Draw one rank-comparison panel per metric and prefetch count."""
    grouped = group_rows_by_n_pf_and_rank(sample_rows)
    n_pf_values = sorted(grouped)
    rank_fns = sorted(
        {rank_fn for rank_groups in grouped.values() for rank_fn in rank_groups},
        key=rank_sort_key,
    )
    colors = assign_rank_colors(rank_fns)
    edges_by_metric = {
        value_key: shared_bin_edges(
            [float(row[value_key]) for row in sample_rows],
            bins,
        )
        for value_key, _ in PANEL_SPECS
    }
    figure, axes = plt.subplots(
        len(PANEL_SPECS),
        len(n_pf_values),
        figsize=(
            PANEL_WIDTH * len(n_pf_values),
            PANEL_HEIGHT * len(PANEL_SPECS),
        ),
        sharex="row",
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    panel_index = 0
    for row_index, (value_key, x_label) in enumerate(PANEL_SPECS):
        for column_index, n_pf in enumerate(n_pf_values):
            ax = axes[row_index, column_index]
            panel_ranks = [
                rank_fn for rank_fn in rank_fns if rank_fn in grouped[n_pf]
            ]
            draw_histogram(
                ax,
                grouped[n_pf],
                panel_ranks,
                colors,
                value_key=value_key,
                x_label=x_label,
                edges=edges_by_metric[value_key],
                show_y_label=column_index == 0,
            )
            ax.set_title(rf"$N_{{\mathrm{{pf}}}}={n_pf}$", pad=4.0)
            add_panel_label(ax, f"({chr(ord('a') + panel_index)})")
            panel_index += 1

    figure.legend(
        handles=histogram_legend_handles(rank_fns, colors),
        loc="outside upper center",
        ncols=4,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    return save_figure(figure, output_prefix)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract length-normalized base-model log-likelihood and "
            "confidence from sampler logs and plot their density histograms."
        )
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        dest="log_dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Folder recursively scanned for *.log files when --logs is absent.",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="+",
        help="Explicit sampler logs, overriding --log-dir discovery.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        help=(
            "Optional resolved-configuration key for CSV and console summaries. "
            "The figure always fixes N_pf per panel and compares rank_fn."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of shared histogram bins in each panel.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "Prefix for CSV and PDF outputs. Defaults beside the logs when all "
            "inputs share one directory."
        ),
    )
    return parser


def resolve_output_prefix(log_paths: Sequence[Path], requested: Path | None) -> Path:
    """Choose an explicit prefix or place outputs beside co-located logs."""
    if requested is not None:
        return requested
    parents = {path.resolve().parent for path in log_paths}
    parent = parents.pop() if len(parents) == 1 else SCRIPT_DIR
    return parent / DEFAULT_OUTPUT_STEM


def main() -> None:
    """Extract logs, write source data, draw the figure, and report outputs."""
    args = build_parser().parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    log_paths = args.logs if args.logs else discover_logs(args.log_dir)
    missing = [path for path in log_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Log files do not exist: {missing}")

    runs = [parse_log(log_path) for log_path in log_paths]
    sample_rows = build_sample_rows(runs, args.group_by)
    if not sample_rows:
        raise ValueError(
            "No response diagnostics found; logs must contain "
            "'response length: ..., log-likelihood: ..., confidence: ...' lines"
        )

    summary_rows = build_summary_rows(sample_rows)
    output_prefix = resolve_output_prefix(log_paths, args.output_prefix)

    output_paths = [
        write_csv(sample_rows, output_path(output_prefix, ".samples.csv")),
        write_csv(summary_rows, output_path(output_prefix, ".summary.csv")),
    ]
    output_paths.append(
        plot_panels(
            sample_rows,
            output_prefix,
            bins=args.bins,
        )
    )

    unscored = sum(run.unscored_samples for run in runs)
    print(
        f"Scanned {len(log_paths)} logs and extracted {len(sample_rows)} scored "
        f"responses ({unscored} samples without diagnostics)."
    )
    for row in summary_rows:
        print(
            f"  {row['group']}: n={row['response_count']}, "
            f"log-likelihood {row['mean_log_likelihood']:.6f} "
            f"+/- {row['sd_log_likelihood']:.6f}, "
            f"confidence {row['mean_confidence']:.6f} "
            f"+/- {row['sd_confidence']:.6f}"
        )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

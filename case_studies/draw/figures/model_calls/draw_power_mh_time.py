#!/usr/bin/env python3
"""Plot the PowerMH baseline's cumulative model-call time on its own.

The same measurement is drawn as a reference line behind every panel of ``draw_model_calls_step.py``'s empirical-time
figure. There it answers "how far does a prefetch budget get for the baseline's money"; here it is the subject, so the
per-sample spread behind that mean is drawn as well.

The input directory is searched recursively and every PowerMH log found gets its own figure, named after its run
configuration and written beside the log.

Run with:

    case_studies/draw/.venv/bin/python \
        case_studies/draw/draw_power_mh_time.py \
        --directory case_studies/logs/math500
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.rank_plotting import draw_cumulative_time_panel
from case_studies.extract.logs.model_call_traces import (
    POWER_MH_LOG_GLOB,
    Row,
    build_power_mh_time_rows,
    build_power_mh_time_summary_rows,
    power_mh_trace_times,
)
from case_studies.extract.run_naming import config_output_prefix, output_path
from case_studies.plot_config import BLUE as POWER_COLOR
from case_studies.plot_config import apply_plot_style

apply_plot_style()

DEFAULT_DIRECTORY = paths.LOGS_DIR / "math500"
OUTPUT_SUFFIX = ".power-mh-time"
FIGURE_SUFFIX = f"{OUTPUT_SUFFIX}.pdf"
DATA_SUFFIX = f"{OUTPUT_SUFFIX}.csv"
SUMMARY_SUFFIX = f"{OUTPUT_SUFFIX}.summary.csv"

FIGURE_SIZE = (4.6, 3.4)
# Keeps the last trace's endpoint off the right spine.
X_LIMIT_HEADROOM = 1.02


def draw_figure(
    trajectories: Sequence[tuple[str, list[float]]],
    summary: Sequence[Row],
) -> plt.Figure:
    """Render one PowerMH log's cumulative-time trajectories.

    The time axis is linear, unlike the empirical-time panels this figure is read beside. Those share one axis across
    six prefetch budgets that differ by orders of magnitude; a single baseline has nothing to accommodate, and on a
    linear axis its near-constant cost per transition reads as the straight line it is.

    Left untitled, like those panels: the run configuration is already in the filename. PowerMH logs written before the
    runner echoed its resolved configuration carry nothing to read a title from.
    """
    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    draw_cumulative_time_panel(
        ax,
        [cumulative for _, cumulative in trajectories],
        summary,
        color=POWER_COLOR,
    )
    ax.set_xlim(0, max(cumulative[-1] for _, cumulative in trajectories) * X_LIMIT_HEADROOM)
    ax.set_ylim(0, max(len(cumulative) for _, cumulative in trajectories))
    return figure


def render_log(log_path: Path) -> list[Path]:
    """Write one PowerMH log's figure and source data beside the log."""
    trajectories = power_mh_trace_times(log_path)
    summary = build_power_mh_time_summary_rows(trajectories)
    output_prefix = config_output_prefix(log_path)
    return [
        save_figure(
            draw_figure(trajectories, summary),
            output_path(output_prefix, FIGURE_SUFFIX),
            pad_inches=0.02,
        ),
        write_dict_rows(
            output_path(output_prefix, DATA_SUFFIX),
            build_power_mh_time_rows(log_path, trajectories),
        ),
        write_dict_rows(output_path(output_prefix, SUMMARY_SUFFIX), summary),
    ]


def discover_power_mh_logs(directory: Path) -> list[Path]:
    """Recursively discover PowerMH baseline logs."""
    log_paths = sorted(
        path for path in directory.rglob(POWER_MH_LOG_GLOB) if path.is_file()
    )
    if not log_paths:
        raise ValueError(f"No {POWER_MH_LOG_GLOB} files found under {directory}")
    return log_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the PowerMH baseline timing CLI."""
    parser = argparse.ArgumentParser(
        description="Plot cumulative model-call time for PowerMH baseline runs."
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Draw only this PowerMH log instead of searching --directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.log is not None:
        try:
            outputs = render_log(args.log.resolve())
        except (OSError, ValueError) as error:
            raise SystemExit(f"Failed {args.log}: {error}") from error
        for path in outputs:
            print(f"wrote {path}")
        return

    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"missing directory {directory}")
    try:
        log_paths = discover_power_mh_logs(directory)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # A run killed during engine startup leaves a log with no traces in it, and a directory of runs normally holds a
    # few. Sweeping past them is what lets the rest of the directory render; a named --log still fails loudly.
    failures: list[str] = []
    for log_path in log_paths:
        try:
            outputs = render_log(log_path)
        except (OSError, ValueError) as error:
            print(f"FAILED {log_path.name}: {error}")
            failures.append(log_path.name)
            continue
        for path in outputs:
            print(f"wrote {path}")

    if failures:
        raise SystemExit(
            f"failed {len(failures)}/{len(log_paths)} log(s): {', '.join(failures)}"
        )


if __name__ == "__main__":
    main()

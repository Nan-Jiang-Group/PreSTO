"""Parse target-model-call traces and build the tables the figures plot.

One trace is one benchmark sample's generation block: the MH steps it ran and the batched target-model calls those steps
cost. The drawing CLI turns these into cumulative trajectories, so everything from reading a log to the summary rows
lives here, and the plotting file stays about figures.

This library is not run directly; invoke a sibling ``draw_*.py`` CLI instead.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev

from case_studies.extract.run_naming import (
    LOG_GLOB,
    RANK_TOKEN_PREFIX,
    batch_size_in,
    canonical_rank,
    dataset_model_prefix,
    excluded_rank_log,
    prefetch_budget_token,
    rank_of,
    rename_budget_token,
    strip_backend_suffix,
    strip_batch_size,
    strip_tokens,
)

Row = dict[str, object]
TraceKey = tuple[str, str]

SAMPLE_PATTERN = re.compile(r"^IDX\s+(?P<sample>\d+)\b")
# The PowerMH baseline runner prints no IDX banner. Its sampler instead reports each finished prompt, so a sample is
# delimited by its own end rather than by the next one's start, and the index names the sample that just closed.
POWER_MH_SAMPLE_END_PATTERN = re.compile(
    r"mcmc_power_sampler took [\d.]+ seconds for (?P<sample>\d+)-th prompts"
)
POWER_MH_LOG_GLOB = "*.powerMH.*.log"
POWER_MH_LOG_MARKER = ".powerMH."


def is_power_mh_log(log_path: Path) -> bool:
    """Whether a log came from the PowerMH baseline rather than a subtree run."""
    return POWER_MH_LOG_MARKER in log_path.name
# Anchored at end of line so the samplers' "block [i/N] took ... seconds" timing line is not mistaken for the
# announcement that opens the block.
BLOCK_PATTERN = re.compile(r"block \[(?P<block>\d+)/(?P<blocks>\d+)\]\s*$")
CALL_PATTERN = re.compile(
    r"MH step (?P<step>\d+)/(?P<steps>\d+) took "
    r"(?P<seconds>\d+(?:\.\d+)?) seconds"
)
BENCHMARK_NAME_PATTERN = re.compile(
    r"^\s*benchmark_name:\s*['\"](?P<value>[^'\"]+)['\"]\s*$"
)
MODEL_STR_PATTERN = re.compile(r"^\s*model_str:\s*['\"](?P<value>[^'\"]+)['\"]\s*$")
COMPLETION_MARKER = ">>> completion:"

TRAJECTORY_COLUMNS = (
    "source_log",
    "rank",
    "trace_id",
    "sample_idx",
    "block_idx",
    "trace_complete",
    "mh_iteration",
)


class NoModelCallTracesError(ValueError):
    """Raised when a log contains no parseable target-model-call traces."""


@dataclass(frozen=True)
class ModelCall:
    """One batched target-model call recorded by the sampler."""

    mh_step: int
    total_mh_steps: int
    duration_seconds: float


@dataclass
class ModelCallTrace:
    """Target-model calls for one benchmark sample and generation block."""

    sample_idx: int
    block_idx: int
    declared_blocks: int
    calls: list[ModelCall] = field(default_factory=list)
    complete: bool = False

    @property
    def trace_id(self) -> str:
        return f"sample-{self.sample_idx:03d}.block-{self.block_idx:02d}"

    @property
    def total_mh_steps(self) -> int | None:
        totals = {call.total_mh_steps for call in self.calls}
        if not totals:
            return None
        if len(totals) != 1:
            raise ValueError(
                f"Inconsistent MH-step totals in {self.trace_id}: {sorted(totals)}"
            )
        return totals.pop()

    @property
    def observed_mh_steps(self) -> int:
        """Return the defensible prefix length for a complete or open trace."""
        total_mh_steps = self.total_mh_steps
        if total_mh_steps is None:
            return 0
        if self.complete:
            return total_mh_steps
        return min(max(call.mh_step for call in self.calls) + 1, total_mh_steps)

    def validate(self, source_log: str) -> None:
        """Check event order and configured MH-step bounds for one trace."""
        total_mh_steps = self.total_mh_steps
        if total_mh_steps is None:
            raise ValueError(f"Trace {source_log}:{self.trace_id} has no model calls")
        call_steps = [call.mh_step for call in self.calls]
        if call_steps != sorted(set(call_steps)):
            raise ValueError(
                f"Non-increasing or duplicate model-call steps in "
                f"{source_log}:{self.trace_id}: {call_steps}"
            )
        if call_steps[0] < 0 or call_steps[-1] >= total_mh_steps:
            raise ValueError(
                f"Model-call step outside [0, {total_mh_steps}) in "
                f"{source_log}:{self.trace_id}"
            )
        if any(call.duration_seconds <= 0 for call in self.calls):
            raise ValueError(
                f"Non-positive model-call duration in {source_log}:{self.trace_id}"
            )

    def identity(self, source_log: str, rank_name: str) -> Row:
        """The columns every row derived from this trace repeats."""
        return {
            "source_log": source_log,
            "rank": rank_name,
            "trace_id": self.trace_id,
            "sample_idx": self.sample_idx,
            "block_idx": self.block_idx,
            "trace_complete": self.complete,
        }


@dataclass(frozen=True)
class ExperimentGroup:
    """Comparable rank-policy logs sharing every other run setting."""

    signature: str
    log_paths: tuple[Path, ...]
    output_prefix: Path

    @property
    def directory(self) -> Path:
        return self.log_paths[0].parent


@dataclass
class ExtractedLogs:
    """Parsed records and provenance for one experiment group."""

    log_paths: list[Path]
    data_log_paths: list[Path]
    empty_log_paths: list[Path]
    traces: list[ModelCallTrace]
    event_rows: list[Row]
    cumulative_rows: list[Row]


@dataclass
class AnalysisRows:
    """Selected trajectories and the source-data tables derived from them."""

    events: list[Row]
    cumulative: list[Row]
    summary: list[Row]
    empirical_time: list[Row]
    empirical_time_summary: list[Row]
    time_per_call: list[Row]
    time_per_call_summary: list[Row]
    transitions_per_call: list[Row]
    transitions_per_call_summary: list[Row]
    traces_per_rank: int


@dataclass(frozen=True)
class BatchComparison:
    """Compatible analyzed runs that differ only by prefetch budget."""

    directory: Path
    signature: str
    output_prefix: Path
    figure_title: str
    batch_analyses: tuple[tuple[int, AnalysisRows], ...]


def extract_model_call_traces(log_path: Path) -> list[ModelCallTrace]:
    """Parse sample/block traces and target-model-call events from one log."""
    traces: list[ModelCallTrace] = []
    current_sample: int | None = None
    current_trace: ModelCallTrace | None = None

    def close(trace: ModelCallTrace | None) -> None:
        if trace is not None:
            trace.complete = True
            traces.append(trace)

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if sample_match := SAMPLE_PATTERN.search(line):
                close(current_trace)
                current_trace = None
                current_sample = int(sample_match.group("sample"))
                continue

            if end_match := POWER_MH_SAMPLE_END_PATTERN.search(line):
                close(current_trace)
                current_trace = None
                # The line names the sample that just ended, so the next one is its successor. Trusting it over a
                # running count keeps the numbering right even if a sample produced no blocks at all.
                current_sample = int(end_match.group("sample")) + 1
                continue

            if block_match := BLOCK_PATTERN.search(line):
                if current_sample is None:
                    if not is_power_mh_log(log_path):
                        raise ValueError(
                            f"Block encountered before sample at "
                            f"{log_path}:{line_number}"
                        )
                    # A PowerMH log opens straight into its first block, with no banner ahead of it; that first sample
                    # is sample 0.
                    current_sample = 0
                close(current_trace)
                current_trace = ModelCallTrace(
                    sample_idx=current_sample,
                    block_idx=int(block_match.group("block")),
                    declared_blocks=int(block_match.group("blocks")),
                )
                continue

            if call_match := CALL_PATTERN.search(line):
                if current_trace is None:
                    raise ValueError(
                        "Model call encountered before block at "
                        f"{log_path}:{line_number}"
                    )
                current_trace.calls.append(
                    ModelCall(
                        mh_step=int(call_match.group("step")),
                        total_mh_steps=int(call_match.group("steps")),
                        duration_seconds=float(call_match.group("seconds")),
                    )
                )
                continue

            if line.startswith(COMPLETION_MARKER) and current_trace is not None:
                close(current_trace)
                current_trace = None

    # A log cut short at EOF leaves its last trace open rather than complete.
    if current_trace is not None:
        traces.append(current_trace)
    if not traces:
        raise NoModelCallTracesError(f"No target-model-call traces found in {log_path}")
    return traces


def discover_log_paths(directory: Path) -> list[Path]:
    """Recursively discover subtree-prefetch sampler logs.

    Excluded traversal rules are dropped here rather than at the panel list, so that they cannot reach the shared axis
    limits either. A log named directly on a CLI is still drawn; only discovery skips them.
    """
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
    log_paths = sorted(
        path
        for path in directory.rglob(LOG_GLOB)
        if path.is_file() and not excluded_rank_log(path)
    )
    if not log_paths:
        raise ValueError(f"No {LOG_GLOB} files found under {directory}")
    return log_paths


def experiment_signature(log_path: Path) -> str:
    """Remove only the traversal-rank token from a sampler log filename."""
    return strip_tokens(log_path.name, (RANK_TOKEN_PREFIX,))


def batch_size_tag(signature: str) -> str | None:
    """Return one normalized prefetch-budget output tag."""
    batch_size = batch_size_in(signature)
    return None if batch_size is None else prefetch_budget_token(batch_size)


# The output stem the MH-yield figures and the endpoint figures built from them share. Both pass it to
# discover_experiment_groups below, and a group only lines up with the files it names if they agree, so they read one
# constant.
MODEL_CALL_OUTPUT_SUFFIX = ".model-calls-step"


def discover_experiment_groups(
    directory: Path, output_suffix: str
) -> list[ExperimentGroup]:
    """Group recursive logs without pooling incompatible run settings."""
    grouped_paths: dict[tuple[Path, str], list[Path]] = {}
    for log_path in discover_log_paths(directory):
        key = log_path.parent, experiment_signature(log_path)
        grouped_paths.setdefault(key, []).append(log_path)

    provisional = [
        (
            parent,
            signature,
            tuple(sorted(paths)),
            # The budget names the comparison point within an experiment; the dataset and base model name the experiment
            # itself, which nothing else in a per-budget filename records.
            f"{dataset_model_prefix(signature)}."
            f"{batch_size_tag(signature) or 'all-rank'}",
        )
        for (parent, signature), paths in sorted(
            grouped_paths.items(), key=lambda item: (str(item[0][0]), item[0][1])
        )
    ]
    # A tag shared by two groups in one directory would collide on disk, so those groups fall back to their full run
    # configuration instead.
    tag_counts = Counter((parent, tag) for parent, _, _, tag in provisional)
    return [
        ExperimentGroup(
            signature=signature,
            log_paths=paths,
            output_prefix=parent
            / (
                f"{rename_budget_token(strip_backend_suffix(signature))}{output_suffix}"
                if tag_counts[(parent, tag)] > 1
                else f"{tag}{output_suffix}"
            ),
        )
        for parent, signature, paths, tag in provisional
    ]


def experiment_figure_title(log_path: Path) -> str:
    """Read the dataset and resolved base-model names from one run log."""
    found: dict[str, str] = {}
    patterns = {"benchmark": BENCHMARK_NAME_PATTERN, "model": MODEL_STR_PATTERN}
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            for name, pattern in patterns.items():
                if match := pattern.match(line):
                    found[name] = match.group("value")
            if len(found) == len(patterns):
                break
    if len(found) != len(patterns):
        raise ValueError(
            f"Could not read benchmark_name and model_str from {log_path}"
        )
    return f"{found['benchmark']}, {found['model'].rsplit('/', 1)[-1]}"


def build_batch_comparisons(
    completed: Sequence[tuple[ExperimentGroup, AnalysisRows]],
) -> list[BatchComparison]:
    """Group analyzed runs that differ only by prefetch budget."""
    grouped: dict[tuple[Path, str], list[tuple[int, AnalysisRows]]] = {}
    titles: dict[tuple[Path, str], set[str]] = {}
    for group, analysis in completed:
        batch_size = batch_size_in(group.signature)
        if batch_size is None:
            continue
        key = group.directory, strip_batch_size(group.signature)
        grouped.setdefault(key, []).append((batch_size, analysis))
        titles.setdefault(key, set()).add(
            experiment_figure_title(group.log_paths[0])
        )

    comparable = {
        key: entries for key, entries in grouped.items() if len(entries) > 1
    }
    groups_per_directory = Counter(directory for directory, _ in comparable)
    comparisons: list[BatchComparison] = []
    for (directory, signature), entries in sorted(
        comparable.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        figure_titles = titles[(directory, signature)]
        if len(figure_titles) != 1:
            raise ValueError(
                "Cross-batch comparison mixes dataset or base-LLM names: "
                f"{sorted(figure_titles)}"
            )
        # Pooling every budget leaves no budget to name the figure after, so it takes the dataset and base model the
        # comparison holds fixed. A directory holding several comparisons would collide on that, and falls back to the
        # full run configuration.
        tag = dataset_model_prefix(signature)
        if groups_per_directory[directory] > 1:
            tag = strip_backend_suffix(signature)
        comparisons.append(
            BatchComparison(
                directory=directory,
                signature=signature,
                output_prefix=directory / tag,
                figure_title=next(iter(figure_titles)),
                batch_analyses=tuple(sorted(entries)),
            )
        )
    return comparisons


def extract_rank_name(log_path: Path, known_ranks: Sequence[str]) -> str:
    """Extract and canonicalize the traversal-rank category."""
    try:
        rank_name = canonical_rank(rank_of(log_path))
    except ValueError as error:
        raise ValueError(f"Missing rank-<category> token in {log_path}") from error
    if rank_name not in known_ranks:
        raise ValueError(
            f"No shared color configured for rank {rank_name!r} in {log_path.name}"
        )
    return rank_name


def build_event_rows(
    traces: Sequence[ModelCallTrace], source_log: str, rank_name: str
) -> list[Row]:
    """Create one source-data row per logged target-model call."""
    rows: list[Row] = []
    for trace in traces:
        if not trace.calls:
            continue
        trace.validate(source_log)
        for call_number, call in enumerate(trace.calls, start=1):
            rows.append(
                {
                    "source_log": source_log,
                    "rank": rank_name,
                    "trace_id": trace.trace_id,
                    "sample_idx": trace.sample_idx,
                    "block_idx": trace.block_idx,
                    "declared_blocks": trace.declared_blocks,
                    "trace_complete": trace.complete,
                    "mh_step": call.mh_step,
                    "mh_iteration": call.mh_step + 1,
                    "total_mh_steps": call.total_mh_steps,
                    "call_number": call_number,
                    "duration_seconds": call.duration_seconds,
                }
            )
    return rows


def build_cumulative_rows(
    traces: Sequence[ModelCallTrace],
    source_log: str,
    rank_name: str,
    include_incomplete: bool = False,
) -> list[Row]:
    """Expand selected traces into cumulative target-model-call trajectories."""
    rows: list[Row] = []
    for trace in traces:
        if not trace.calls or not (trace.complete or include_incomplete):
            continue
        trace.validate(source_log)
        identity = trace.identity(source_log, rank_name)
        call_steps = [call.mh_step for call in trace.calls]
        for mh_iteration in range(trace.observed_mh_steps + 1):
            rows.append(
                {
                    **identity,
                    "mh_iteration": mh_iteration,
                    # Calls already made when this iteration begins; PowerMH spends exactly one call per step, which is
                    # the reference.
                    "subtree_cumulative_calls": sum(
                        step < mh_iteration for step in call_steps
                    ),
                    "power_mh_cumulative_calls": mh_iteration,
                }
            )
    return rows


def row_trace_key(row: Row) -> TraceKey:
    """Return the stable source-log and trace-id key for one record."""
    return str(row["source_log"]), str(row["trace_id"])


def group_rows(
    rows: Sequence[Row], key: Callable[[Row], object]
) -> dict[object, list[Row]]:
    """Group records by one key and sort each group by MH iteration."""
    grouped: dict[object, list[Row]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["mh_iteration"]))
    return grouped


def group_rows_by_trace(rows: Sequence[Row]) -> dict[TraceKey, list[Row]]:
    """Group records by trace and sort each trace by MH iteration."""
    return group_rows(rows, row_trace_key)


def group_rows_by_rank(rows: Sequence[Row]) -> dict[str, list[Row]]:
    """Group records by rank and sort by MH iteration."""
    return group_rows(rows, lambda row: str(row["rank"]))


def map_trace_ranks(rows_by_trace: dict[TraceKey, list[Row]]) -> dict[TraceKey, str]:
    """Validate and return the single rank category for every trace."""
    rank_by_trace: dict[TraceKey, str] = {}
    for trace_key, trace_rows in rows_by_trace.items():
        rank_names = {str(row["rank"]) for row in trace_rows}
        if len(rank_names) != 1:
            raise ValueError(
                f"Trace {trace_key} spans multiple ranks: {sorted(rank_names)}"
            )
        rank_by_trace[trace_key] = rank_names.pop()
    return rank_by_trace


def select_rank_traces(
    cumulative_rows: Sequence[Row], requested_traces_per_rank: int
) -> tuple[list[Row], int]:
    """Select a deterministic, equal-size trace set for every rank."""
    trace_keys_by_rank: dict[str, set[TraceKey]] = {}
    for row in cumulative_rows:
        trace_keys_by_rank.setdefault(str(row["rank"]), set()).add(row_trace_key(row))
    if not trace_keys_by_rank:
        raise ValueError("No complete model-call traces are available")

    # Every rank contributes the same number of traces, so the panels compare like with like; the smallest rank sets
    # that number.
    traces_per_rank = min(
        requested_traces_per_rank,
        min(len(trace_keys) for trace_keys in trace_keys_by_rank.values()),
    )
    selected_keys = {
        trace_key
        for trace_keys in trace_keys_by_rank.values()
        for trace_key in sorted(trace_keys)[:traces_per_rank]
    }
    return (
        [row for row in cumulative_rows if row_trace_key(row) in selected_keys],
        traces_per_rank,
    )


def mean_and_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return the arithmetic mean and sample standard deviation."""
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def _summarize_by_rank_iteration(
    rows: Sequence[Row], value_key: str, mean_key: str, sd_key: str
) -> list[Row]:
    """Reduce per-trace rows to a mean and sample SD per rank and iteration."""
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = str(row["rank"]), int(row["mh_iteration"])
        grouped.setdefault(key, []).append(float(row[value_key]))

    summary: list[Row] = []
    for (rank_name, mh_iteration), values in sorted(grouped.items()):
        mean_value, sd_value = mean_and_sample_sd(values)
        summary.append(
            {
                "rank": rank_name,
                "mh_iteration": mh_iteration,
                "trace_count": len(values),
                mean_key: mean_value,
                sd_key: sd_value,
            }
        )
    return summary


def build_summary_rows(cumulative_rows: Sequence[Row]) -> list[Row]:
    """Compute per-rank mean and sample SD at every MH iteration."""
    return [
        {**row, "power_mh_cumulative_calls": row["mh_iteration"]}
        for row in _summarize_by_rank_iteration(
            cumulative_rows,
            "subtree_cumulative_calls",
            "subtree_mean_calls",
            "subtree_std_calls",
        )
    ]


def build_empirical_time_summary_rows(empirical_time_rows: Sequence[Row]) -> list[Row]:
    """Summarize cumulative empirical time by rank and MH iteration."""
    return _summarize_by_rank_iteration(
        empirical_time_rows,
        "cumulative_empirical_seconds",
        "mean_cumulative_seconds",
        "std_cumulative_seconds",
    )


def build_transitions_per_call_rows(
    cumulative_rows: Sequence[Row],
) -> list[Row]:
    """Build running realized MH transitions per proposal-model call.

    The zero-call origin is omitted because its ratio is undefined.
    """
    rows: list[Row] = []
    for row in cumulative_rows:
        cumulative_calls = float(row["subtree_cumulative_calls"])
        mh_iteration = int(row["mh_iteration"])
        if cumulative_calls == 0.0:
            if mh_iteration != 0:
                raise ValueError(
                    "A positive MH iteration cannot have zero cumulative "
                    f"proposal-model calls: {row_trace_key(row)}"
                )
            continue
        rows.append(
            {
                **{column: row[column] for column in TRAJECTORY_COLUMNS},
                "mh_transitions_per_call": mh_iteration / cumulative_calls,
            }
        )
    return rows


def build_transitions_per_call_summary_rows(
    transitions_per_call_rows: Sequence[Row],
) -> list[Row]:
    """Summarize MH transitions per call by rank and MH iteration."""
    return _summarize_by_rank_iteration(
        transitions_per_call_rows,
        "mh_transitions_per_call",
        "mean_mh_transitions_per_call",
        "std_mh_transitions_per_call",
    )


def build_time_per_call_rows(
    cumulative_rows: Sequence[Row],
    empirical_time_rows: Sequence[Row],
) -> list[Row]:
    """Build the running empirical seconds per proposal-model call.

    At each realized MH transition, the non-cumulative value is the direct cumulative MH-step time divided by the
    cumulative number of proposal-model calls. The zero-call origin is omitted because its ratio is undefined.
    """
    calls_by_trace_iteration = {
        (row_trace_key(row), int(row["mh_iteration"])): row
        for row in cumulative_rows
    }
    if len(calls_by_trace_iteration) != len(cumulative_rows):
        raise ValueError("Duplicate cumulative call rows for one trace iteration")

    rows: list[Row] = []
    for time_row in empirical_time_rows:
        key = row_trace_key(time_row), int(time_row["mh_iteration"])
        call_row = calls_by_trace_iteration.get(key)
        if call_row is None:
            raise ValueError(
                "Empirical-time row has no matching cumulative-call row: "
                f"{key}"
            )
        cumulative_calls = float(call_row["subtree_cumulative_calls"])
        cumulative_seconds = float(
            time_row["cumulative_empirical_seconds"]
        )
        if cumulative_calls == 0.0:
            if cumulative_seconds != 0.0:
                raise ValueError(
                    "Positive empirical time cannot precede the first "
                    f"proposal-model call: {key}"
                )
            continue
        rows.append(
            {
                **{
                    column: time_row[column]
                    for column in TRAJECTORY_COLUMNS
                },
                "empirical_seconds_per_call": (
                    cumulative_seconds / cumulative_calls
                ),
            }
        )
    return rows


def build_time_per_call_summary_rows(
    time_per_call_rows: Sequence[Row],
) -> list[Row]:
    """Summarize empirical seconds per call by rank and MH iteration."""
    return _summarize_by_rank_iteration(
        time_per_call_rows,
        "empirical_seconds_per_call",
        "mean_empirical_seconds_per_call",
        "std_empirical_seconds_per_call",
    )


def power_mh_trace_times(log_path: Path) -> list[tuple[str, list[float]]]:
    """Return each complete PowerMH trace's cumulative model-call time.

    PowerMH spends one target-model call on every MH step, so a trace's trajectory is just the running sum of its step
    durations and needs no separate iteration axis: position ``i`` holds the time to reach iteration ``i + 1``.
    """
    if not is_power_mh_log(log_path):
        raise ValueError(f"Not a PowerMH log: {log_path}")

    trajectories: list[tuple[str, list[float]]] = []
    for trace in extract_model_call_traces(log_path):
        if not trace.complete:
            continue
        elapsed = 0.0
        cumulative: list[float] = []
        for call in trace.calls:
            elapsed += call.duration_seconds
            cumulative.append(elapsed)
        if cumulative:
            trajectories.append((trace.trace_id, cumulative))

    if not trajectories:
        raise NoModelCallTracesError(
            f"No complete PowerMH traces to time in {log_path}"
        )
    return trajectories


def _power_mh_times_by_iteration(
    trajectories: Sequence[tuple[str, list[float]]],
) -> dict[int, list[float]]:
    """Collect cumulative times per MH iteration across PowerMH traces."""
    by_iteration: dict[int, list[float]] = {}
    for _, cumulative in trajectories:
        for iteration, seconds in enumerate(cumulative, start=1):
            by_iteration.setdefault(iteration, []).append(seconds)
    return by_iteration


def build_power_mh_time_summary_rows(
    trajectories: Sequence[tuple[str, list[float]]],
) -> list[Row]:
    """Summarize PowerMH cumulative time by MH iteration.

    Iterations are averaged over the traces that actually reached them, which keeps a run cut short at EOF from dragging
    the tail down.
    """
    summary: list[Row] = []
    for iteration, values in sorted(_power_mh_times_by_iteration(trajectories).items()):
        mean_value, sd_value = mean_and_sample_sd(values)
        summary.append(
            {
                "mh_iteration": iteration,
                "trace_count": len(values),
                "mean_cumulative_seconds": mean_value,
                "std_cumulative_seconds": sd_value,
            }
        )
    return summary


def power_mh_time_per_call_trajectories(
    trajectories: Sequence[tuple[str, list[float]]],
) -> list[tuple[str, list[float]]]:
    """Convert PowerMH cumulative times to running seconds per call."""
    return [
        (
            trace_id,
            [
                cumulative_seconds / iteration
                for iteration, cumulative_seconds in enumerate(
                    cumulative, start=1
                )
            ],
        )
        for trace_id, cumulative in trajectories
    ]


def build_power_mh_time_per_call_summary_rows(
    trajectories: Sequence[tuple[str, list[float]]],
) -> list[Row]:
    """Summarize PowerMH running seconds per call by MH iteration."""
    summary: list[Row] = []
    time_per_call = power_mh_time_per_call_trajectories(trajectories)
    for iteration, values in sorted(
        _power_mh_times_by_iteration(time_per_call).items()
    ):
        mean_value, sd_value = mean_and_sample_sd(values)
        summary.append(
            {
                "mh_iteration": iteration,
                "trace_count": len(values),
                "mean_empirical_seconds_per_call": mean_value,
                "std_empirical_seconds_per_call": sd_value,
            }
        )
    return summary


def build_power_mh_time_rows(
    log_path: Path,
    trajectories: Sequence[tuple[str, list[float]]],
) -> list[Row]:
    """Flatten PowerMH trajectories into one row per trace and MH iteration.

    The column names match the subtree empirical-time rows, so the standalone figure's table and the comparison figure's
    table can be read the same way and, in the comparison, concatenated under one ``method`` column.
    """
    return [
        {
            "source_log": log_path.name,
            "trace_id": trace_id,
            "mh_iteration": iteration,
            "cumulative_empirical_seconds": seconds,
        }
        for trace_id, cumulative in trajectories
        for iteration, seconds in enumerate(cumulative, start=1)
    ]


def power_mh_time_reference(log_path: Path) -> list[tuple[int, float]]:
    """Mean cumulative model-call time per MH iteration in a PowerMH log.

    Returns (iteration, mean cumulative seconds) pairs, iteration ascending and starting at 0 seconds, ready to plot
    against the same axes as the subtree trajectories. The origin is prepended so the line reaches the axis; the summary
    rows leave it out, having no measurement there to report.
    """
    totals = _power_mh_times_by_iteration(power_mh_trace_times(log_path))
    return [(0, 0.0)] + [
        (iteration, fmean(totals[iteration])) for iteration in sorted(totals)
    ]


def build_empirical_time_rows(
    event_rows: Sequence[Row], cumulative_rows: Sequence[Row]
) -> list[Row]:
    """Build cumulative empirical call time for every selected trace."""
    selected_trace_keys = {row_trace_key(row) for row in cumulative_rows}
    events_by_trace: dict[TraceKey, list[tuple[int, float]]] = {}
    for row in event_rows:
        trace_key = row_trace_key(row)
        if trace_key in selected_trace_keys:
            events_by_trace.setdefault(trace_key, []).append(
                (int(row["mh_step"]), float(row["duration_seconds"]))
            )

    if missing_trace_keys := selected_trace_keys - events_by_trace.keys():
        raise ValueError(
            "Selected traces have no empirical timing events: "
            f"{sorted(missing_trace_keys)}"
        )

    rows: list[Row] = []
    for row in cumulative_rows:
        mh_iteration = int(row["mh_iteration"])
        rows.append(
            {
                **{column: row[column] for column in TRAJECTORY_COLUMNS},
                "cumulative_empirical_seconds": sum(
                    seconds
                    for mh_step, seconds in events_by_trace[row_trace_key(row)]
                    if mh_step < mh_iteration
                ),
            }
        )
    return rows


def extract_logs(
    log_paths: Sequence[Path],
    include_incomplete: bool,
    known_ranks: Sequence[str],
) -> ExtractedLogs:
    """Parse one comparable group and collect event and trajectory records."""
    data_log_paths: list[Path] = []
    empty_log_paths: list[Path] = []
    traces: list[ModelCallTrace] = []
    event_rows: list[Row] = []
    cumulative_rows: list[Row] = []
    observed_ranks: set[str] = set()

    for log_path in log_paths:
        rank_name = extract_rank_name(log_path, known_ranks)
        if rank_name in observed_ranks:
            raise ValueError(
                f"More than one log for canonical rank {rank_name!r} in "
                f"{log_path.parent}"
            )
        observed_ranks.add(rank_name)
        try:
            log_traces = extract_model_call_traces(log_path)
        except NoModelCallTracesError:
            empty_log_paths.append(log_path)
            continue
        data_log_paths.append(log_path)
        source_log = log_path.name
        traces.extend(log_traces)
        event_rows.extend(build_event_rows(log_traces, source_log, rank_name))
        cumulative_rows.extend(
            build_cumulative_rows(
                log_traces, source_log, rank_name, include_incomplete=include_incomplete
            )
        )

    if not event_rows:
        raise ValueError("No target-model-call events were extracted")
    if not cumulative_rows:
        trace_scope = "complete or incomplete" if include_incomplete else "complete"
        raise ValueError(f"No {trace_scope} target-model-call traces were extracted")
    missing_ranks = sorted(
        {str(row["rank"]) for row in event_rows}
        - {str(row["rank"]) for row in cumulative_rows}
    )
    if missing_ranks:
        trace_scope = "complete or incomplete" if include_incomplete else "complete"
        raise ValueError(
            f"No {trace_scope} trace for rank(s): " + ", ".join(missing_ranks)
        )

    return ExtractedLogs(
        log_paths=list(log_paths),
        data_log_paths=data_log_paths,
        empty_log_paths=empty_log_paths,
        traces=traces,
        event_rows=event_rows,
        cumulative_rows=cumulative_rows,
    )


def build_analysis_rows(
    extracted: ExtractedLogs, requested_traces_per_rank: int
) -> AnalysisRows:
    """Select equal-size rank samples and build all derived tables."""
    cumulative_rows, traces_per_rank = select_rank_traces(
        extracted.cumulative_rows, requested_traces_per_rank
    )
    empirical_time_rows = build_empirical_time_rows(
        extracted.event_rows, cumulative_rows
    )
    time_per_call_rows = build_time_per_call_rows(
        cumulative_rows, empirical_time_rows
    )
    transitions_per_call_rows = build_transitions_per_call_rows(
        cumulative_rows
    )
    return AnalysisRows(
        events=extracted.event_rows,
        cumulative=cumulative_rows,
        summary=build_summary_rows(cumulative_rows),
        empirical_time=empirical_time_rows,
        empirical_time_summary=build_empirical_time_summary_rows(empirical_time_rows),
        time_per_call=time_per_call_rows,
        time_per_call_summary=build_time_per_call_summary_rows(
            time_per_call_rows
        ),
        transitions_per_call=transitions_per_call_rows,
        transitions_per_call_summary=(
            build_transitions_per_call_summary_rows(
                transitions_per_call_rows
            )
        ),
        traces_per_rank=traces_per_rank,
    )


def subtree_mh_trace_times(
    log_path: Path,
    requested_traces_per_rank: int,
    known_ranks: Sequence[str],
    *,
    include_incomplete: bool = False,
) -> tuple[list[tuple[str, list[float]]], AnalysisRows]:
    """Return direct cumulative MH-step times for one subtree-prefetch log.

    Each trajectory sums the durations from the log's individual ``MH step ... took ... seconds`` records. The
    zero-second origin stays in the summary table but is omitted from the per-trace lists, whose position ``i``
    therefore remains the time to reach MH iteration ``i + 1``. An open trace from a crashed or interrupted run is
    included only when explicitly requested and remains marked by ``trace_complete=False`` in the rows.
    """
    if is_power_mh_log(log_path):
        raise ValueError(f"Not a subtree-prefetch log: {log_path}")

    extracted = extract_logs(
        [log_path],
        include_incomplete=include_incomplete,
        known_ranks=known_ranks,
    )
    analysis = build_analysis_rows(extracted, requested_traces_per_rank)
    trajectories: list[tuple[str, list[float]]] = []
    for (_, trace_id), trace_rows in sorted(
        group_rows_by_trace(analysis.empirical_time).items()
    ):
        cumulative = [
            float(row["cumulative_empirical_seconds"])
            for row in trace_rows
            if int(row["mh_iteration"]) > 0
        ]
        if cumulative:
            trajectories.append((trace_id, cumulative))

    if not trajectories:
        raise NoModelCallTracesError(
            f"No subtree-prefetch MH-step timing trajectories found in {log_path}"
        )
    return trajectories, analysis

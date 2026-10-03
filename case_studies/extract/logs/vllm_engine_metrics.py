"""Parse periodic vLLM engine metrics from subtree-prefetch logs.

This library is not run directly. To parse a log, write its source-data CSV,
and draw the corresponding figure, run ``draw_vllm_engine_metrics.py``:

    src/.venv/bin/python \
        case_studies/draw/draw_vllm_engine_metrics.py \
        --log /absolute/path/to/run.subtreePrefetch.vllm.log
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

NUMBER_PATTERN = r"\d+(?:\.\d+)?"
ENGINE_METRIC_PATTERN = re.compile(
    r"\bINFO\s+"
    r"(?P<month>\d{2})-(?P<day>\d{2})\s+"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\s+"
    r"\[loggers\.py:\d+\]\s+Engine\s+[^:]+:\s+"
    rf"Avg prompt throughput:\s+(?P<prompt>{NUMBER_PATTERN})\s+tokens/s,\s+"
    rf"Avg generation throughput:\s+(?P<generation>{NUMBER_PATTERN})\s+tokens/s,\s+"
    r"Running:\s+(?P<running>\d+)\s+reqs,\s+"
    r"Waiting:\s+(?P<waiting>\d+)\s+reqs,\s+"
    rf"GPU KV cache usage:\s+(?P<kv_cache>{NUMBER_PATTERN})%,\s+"
    rf"Prefix cache hit rate:\s+(?P<prefix_cache>{NUMBER_PATTERN})%"
)
DATED_DIRECTORY_PATTERN = re.compile(r"(?P<year>\d{4})-\d{2}-\d{2}")


@dataclass(frozen=True)
class EngineMetricSample:
    """One periodic engine-statistics record emitted by vLLM."""

    source_line: int
    timestamp: datetime
    prompt_throughput_tokens_s: float
    generation_throughput_tokens_s: float
    running_requests: int
    waiting_requests: int
    gpu_kv_cache_usage_percent: float
    prefix_cache_hit_rate_percent: float


def infer_log_year(log_path: Path) -> int:
    """Infer the timestamp year from the nearest date-stamped directory."""
    for part in reversed(log_path.parts):
        match = DATED_DIRECTORY_PATTERN.fullmatch(part)
        if match is not None:
            return int(match.group("year"))
    return 2000


def _validate_percentage(value: float, field: str, path: Path, line: int) -> None:
    """Reject malformed cache percentages before they reach a figure."""
    if not 0.0 <= value <= 100.0:
        raise ValueError(
            f"{field} outside [0, 100] at {path}:{line}: {value:g}%"
        )


def parse_engine_metrics(log_path: Path) -> list[EngineMetricSample]:
    """Parse timestamped throughput, request, and cache metrics from one log."""
    samples: list[EngineMetricSample] = []
    timestamp_year = infer_log_year(log_path)
    previous_timestamp: datetime | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            match = ENGINE_METRIC_PATTERN.search(line)
            if match is None:
                continue

            month = int(match.group("month"))
            day = int(match.group("day"))
            timestamp = datetime(
                timestamp_year,
                month,
                day,
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second")),
            )
            if (
                previous_timestamp is not None
                and previous_timestamp.month == 12
                and month == 1
            ):
                timestamp_year += 1
                timestamp = timestamp.replace(year=timestamp_year)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    "Engine-metric timestamps are not monotonic at "
                    f"{log_path}:{line_number}: {timestamp} follows "
                    f"{previous_timestamp}"
                )

            kv_cache = float(match.group("kv_cache"))
            prefix_cache = float(match.group("prefix_cache"))
            _validate_percentage(
                kv_cache, "GPU KV cache usage", log_path, line_number
            )
            _validate_percentage(
                prefix_cache, "Prefix cache hit rate", log_path, line_number
            )
            samples.append(
                EngineMetricSample(
                    source_line=line_number,
                    timestamp=timestamp,
                    prompt_throughput_tokens_s=float(match.group("prompt")),
                    generation_throughput_tokens_s=float(
                        match.group("generation")
                    ),
                    running_requests=int(match.group("running")),
                    waiting_requests=int(match.group("waiting")),
                    gpu_kv_cache_usage_percent=kv_cache,
                    prefix_cache_hit_rate_percent=prefix_cache,
                )
            )
            previous_timestamp = timestamp

    if not samples:
        raise ValueError(f"No vLLM engine-metric records found in {log_path}")
    return samples


def elapsed_seconds(samples: list[EngineMetricSample]) -> list[float]:
    """Return actual elapsed seconds relative to the first logger sample."""
    if not samples:
        raise ValueError("Cannot compute elapsed time for an empty sample list")
    start = samples[0].timestamp
    return [(sample.timestamp - start).total_seconds() for sample in samples]

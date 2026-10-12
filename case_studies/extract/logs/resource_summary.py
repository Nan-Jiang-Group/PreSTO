"""Parse and group run-level ``*.resources.json`` measurements.

This library is not run directly. To collect a directory and draw its
comparison figure, run:

    src/.venv/bin/python \
        case_studies/draw/draw_resource_usage.py \
        --directory /absolute/path/to/vllm/results
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from case_studies.extract.run_naming import (
    BATCH_SIZE_TOKEN_PREFIXES,
    dataset_model_prefix,
    is_excluded_rank,
    strip_tokens,
)

RESOURCE_GLOB = "*resources.json"
RESOURCE_SUFFIX = ".resources.json"


@dataclass(frozen=True)
class ResourceSummary:
    """Validated resource measurements for one completed vLLM run."""

    source_path: Path
    run: str
    rank_fn: str
    prefetch_budget: int
    num_blocks: int
    mcmc_steps: int
    prefix_cache: bool
    enabled: bool
    peak_gpu_memory_gib: float | None
    device_peak_gpu_memory_gib: float | None
    baseline_gpu_memory_gib: float | None
    gpu_memory_utilization: float | None
    num_gpu_blocks: int | None
    peak_kv_cache_occupancy: float | None
    prefix_cache_hit_rate: float | None
    prefix_cache_queried_tokens: int | None
    prefix_cache_hit_tokens: int | None
    engine_steps_sampled: int | None
    kv_source: str | None
    prefix_source: str | None
    nvml_error: str | None


@dataclass(frozen=True)
class ResourceGroup:
    """Comparable resource summaries sharing every non-rank, non-batch setting."""

    signature: str
    summaries: tuple[ResourceSummary, ...]
    output_prefix: Path

    @property
    def directory(self) -> Path:
        return self.summaries[0].source_path.parent


class _Reader:
    """Read one validated JSON payload, reporting the file on every failure."""

    def __init__(self, payload: dict[str, object], path: Path) -> None:
        self._payload = payload
        self._path = path

    def _typed(
        self,
        key: str,
        kind: type | tuple[type, ...],
        description: str,
        *,
        optional: bool,
    ) -> object | None:
        value = self._payload.get(key)
        if value is None and optional:
            return None
        # ``bool`` is a subclass of ``int``, and a Boolean where a count belongs is a malformed probe result rather than
        # a value worth coercing.
        if not isinstance(value, kind) or (
            kind is not bool and isinstance(value, bool)
        ):
            suffix = " or null" if optional else ""
            raise ValueError(f"Expected {description}{suffix} {key!r} in {self._path}")
        return value

    def string(self, key: str) -> str:
        value = self._typed(key, str, "non-empty string", optional=False)
        if not value:
            raise ValueError(f"Expected non-empty string {key!r} in {self._path}")
        return str(value)

    def integer(self, key: str) -> int:
        return int(self._typed(key, int, "integer", optional=False))

    def integer_either(self, key: str, legacy_key: str) -> int:
        """Read an integer under ``key``, falling back to a former spelling.

        Probes written before a field was renamed still carry the old key, and those runs stay readable rather than
        being reprocessed.
        """
        if key not in self._payload and legacy_key in self._payload:
            return self.integer(legacy_key)
        return self.integer(key)

    def boolean(self, key: str) -> bool:
        return bool(self._typed(key, bool, "Boolean", optional=False))

    def optional_float(self, key: str) -> float | None:
        value = self._typed(key, (int, float), "numeric", optional=True)
        return None if value is None else float(value)

    def optional_integer(self, key: str) -> int | None:
        value = self._typed(key, int, "integer", optional=True)
        return None if value is None else int(value)

    def optional_string(self, key: str) -> str | None:
        value = self._typed(key, str, "string", optional=True)
        return None if value is None else str(value)

    def fraction(self, key: str) -> float | None:
        """Read an optional value that must be a probability if present."""
        value = self.optional_float(key)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} outside [0, 1] in {self._path}: {value:g}")
        return value

    def non_negative_integer(self, key: str) -> int | None:
        value = self.optional_integer(key)
        if value is not None and value < 0:
            raise ValueError(f"Negative {key} in {self._path}")
        return value


def parse_resource_summary(path: Path) -> ResourceSummary:
    """Read one resource summary and validate its plotted measurements."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    reader = _Reader(payload, path)
    run = reader.string("run")
    rank_fn = reader.string("rank_fn")
    prefetch_budget = reader.integer_either("prefetch_budget", "subtree_batch_size")
    if path.name != f"{run}{RESOURCE_SUFFIX}":
        raise ValueError(
            f"Run name {run!r} does not match resource filename {path.name!r}"
        )
    if f"rank-{rank_fn}" not in run:
        raise ValueError(f"rank_fn={rank_fn!r} does not match run name in {path}")
    if not any(
        f"{prefix}{prefetch_budget}" in run for prefix in BATCH_SIZE_TOKEN_PREFIXES
    ):
        raise ValueError(f"prefetch_budget={prefetch_budget} does not match {path}")

    peak_gpu_memory = reader.optional_float("peak_gpu_memory_gib")
    if peak_gpu_memory is not None and peak_gpu_memory < 0.0:
        raise ValueError(f"Negative peak_gpu_memory_gib in {path}")
    prefix_hit_rate = reader.fraction("prefix_cache_hit_rate")
    queried_tokens = reader.non_negative_integer("prefix_cache_queried_tokens")
    hit_tokens = reader.non_negative_integer("prefix_cache_hit_tokens")
    if (
        queried_tokens is not None
        and hit_tokens is not None
        and hit_tokens > queried_tokens
    ):
        raise ValueError(f"Prefix-cache hits exceed queries in {path}")
    if (
        prefix_hit_rate is not None
        and queried_tokens
        and hit_tokens is not None
        and not math.isclose(
            prefix_hit_rate,
            hit_tokens / queried_tokens,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"Prefix-cache hit rate does not match token counts in {path}")

    return ResourceSummary(
        source_path=path,
        run=run,
        rank_fn=rank_fn,
        prefetch_budget=prefetch_budget,
        num_blocks=reader.integer("num_blocks"),
        mcmc_steps=reader.integer("mcmc_steps"),
        prefix_cache=reader.boolean("prefix_cache"),
        enabled=reader.boolean("enabled"),
        peak_gpu_memory_gib=peak_gpu_memory,
        device_peak_gpu_memory_gib=reader.optional_float(
            "device_peak_gpu_memory_gib"
        ),
        baseline_gpu_memory_gib=reader.optional_float("baseline_gpu_memory_gib"),
        gpu_memory_utilization=reader.fraction("gpu_memory_utilization"),
        num_gpu_blocks=reader.optional_integer("num_gpu_blocks"),
        peak_kv_cache_occupancy=reader.fraction("peak_kv_cache_occupancy"),
        prefix_cache_hit_rate=prefix_hit_rate,
        prefix_cache_queried_tokens=queried_tokens,
        prefix_cache_hit_tokens=hit_tokens,
        engine_steps_sampled=reader.optional_integer("engine_steps_sampled"),
        kv_source=reader.optional_string("kv_source"),
        prefix_source=reader.optional_string("prefix_source"),
        nvml_error=reader.optional_string("nvml_error"),
    )


def discover_resource_groups(directory: Path) -> list[ResourceGroup]:
    """Recursively collect resource files without pooling incompatible runs."""
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
    paths = sorted(path for path in directory.rglob(RESOURCE_GLOB) if path.is_file())
    if not paths:
        raise ValueError(f"No {RESOURCE_GLOB} files found under {directory}")

    grouped: dict[tuple[Path, str], list[ResourceSummary]] = defaultdict(list)
    for path in paths:
        # Sequential baselines have no rank or prefetch budget. Their resources
        # are compared by draw_combined_cache_metrics, not this subtree sweep.
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("method") in {"power_mh", "entropycut_power_mh"}:
            continue
        summary = parse_resource_summary(path)
        if is_excluded_rank(summary.rank_fn):
            continue
        grouped[(path.parent, strip_tokens(summary.run))].append(summary)

    # A directory holding one comparison is named for the dataset and base model it pools every rank and budget over;
    # only a directory with several needs each figure disambiguated by its full configuration.
    groups_per_directory = Counter(parent for parent, _ in grouped)
    return [
        ResourceGroup(
            signature=signature,
            summaries=tuple(
                sorted(
                    summaries,
                    key=lambda item: (item.prefetch_budget, item.rank_fn),
                )
            ),
            output_prefix=parent
            / (
                f"{dataset_model_prefix(signature)}.resource-usage"
                if groups_per_directory[parent] == 1
                else f"{signature}.resource-usage"
            ),
        )
        for (parent, signature), summaries in sorted(
            grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])
        )
    ]

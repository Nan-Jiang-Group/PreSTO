"""Build measured EntropyCut timing, workload, and engine comparisons.

Run this through the adjacent draw_entropycut_analysis.py CLI. It uses all
completed sample traces, validates result/resource counts, and writes the data
behind each comparison beside its PDF. No GPU or model loading is required.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.extract.logs.model_call_traces import extract_model_call_traces
from case_studies.extract.logs.vllm_engine_metrics import parse_engine_metrics
from case_studies.extract.run_naming import (
    config_output_prefix,
    model_of,
    prefetch_budget_of,
    rank_label,
    rank_of,
    sort_rank_names,
)
from case_studies.plot_config import (
    BLUE,
    DARK_GRAY,
    GREEN,
    POWER_METHOD_LABEL,
    SUBTREE_METHOD_LABEL,
    TRANSITION_RANK_COLORS,
    apply_plot_style,
)

SAMPLER_TIME = re.compile(
    r"(?:mcmc_power_sampler|subtree_prefetching_sampling) took ([\d.]+) seconds for (\d+)-th prompts"
)
POWER_BREAKDOWN = re.compile(r"step \d+ breakdown: build ([\d.]+) sec, generate ([\d.]+) sec, score ([\d.]+) sec")
SUBTREE_BREAKDOWN = re.compile(r"proposal call: \d+ requests, build ([\d.]+), generate ([\d.]+), score ([\d.]+) seconds")
EVICTIONS = re.compile(r"KV cache eviction rate: ([\d.]+)% \((\d+)/(\d+) blocks\)")
CUT_DRAWS = re.compile(r"drawn cut indices \(batch, pop order\): (\[[^\]]*\]); sampling probabilities \(same order\): (\[[^\]]*\])")


def read_run(log: Path) -> tuple[dict, list[dict], list[dict]]:
    """Read one complete run and return run, sample, and engine measurement rows."""
    prefix = config_output_prefix(log)
    text = log.read_text(encoding="utf-8")
    traces = extract_model_call_traces(log)
    resource_path = Path(str(prefix) + ".resources.json")
    resources = json.loads(resource_path.read_text())
    with Path(str(prefix) + ".csv").open(newline="") as handle:
        results = list(csv.DictReader(handle))
    sampler_times = {int(index): float(seconds) for seconds, index in SAMPLER_TIME.findall(text)}
    sample_ids = sorted({trace.sample_idx for trace in traces})
    if len(sample_ids) != len(results) or set(sample_ids) != set(sampler_times):
        raise ValueError(f"Trace/result/sampler counts disagree: {log}")
    baseline = ".powerMH." in log.name
    method = POWER_METHOD_LABEL if baseline else SUBTREE_METHOD_LABEL
    rank = "" if baseline else rank_of(log)
    samples = []
    for index in sample_ids:
        selected = [trace for trace in traces if trace.sample_idx == index]
        if not all(trace.complete for trace in selected):
            raise ValueError(f"Incomplete sample {index}: {log}")
        samples.append({
            "source_log": str(log), "method": method, "rank": rank,
            "prefetch_budget": resources.get("prefetch_budget", 1), "sample_idx": index,
            "sampler_seconds": sampler_times[index],
            "mh_seconds": sum(call.duration_seconds for trace in selected for call in trace.calls),
            "mh_calls": sum(len(trace.calls) for trace in selected),
            "mh_transitions": sum(trace.total_mh_steps for trace in selected),
        })
    calls = sum(row["mh_calls"] for row in samples)
    breakdown = (POWER_BREAKDOWN if baseline else SUBTREE_BREAKDOWN).findall(text)
    if len(breakdown) != calls:
        raise ValueError(f"Expected {calls} call breakdowns, found {len(breakdown)}: {log}")
    queried, hit = resources["prefix_cache_queried_tokens"], resources["prefix_cache_hit_tokens"]
    allocated, evicted = resources["kv_cache_allocated_blocks"], resources["kv_cache_evicted_blocks"]
    if not 0 <= hit <= queried or not math.isclose(hit / queried, resources["prefix_cache_hit_rate"]):
        raise ValueError(f"Inconsistent prefix-cache counts: {log}")
    if not 0 <= evicted <= allocated or not math.isclose(evicted / allocated, resources["kv_cache_eviction_rate"]):
        raise ValueError(f"Inconsistent exact eviction counts: {log}")
    row = {
        "source_log": str(log), "source_resources": str(resource_path), "method": method,
        "rank": rank, "prefetch_budget": resources.get("prefetch_budget", 1),
        "completed_samples": len(samples), "mh_steps_per_sample": samples[0]["mh_transitions"],
        "mean_sampler_seconds": float(np.mean([s["sampler_seconds"] for s in samples])),
        "mean_mh_seconds": float(np.mean([s["mh_seconds"] for s in samples])),
        "mean_mh_calls": calls / len(samples),
        "transitions_per_mh_call": sum(s["mh_transitions"] for s in samples) / calls,
        "mean_seconds_per_mh_call": sum(s["mh_seconds"] for s in samples) / calls,
        "mean_build_seconds": sum(float(b[0]) for b in breakdown) / len(samples),
        "mean_generate_seconds": sum(float(b[1]) for b in breakdown) / len(samples),
        "mean_score_seconds": sum(float(b[2]) for b in breakdown) / len(samples),
        "total_llm_generate_calls": resources["llm_generate_calls"],
        "peak_gpu_memory_gib": resources["peak_gpu_memory_gib"],
        "generation_memory_delta_mb": resources["peak_llm_generate_gpu_memory_delta_gib"] * 2**30 / 1e6,
        "peak_kv_occupancy_percent": resources["peak_kv_cache_occupancy"] * 100,
        "prefix_hit_percent": hit / queried * 100,
        "prefix_miss_tokens_per_sample": (queried - hit) / len(samples),
        "allocated_blocks_per_sample": allocated / len(samples),
        "evicted_blocks_per_sample": evicted / len(samples),
        "eviction_rate_percent": evicted / allocated * 100,
        "correct_samples": sum(r["is_correct"].lower() == "true" for r in results),
    }
    stats_path = Path(str(prefix) + ".stats.json")
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    row["acceptance_rate"] = None
    row["proposal_requests_per_sample"] = calls / len(samples) if baseline else None
    if stats is not None:
        if sorted(s["idx"] for s in stats) != sample_ids or sum(s["total_nfe"] for s in stats) != calls:
            raise ValueError(f"Stats and timing traces disagree: {log}")
        row["acceptance_rate"] = sum(s["total_acceptances"] for s in stats) / sum(s["total_walked_steps"] for s in stats)
        row["proposal_requests_per_sample"] = sum(s["total_workload"] for s in stats) / len(samples)
    engine = []
    engine_samples = parse_engine_metrics(log)
    lines = text.splitlines()
    for sample in engine_samples:
        event = {
            "source_log": str(log), "method": method, **asdict(sample),
            "elapsed_minutes": (sample.timestamp - engine_samples[0].timestamp).total_seconds() / 60,
            "eviction_rate_percent": None, "evicted_blocks": None, "allocated_blocks": None,
        }
        if match := EVICTIONS.search(lines[sample.source_line - 1]):
            rate, evicted_count, allocated_count = map(float, match.groups())
            if allocated_count <= 0 or not math.isclose(rate, 100 * evicted_count / allocated_count, abs_tol=5.1e-7):
                raise ValueError(f"Invalid interval eviction counters: {log}:{sample.source_line}")
            event.update(eviction_rate_percent=rate, evicted_blocks=int(evicted_count), allocated_blocks=int(allocated_count))
        engine.append(event)
    return row, samples, engine


def plot_budget_metrics(rows: list[dict], prefix: Path, suffix: str, panels: tuple) -> None:
    """Compare all recorded ranks and budgets against their matching baseline."""
    apply_plot_style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9})
    baseline = next(row for row in rows if row["method"] == POWER_METHOD_LABEL)
    ranks = sort_rank_names(row["rank"] for row in rows if row["rank"])
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 6.8))
    for ax, (key, label) in zip(axes.flat, panels, strict=True):
        for rank in ranks:
            selected = sorted((row for row in rows if row["rank"] == rank), key=lambda row: row["prefetch_budget"])
            ax.plot([row["prefetch_budget"] for row in selected], [row[key] for row in selected],
                    marker="o", color=TRANSITION_RANK_COLORS[rank], label=rank_label(rank))
        ax.axhline(baseline[key], color=DARK_GRAY, linestyle="--", label=POWER_METHOD_LABEL)
        ax.set(ylabel=label, xlabel="Prefetch budget", xticks=sorted({row["prefetch_budget"] for row in rows if row["rank"]}))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22)
    figure.suptitle(f"{model_of(Path(baseline['source_log']))} | {baseline['completed_samples']} samples, "
                   f"{baseline['mh_steps_per_sample']} MH transitions per sample\n"
                   f"{SUBTREE_METHOD_LABEL} by traversal rule vs. {POWER_METHOD_LABEL}", y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    figure.tight_layout(rect=(0, 0.09, 1, 0.93), h_pad=2)
    save_figure(figure, Path(str(prefix) + suffix + ".pdf"))


def plot_engine_comparison(baseline: list[dict], subtree: list[dict], log: Path) -> None:
    """Overlay both methods' raw engine intervals, retaining unavailable evictions."""
    panels = (
        ("generation_throughput_tokens_s", "Generation throughput (tokens/s)"),
        ("prompt_throughput_tokens_s", "Prompt throughput (tokens/s)"),
        ("running_requests", "Running requests"),
        ("gpu_kv_cache_usage_percent", r"GPU KV-cache usage (\%)"),
        ("prefix_cache_hit_rate_percent", r"Prefix-cache hit rate (\%)"),
        ("eviction_rate_percent", r"Interval KV eviction rate (\%)"),
    )
    apply_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 6.2))
    for ax, (key, label) in zip(axes.flat, panels, strict=True):
        for events, color, method in ((baseline, BLUE, POWER_METHOD_LABEL), (subtree, GREEN, SUBTREE_METHOD_LABEL)):
            ax.plot([row["elapsed_minutes"] for row in events],
                    [np.nan if row[key] is None else row[key] for row in events],
                    color=color, linewidth=0.75, alpha=0.85, label=method)
        ax.set(xlabel="Time since first engine record (min)", ylabel=label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22)
    figure.suptitle(f"{model_of(log)} | {rank_label(rank_of(log))} | "
                   f"prefetch budget {prefetch_budget_of(log)}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    prefix = str(config_output_prefix(log)) + ".entropycut-engine-comparison"
    save_figure(figure, Path(prefix + ".pdf"))
    write_dict_rows(Path(prefix + ".csv"), baseline + subtree)


def plot_sample_diagnostics(log: Path) -> None:
    """Plot observed call yields, durations, cut positions, and draw probabilities."""
    traces = extract_model_call_traces(log)
    yields, durations = [], []
    for trace in traces:
        starts = [call.mh_step for call in trace.calls] + [trace.total_mh_steps]
        yields.extend(np.diff(starts))
        durations.extend(call.duration_seconds for call in trace.calls)
    draws = CUT_DRAWS.findall(log.read_text())
    cuts, probabilities, draw_rows = [], [], []
    for draw_index, (indices, masses) in enumerate(draws):
        batch_cuts, batch_probs = ast.literal_eval(indices), ast.literal_eval(masses)
        if len(batch_cuts) != len(batch_probs) or any(not 0 < p <= 1 for p in batch_probs):
            raise ValueError(f"Invalid sampled cut probabilities: {log}")
        cuts.extend(batch_cuts)
        probabilities.extend(batch_probs)
        draw_rows.extend({"source_log": str(log), "draw_index": draw_index, "pop_index": i,
                          "cut_index": cut, "sampling_probability": probability}
                         for i, (cut, probability) in enumerate(zip(batch_cuts, batch_probs, strict=True)))
    apply_plot_style()
    figure, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    axes[0].hist(yields, bins=np.arange(0.5, max(yields) + 1.5), color=BLUE)
    axes[1].hist(durations, bins=25, color=BLUE)
    axes[2].hist(cuts, bins=30, color=GREEN)
    axes[3].hist(probabilities, bins=30, color=GREEN)
    for ax, label in zip(axes, ("Realized MH transitions per call", "Logged time per MH call (s)", "Sampled cut index", "Probability at sampled cut"), strict=True):
        ax.set(xlabel=label, ylabel="Count")
    figure.suptitle(f"{SUBTREE_METHOD_LABEL} | {model_of(log)} | {rank_label(rank_of(log))} | budget {prefetch_budget_of(log)}")
    figure.tight_layout()
    prefix = str(config_output_prefix(log)) + ".entropycut-call-and-cut-diagnostics"
    save_figure(figure, Path(prefix + ".pdf"))
    write_dict_rows(Path(prefix + ".calls.csv"), [{"source_log": str(log), "call_index": i, "realized_transitions": int(y), "logged_mh_seconds": d} for i, (y, d) in enumerate(zip(yields, durations, strict=True))])
    if draw_rows:
        write_dict_rows(Path(prefix + ".cuts.csv"), draw_rows)


def render_summary(logs: list[Path], prefix: Path) -> None:
    """Write full-sample timing/workload summaries and paired engine trajectories."""
    parsed = {log: read_run(log) for log in logs}
    rows = [value[0] for value in parsed.values()]
    baseline_log = next(log for log in logs if ".powerMH." in log.name)
    baseline = parsed[baseline_log][0]
    for row in rows:
        row["sampler_speedup"] = baseline["mean_sampler_seconds"] / row["mean_sampler_seconds"]
        row["mh_speedup"] = baseline["mean_mh_seconds"] / row["mean_mh_seconds"]
    write_dict_rows(Path(str(prefix) + ".entropycut-run-summary.csv"), rows)
    write_dict_rows(Path(str(prefix) + ".entropycut-sample-timing.csv"), [row for value in parsed.values() for row in value[1]])
    plot_budget_metrics(rows, prefix, ".entropycut-time-comparison", (
        ("mean_sampler_seconds", "Sampler time per sample (s)"),
        ("mean_mh_seconds", "Logged MH time per sample (s)"),
        ("mean_mh_calls", "MH model calls per sample"),
        ("transitions_per_mh_call", "MH transitions per model call"),
        ("sampler_speedup", "Sampler time speedup"),
        ("mh_speedup", "Logged MH time speedup"),
    ))
    plot_budget_metrics(rows, prefix, ".entropycut-workload-and-time-breakdown", (
        ("mean_build_seconds", "Build time per sample (s)"),
        ("mean_generate_seconds", "Generate time per sample (s)"),
        ("mean_score_seconds", "Score time per sample (s)"),
        ("proposal_requests_per_sample", "Proposal requests per sample"),
        ("allocated_blocks_per_sample", "KV blocks allocated per sample"),
        ("evicted_blocks_per_sample", "Cached KV blocks evicted per sample"),
    ))
    for log in logs:
        if log != baseline_log:
            plot_engine_comparison(parsed[baseline_log][2], parsed[log][2], log)
            plot_sample_diagnostics(log)

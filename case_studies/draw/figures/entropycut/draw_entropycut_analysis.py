#!/usr/bin/env python3
"""Render all available EntropyCut case-study diagnostics and comparisons.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_entropycut_analysis.py

Use --directory /absolute/path/to/logs-entropycut for another log root, or
--aggregate-only to regenerate comparisons without per-log panels. Figures and
source CSVs are written beside their inputs. This process configures the shared
renderers for EntropyCut labels and all recorded ranks before importing them;
the default labels and rank exclusions of other drawing commands are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import sys
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/power-sharpening-entropycut/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/power-sharpening-entropycut")
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)

from case_studies import paths, plot_config  # noqa: E402
from case_studies.draw import figures  # noqa: E402
from case_studies.extract import run_naming  # noqa: E402

DEFAULT_DIRECTORY = paths.CASE_STUDIES_DIR / "logs-entropycut"


def configure_renderers() -> None:
    """Select the method labels and complete traversal palette for this run."""
    plot_config.POWER_METHOD_LABEL = "EntropyCut"
    plot_config.SUBTREE_METHOD_LABEL = "PreSTO-EntropyCut"
    plot_config.TRANSITION_RANK_COLORS["bfs_accept_first"] = plot_config.PURPLE
    run_naming.EXCLUDED_RANKS = frozenset()


def call_cli(name: str, arguments: list[str], failures: list[dict]) -> None:
    """Run a shared renderer, recording any failure while continuing the sweep."""
    old_argv = sys.argv
    sys.argv = [name, *arguments]
    print(f"RENDER {name} {' '.join(arguments)}", flush=True)
    try:
        runpy.run_module(figures.module_path(name), run_name="__main__", alter_sys=True)
    except (Exception, SystemExit) as error:
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            failures.append({"renderer": name, "arguments": arguments, "error": str(error)})
            traceback.print_exc()
    finally:
        sys.argv = old_argv


def render_rank_comparisons(logs: list[Path], prefix: Path) -> None:
    """Draw proposal certainty and realized transitions for one matched config."""
    from case_studies.draw.common.figure_io import save_figure, write_dict_rows
    from case_studies.draw.figures.proposals import draw_rank_proposal_certainty as certainty
    from case_studies.draw.figures.proposals import draw_rank_realized_mh_transitions as transitions
    from case_studies.extract.analysis.rank_analysis import RankComparison, collect_rank_calls

    by_budget: dict[int, list[Path]] = defaultdict(list)
    for log in logs:
        by_budget[run_naming.prefetch_budget_of(log)].append(log)
    comparisons = [
        RankComparison(tuple(collect_rank_calls(paths)), budget)
        for budget, paths in sorted(by_budget.items())
    ]
    save_figure(transitions.draw_figure(comparisons), Path(str(prefix) + transitions.FIGURE_SUFFIX))
    write_dict_rows(
        Path(str(prefix) + transitions.CSV_SUFFIX),
        [row for comparison in comparisons for row in comparison.transition_summary_rows()],
    )
    # The certainty CLI exports its own detailed probability histogram table.
    call_failures: list[dict] = []
    pattern = logs[0].name
    import re
    pattern = re.sub(r"prefetch-budget-\d+", "prefetch-budget-*", pattern)
    pattern = re.sub(r"rank-[^.]+", "rank-*", pattern)
    call_cli("draw_rank_proposal_certainty", [
        "--directory", str(logs[0].parent), "--log-glob", pattern,
        "--output", str(prefix) + certainty.FIGURE_SUFFIX,
        "--csv-output", str(prefix) + certainty.CSV_SUFFIX,
    ], call_failures)
    if call_failures:
        raise ValueError(call_failures)


def render_accept_first_comparison(logs: list[Path], prefix: Path) -> Path:
    """Reproduce the dedicated three-panel layout for one matched EntropyCut run.

    Args:
        logs: Complete baseline and subtree logs with identical shared settings.
        prefix: Absolute output prefix for that shared run configuration.

    Returns:
        The PDF path, with a sibling CSV recording every plotted mean and SD.
    """
    from case_studies.draw.common.figure_io import write_dict_rows
    from case_studies.draw.figures.model_calls import (
        draw_accept_first_power_mh_model_calls_and_empirical_time as combined,
    )
    from case_studies.extract.logs.model_call_traces import extract_model_call_traces

    combined.SUBTREE_LABEL_FONT_SIZE = 10.0
    completed = []
    for group in combined.discover_experiment_groups(prefix.parent, combined.MODEL_CALL_OUTPUT_SUFFIX):
        if run_naming.shared_run_config_of(group.log_paths[0]) != prefix.name:
            continue
        analysis = combined.accept_first_analysis(
            group, traces_per_rank=max(len(extract_model_call_traces(log)) for log in group.log_paths),
            include_incomplete=False,
        )
        if analysis is not None:
            completed.append((group, analysis))
    comparisons = combined.build_batch_comparisons(completed)
    if len(comparisons) != 1:
        raise ValueError(f"Expected one accept-first comparison for {prefix.name}")
    comparison = comparisons[0]
    baseline_log = next(log for log in logs if ".powerMH." in log.name)
    baseline_log = combined.find_power_mh_log(
        prefix.parent, combined.comparison_source_log(comparison), baseline_log,
    )
    baseline = combined.load_power_mh_time_data(baseline_log)
    output = prefix.parent / (
        run_naming.dataset_model_prefix(baseline_log)
        + combined.FIGURE_SUFFIX.replace("power-mh", "entropycut-mh")
    )
    written, efficiency_budget, time_budget, per_call_budget = combined.plot_figure(comparison, baseline, output)
    summary = []
    for panel, kind, budget, scale in (
        ("a", combined.EMPIRICAL_TIME, time_budget, 1 / 60),
        ("b", combined.TRANSITIONS_PER_CALL, efficiency_budget, 1),
        ("c", combined.TIME_PER_CALL, per_call_budget, 1),
    ):
        analysis = dict(comparison.batch_analyses)[budget]
        for row in kind.summary_rows(analysis):
            summary.append({
                "panel": panel, "metric": kind.axis_label,
                "method": plot_config.SUBTREE_METHOD_LABEL, "prefetch_budget": budget,
                "mh_iteration": row["mh_iteration"], "trace_count": row["trace_count"],
                "mean": float(row[kind.mean_key]) * scale, "sample_sd": float(row[kind.sd_key]) * scale,
            })
        baseline_rows = baseline.time_per_call_summary if panel == "c" else baseline.summary
        mean_key = "mean_empirical_seconds_per_call" if panel == "c" else "mean_cumulative_seconds"
        sd_key = "std_empirical_seconds_per_call" if panel == "c" else "std_cumulative_seconds"
        for row in baseline_rows:
            summary.append({
                "panel": panel, "metric": kind.axis_label, "method": plot_config.POWER_METHOD_LABEL,
                "prefetch_budget": 1, "mh_iteration": row["mh_iteration"], "trace_count": row["trace_count"],
                "mean": 1.0 if panel == "b" else float(row[mean_key]) * scale,
                "sample_sd": 0.0 if panel == "b" else float(row[sd_key]) * scale,
            })
    write_dict_rows(output.with_suffix(".summary.csv"), summary)
    print(f"Dedicated EntropyCut figure: {written}; panel budgets (a,b,c): {time_budget}, {efficiency_budget}, {per_call_budget}", flush=True)
    return written


def write_index(directory: Path, manifest: dict) -> None:
    """Write a browsable index with coverage, measured results, and definitions."""
    complete = sum(row["status"] == "complete" for row in manifest["inputs"])
    lines = [
        "# EntropyCut analysis", "",
        f"{len(manifest['figures'])} PDF 1.7 figures from {complete} complete logs. "
        "Methods: EntropyCut and PreSTO-EntropyCut. Every recorded traversal rule and prefetch budget is included; models and common run settings are kept separate.", "",
        "## Main comparisons", "",
        "| Model | Samples per run | MH transitions per sample | Figures and source data |",
        "| --- | ---: | ---: | --- |",
    ]
    summaries = []
    for table in sorted(directory.rglob("*.entropycut-run-summary.csv")):
        with table.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        baseline = next(row for row in rows if not row["rank"])
        best = min((row for row in rows if row["rank"]), key=lambda row: float(row["mean_sampler_seconds"]))
        prefix = str(table).removesuffix(".entropycut-run-summary.csv")
        links = []
        dedicated = table.parent / (
            run_naming.dataset_model_prefix(Path(baseline["source_log"]))
            + ".all-prefetch-budget.accept-first-entropycut-mh.model-calls-and-empirical-time.pdf"
        )
        if dedicated.exists():
            links.append(f"[Three-panel comparison](<{dedicated}>)")
        for label, suffix in (
            ("Time", "entropycut-time-comparison.pdf"),
            ("KV cache", "entropycut-kv-cache-comparison.pdf"),
            ("Memory and prefix misses", "entropycut-memory-and-prefix-workload.pdf"),
            ("Workload and time breakdown", "entropycut-workload-and-time-breakdown.pdf"),
            ("Run data", "entropycut-run-summary.csv"),
            ("Sample timing", "entropycut-sample-timing.csv"),
        ):
            path = prefix + "." + suffix
            if Path(path).exists():
                links.append(f"[{label}](<{path}>)")
        model = run_naming.model_of(Path(baseline["source_log"]))
        lines.append(f"| {model} | {baseline['completed_samples']} | {baseline['mh_steps_per_sample']} | {' · '.join(links)} |")
        summaries.append(
            f"- {model}: fastest recorded configuration is `{best['rank']}`, budget {best['prefetch_budget']}; "
            f"mean sampler time {float(baseline['mean_sampler_seconds']):.2f} → {float(best['mean_sampler_seconds']):.2f} s "
            f"({float(best['sampler_speedup']):.2f}×). Peak occupied KV blocks: "
            f"{float(baseline['peak_kv_occupancy_percent']):.3f}% → {float(best['peak_kv_occupancy_percent']):.3f}%; "
            f"eviction rate: {float(baseline['eviction_rate_percent']):.2f}% → {float(best['eviction_rate_percent']):.2f}%."
        )
    lines += ["", "## Recorded results", "", *summaries, "", "## Measurement definitions and coverage", "",
        "- Sampler time uses the per-prompt sampler timer, including initial generation and MH sampling. It excludes model startup and answer grading. Logged MH time sums the recorded MH-step durations; these include call preparation and scoring, so they are not pure GPU kernel time.",
        "- Timing and efficiency summaries use every completed sample (5 for 4B; 20 for 9B). Speedup is the ratio of mean times. The paired cumulative-time figures also show sample traces and mean ± one sample standard deviation.",
        "- MH model-call counts exclude initial extension calls. The run-summary CSV separately retains the resource probe's total llm.generate count. Transitions per call is total realized MH transitions divided by total MH calls.",
        "- Cache occupancy is the fraction of the physical KV-block pool occupied, expressed as a percentage. The logs do not provide usable KV-pool GiB measurements, so no KV-cache size in GiB is inferred.",
        "- Exact KV eviction rate is cached physical blocks evicted divided by physical blocks allocated. Prefix hit rate uses hit-token/query-token counts. Prefix miss workload includes unused proposals. Generation memory delta is sampled process GPU memory above the loaded baseline, in decimal MB; recorded zero means the clipped measured delta is zero.",
        "- Engine comparisons plot the original periodic samples against elapsed time from each run's first engine record. Interval eviction measurements marked unavailable remain gaps. Periodic prefix-hit values are retained as logged; aggregate cache comparisons use the resource probe's run-level token counts.",
        "- Call/cut histograms retain all sampled cut draws, including repeated positions. Probabilities are the logged probabilities at sampled positions, without renormalization. Build/generate/score breakdowns retain their logged precision.",
        "- These are comparisons of the supplied runs, not additional independent benchmark repetitions.", "",
        "## Unavailable diagnostics", "",
        "- The complete subtree runs used `print_tree=false`. Proposal-level acceptance/certainty and resampling-versus-ratio panels require missing Node records and cannot be recovered. Realized-transition distributions are reconstructed from consecutive MH-call step indices.",
    ]
    for row in manifest["inputs"]:
        if row["status"] == "skipped":
            lines.append(f"- Skipped [{Path(row['source_log']).name}](<{row['source_log']}>): {row['reason']}.")
    lines += ["", "## Reproduce", "", "PDF 1.7 export requires the `qpdf` executable on PATH.", "", "```bash",
        f"{paths.REPO_ROOT / 'src/.venv/bin/python'} {paths.DRAW_DIR / 'draw_entropycut_analysis.py'} --directory {directory}",
        "```", "", "## All figures", "",
    ]
    for value in manifest["figures"]:
        path = Path(value)
        lines.append(f"- [{path.name}](<{path}>)")
    lines += ["", f"[Input and output manifest](<{directory / 'analysis-manifest.json'}>)", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Audit inputs, render diagnostics, and write an explicit coverage manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"No such directory: {directory}")
    configure_renderers()

    from case_studies.extract.logs.model_call_traces import extract_model_call_traces

    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    inventory: list[dict] = []
    failures: list[dict] = []
    for log in sorted(directory.rglob("*.vllm.log")):
        record = {"source_log": str(log)}
        if ".cut-entropy" not in log.name:
            raise ValueError(f"Expected an entropy-tagged log: {log}")
        try:
            traces = extract_model_call_traces(log)
            for trace in traces:
                trace.validate(str(log))
            record.update(
                traces=len(traces), complete_traces=sum(t.complete for t in traces),
                model_calls=sum(len(t.calls) for t in traces),
            )
            if not all(t.complete for t in traces):
                raise ValueError("Incomplete timing traces; excluded from matched comparisons")
            if "INFO: output saved to" not in log.read_text(encoding="utf-8"):
                raise ValueError("No completed-run output marker")
        except ValueError as error:
            record["status"] = "skipped"
            record["reason"] = str(error)
        else:
            record["status"] = "complete"
            record["proposal_nodes_available"] = "Node(" in log.read_text(encoding="utf-8")
            if ".subtreePrefetch." in log.name and not record["proposal_nodes_available"]:
                record["unavailable_figures"] = "Proposal acceptance/certainty and resampling panels require Node records (print_tree=false in these logs)."
            groups[(log.parent, run_naming.shared_run_config_of(log))].append(log)
            if not args.aggregate_only and ".subtreePrefetch." in log.name:
                scripts = ["draw_vllm_engine_metrics"]
                if record["proposal_nodes_available"]:
                    scripts += ["draw_resample_vs_ratio", "draw_acceptance_panels"]
                for name in scripts:
                    call_cli(name, ["--log", str(log)], failures)
        inventory.append(record)

    for parent in sorted({parent for parent, _ in groups}):
        for name in ("draw_vllm_engine_metrics", "draw_power_mh_time", "draw_model_calls_step", "draw_resource_usage", "draw_peak_kv_cache", "draw_mean_mh_transitions_and_empirical_time"):
            cli_args = ["--directory", str(parent)]
            if name in {"draw_model_calls_step", "draw_peak_kv_cache", "draw_mean_mh_transitions_and_empirical_time"}:
                cli_args += ["--traces-per-rank", str(max(r.get("complete_traces", 0) for r in inventory))]
            call_cli(name, cli_args, failures)

    for (parent, signature), logs in sorted(groups.items()):
        baseline_logs = [log for log in logs if ".powerMH." in log.name]
        subtree_logs = [log for log in logs if ".subtreePrefetch." in log.name]
        prefix = parent / signature
        if subtree_logs and all("Node(" in log.read_text(encoding="utf-8") for log in subtree_logs):
            try:
                render_rank_comparisons(subtree_logs, prefix)
            except Exception as error:
                failures.append({"renderer": "rank comparisons", "config": signature, "error": str(error)})
                traceback.print_exc()
        if len(baseline_logs) != 1 or not subtree_logs:
            continue
        from case_studies.draw.figures.entropycut.entropycut_summary import render_summary
        try:
            render_accept_first_comparison(logs, prefix)
            render_summary(logs, prefix)
        except Exception as error:
            failures.append({"renderer": "entropycut summary", "config": signature, "error": str(error)})
            traceback.print_exc()
        baseline = baseline_logs[0]
        for log in subtree_logs:
            call_cli("draw_power_mh_vs_subtree_time", [
                "--power-mh-log", str(baseline), "--subtree-log", str(log),
                "--traces-per-rank", str(len(extract_model_call_traces(log))),
            ], failures)
        for first, third, suffix in (
            ("peak-occupancy", "eviction-rate", "entropycut-kv-cache-comparison"),
            ("peak-gpu-memory-delta", "miss-tokens-per-answer", "entropycut-memory-and-prefix-workload"),
        ):
            call_cli("draw_combined_cache_metrics", [
                "--baseline-resource-json", str(parent / (signature + ".resources.json")),
                "--dataset-label", "LCB V6" if run_naming.dataset_of(baseline) == "lcb_v6" else run_naming.dataset_of(baseline),
                "--model-label", run_naming.model_of(baseline), "--include-excluded-ranks",
                "--first-metric", first, "--third-metric", third,
                "--hit-rate-tick-step", "5", "--output", str(prefix) + "." + suffix + ".pdf",
            ], failures)
        call_cli("draw_prefix_cache_hit_rate", [
            "--baseline-resource-json", str(parent / (signature + ".resources.json")),
            "--dataset-label", "LCB V6" if run_naming.dataset_of(baseline) == "lcb_v6" else run_naming.dataset_of(baseline),
            "--model-label", run_naming.model_of(baseline), "--include-excluded-ranks",
            "--output", str(prefix) + ".prefix-cache-hit-rate.pdf",
        ], failures)

    manifest = {"pdf_version": "1.7", "inputs": inventory, "render_failures": failures,
                "figures": [str(path) for path in sorted(directory.rglob("*.pdf"))]}
    (directory / "analysis-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(directory, manifest)
    print(f"SUMMARY: {sum(r['status'] == 'complete' for r in inventory)}/{len(inventory)} usable logs; {len(manifest['figures'])} PDFs; {len(failures)} render failures", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

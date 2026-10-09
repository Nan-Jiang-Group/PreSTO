#!/usr/bin/env python3
"""Compare Multiple-Try MH with PowerMH and PreSTO-PowerMH on every dataset/model pair that has MultiTry logs.

Run with:

    MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_multitry_mh_figures.py

MultiTry runs still in progress contribute their finished prompts only; every figure prints that count ``n``. For
each pair the MultiTry log with the most finished prompts is used (TACC and punakha copies of the same run may both
exist). Baselines are the latest complete PowerMH and PreSTO-PowerMH (accept-first BFS, prefetch budget 20, or 10 on
LCB V6) logs with the same alpha, MH steps, block count, sample limit, token limit, and seed. LCB V6 baselines must
come from runs that logged the V6 file hash, so Qwen3.5-9B has none there. Accuracy and per-prompt time compare the
same first ``n`` prompts for all three methods. The runs were collected on different dates and machines, so the
timing panels do not establish a controlled speedup.

Writes four PDFs and one CSV of every plotted value into ``--output-dir``:

* ``multitry-mh.cumulative-time.pdf``: mean cumulative MH-step time against MH transitions (rows: datasets, columns:
  models).
* ``multitry-mh.accuracy.pdf``: accuracy on the first ``n`` prompts.
* ``multitry-mh.accuracy-vs-time.pdf``: accuracy against mean sampling time per prompt.
* ``multitry-mh.diagnostics.pdf``: realized acceptance rate, epsilon-certainty of the MTM acceptance probability,
  and the MultiTry per-step time split between suffix generation and token scoring.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.figures.proposals.draw_dataset_proposal_certainty import DATASET_LABELS, MODEL_LABELS
from case_studies.extract.logs.model_call_traces import (
    NoModelCallTracesError,
    power_mh_trace_times,
    subtree_mh_trace_times,
)
from case_studies.extract.run_naming import RANK_ORDER
from case_studies.plot_config import BLUE, DARK_GRAY, GRAY, GREEN, PURPLE, SUBTREE_BUCKET_COLORS, apply_plot_style

apply_plot_style()
csv.field_size_limit(sys.maxsize)

CASE_DIR = paths.CASE_STUDIES_DIR
DATASETS = ("math500", "aime", "gpqa", "human_eval", "mbpp", "lcb_v6")
MODELS = ("qwen3-4b", "qwen3-8b", "qwen3.5-4b", "qwen3.5-9b", "gemma-12b-it")
SHARED_CONFIG = "alpha4.0.steps100.blocks-1"
SHARED_TAIL = "samples20.maxnew1024.seed10086"
MULTI_TRY_CONFIG = "tries4.temps0.25-0.5-1.0.scorebatch128"
# LCB V6 runs before this date loaded the V5 file under a V6 label.
LCB_V6_VERIFIED_FROM = "2026-09-23"
DEFAULT_OUTPUT_DIR = CASE_DIR / "logs-MultiTry" / "all-datasets" / "2026-09-24" / "vllm"

POWER, PRESTO, MULTI = "PowerMH", "PreSTO-PowerMH", "MultiTry-MH"
METHODS = (POWER, PRESTO, MULTI)
COLORS = {POWER: BLUE, PRESTO: GREEN, MULTI: PURPLE}
MODEL_MARKERS = dict(zip(MODELS, ("o", "s", "^", "D", "v")))
FONT_SIZE = 9.0
# Certainty threshold of the paper's edge-certainty analysis.
EPSILON = 0.01

PROMPT_DONE = re.compile(r"(?:mcmc_power_sampler|subtree_prefetching_sampling) took (?P<seconds>[\d.]+) seconds "
                         r"for (?P<index>\d+)-th prompts")
RUNNING_ACCURACY = re.compile(r"\] INFO: \[(?P<done>\d+)/\d+\] correct=(?P<correct>\d+)/\d+")
MULTI_STEP = re.compile(r"MH step (?P<step>\d+)/\d+: accepted (?P<accepted>\d+)/\d+ prompts, elapsed=(?P<s>[\d.]+)s")
MULTI_DECISION = re.compile(r"MH decision: .*mtm_acceptance_probability=(?P<a>\S+) accepted=(?P<acc>True|False)")
GENERATION_DONE = re.compile(r"Suffix generation finished: .* elapsed=(?P<s>[\d.]+)s")
SCORING_DONE = re.compile(r"Token scoring finished: .* elapsed=(?P<s>[\d.]+)s")


@dataclass
class MultiTryPrompt:
    """One finished MultiTry prompt; lists hold one entry per MH step."""
    total_seconds: float = 0.0
    correct: bool = False
    step_seconds: list[float] = field(default_factory=list)
    accepted: list[bool] = field(default_factory=list)
    acceptance_probability: list[float] = field(default_factory=list)
    generation_seconds: float = 0.0
    scoring_seconds: float = 0.0


@dataclass
class Pair:
    dataset: str
    model: str
    multi_log: Path
    multi: list[MultiTryPrompt]
    power_log: Path | None = None
    presto_log: Path | None = None
    presto_budget: int | None = None

    @property
    def n(self) -> int:
        return len(self.multi)


def parse_multi_try_log(path: Path) -> list[MultiTryPrompt]:
    """Parse the finished prompts of a MultiTry log; an in-progress prompt at EOF is dropped."""
    prompts: list[MultiTryPrompt] = []
    current = MultiTryPrompt()
    correct_so_far = 0
    with path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := MULTI_DECISION.search(line):
                current.acceptance_probability.append(float(match["a"]))
            elif match := MULTI_STEP.search(line):
                current.step_seconds.append(float(match["s"]))
                current.accepted.append(int(match["accepted"]) > 0)
            elif match := GENERATION_DONE.search(line):
                current.generation_seconds += float(match["s"])
            elif match := SCORING_DONE.search(line):
                current.scoring_seconds += float(match["s"])
            elif match := PROMPT_DONE.search(line):
                current.total_seconds = float(match["seconds"])
                prompts.append(current)
                current = MultiTryPrompt()
            elif (match := RUNNING_ACCURACY.search(line)) and int(match["done"]) == len(prompts):
                correct = int(match["correct"])
                prompts[-1].correct = correct > correct_so_far
                correct_so_far = correct
    return prompts


def prompt_seconds(path: Path) -> list[float]:
    """Per-prompt sampling time from a PowerMH or PreSTO log, in prompt order."""
    with path.open(encoding="utf-8", errors="replace") as log:
        return [float(match["seconds"]) for line in log if (match := PROMPT_DONE.search(line))]


def result_correctness(log_path: Path) -> list[bool]:
    """Per-prompt correctness from the result CSV written beside a baseline log."""
    name = re.sub(r"\.(powerMH|subtreePrefetch)\.vllm\.log$", ".csv", log_path.name)
    with (log_path.parent / name).open(encoding="utf-8", newline="") as handle:
        return [row["is_correct"].strip().lower() in ("true", "1", "1.0") for row in csv.DictReader(handle)]


def presto_acceptance(log_path: Path) -> list[float]:
    """Per-prompt realized acceptance ratio of the PreSTO-PowerMH chain (the PowerMH kernel)."""
    name = re.sub(r"\.subtreePrefetch\.vllm\.log$", ".csv", log_path.name)
    with (log_path.parent / name).open(encoding="utf-8", newline="") as handle:
        return [float(row["acceptance_ratio"]) for row in csv.DictReader(handle) if row.get("acceptance_ratio")]


def is_complete(path: Path) -> bool:
    with path.open(encoding="utf-8", errors="replace") as log:
        return any("output saved to" in line for line in log)


def latest_complete(dataset: str, pattern: str) -> Path | None:
    """Newest complete log matching ``pattern`` under ``logs/<dataset>/<date>/vllm``."""
    for log in sorted((CASE_DIR / "logs" / dataset).glob(f"*/vllm/{pattern}"), reverse=True):
        date = log.parent.parent.name
        if dataset == "lcb_v6" and date < LCB_V6_VERIFIED_FROM:
            continue
        if is_complete(log):
            return log
    return None


def discover_pairs() -> list[Pair]:
    pairs = []
    for dataset in DATASETS:
        for model in MODELS:
            stem = f"dataset-{dataset}.model-{model}.{SHARED_CONFIG}"
            candidates = [(parse_multi_try_log(log), log) for log in sorted(
                (CASE_DIR / "logs-MultiTry" / dataset).glob(
                    f"*/vllm/{stem}.{MULTI_TRY_CONFIG}.{SHARED_TAIL}.multiTryMH.vllm.log"))]
            # Most finished prompts wins; ties go to the shorter (TACC) date directory, listed first.
            candidates = [item for item in candidates if item[0]]
            if not candidates:
                continue
            prompts, log = max(candidates, key=lambda item: (len(item[0]), -len(item[1].parent.parent.name)))
            budget = 10 if dataset == "lcb_v6" else 20
            pairs.append(Pair(
                dataset, model, log, prompts,
                power_log=latest_complete(dataset, f"{stem}.{SHARED_TAIL}.powerMH.vllm.log"),
                presto_log=latest_complete(
                    dataset, f"{stem}.prefetch-budget-{budget}.rank-bfs_accept_first.{SHARED_TAIL}"
                             ".subtreePrefetch.vllm.log"),
                presto_budget=budget,
            ))
    return pairs


def cumulative_trajectories(pair: Pair) -> dict[str, list[list[float]]]:
    """Cumulative MH-step seconds per trace; entry ``i`` is the time to reach MH transition ``i + 1``."""
    out = {MULTI: [list(np.cumsum(prompt.step_seconds)) for prompt in pair.multi]}
    if pair.power_log is not None:
        out[POWER] = [cumulative for _, cumulative in power_mh_trace_times(pair.power_log)]
    if pair.presto_log is not None:
        try:
            trajectories, _ = subtree_mh_trace_times(pair.presto_log, 20, tuple(RANK_ORDER))
            out[PRESTO] = [cumulative for _, cumulative in trajectories]
        except NoModelCallTracesError:
            pass
    return out


def mean_sd(trajectories: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and SD over the traces that reached each MH transition."""
    length = max(len(trajectory) for trajectory in trajectories)
    means, sds = [], []
    for index in range(length):
        values = [trajectory[index] for trajectory in trajectories if len(trajectory) > index]
        means.append(np.mean(values))
        sds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
    return np.arange(1, length + 1), np.asarray(means), np.asarray(sds)


def label(pair_or_model: Pair | str) -> str:
    model = pair_or_model.model if isinstance(pair_or_model, Pair) else pair_or_model
    return MODEL_LABELS.get(model, model)


def method_legend(figure, methods: Sequence[str] = METHODS, y: float = 1.0, **kwargs) -> None:
    handles = [Line2D([], [], color=COLORS[method], linewidth=1.6, label=method) for method in methods]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles),
                  frameon=False, fontsize=FONT_SIZE, **kwargs)


def draw_cumulative_time(pairs: list[Pair], output: Path, rows: list[dict]) -> Path:
    datasets = [dataset for dataset in DATASETS if any(pair.dataset == dataset for pair in pairs)]
    figure, axes = plt.subplots(len(datasets), len(MODELS), figsize=(2.3 * len(MODELS), 1.75 * len(datasets)),
                                squeeze=False, sharex=True)
    by_key = {(pair.dataset, pair.model): pair for pair in pairs}
    for r, dataset in enumerate(datasets):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            ax.tick_params(labelsize=FONT_SIZE - 2)
            if r == 0:
                ax.set_title(label(model), fontsize=FONT_SIZE)
            if c == 0:
                ax.set_ylabel(f"{DATASET_LABELS.get(dataset, dataset)}\ntime (sec)", fontsize=FONT_SIZE - 1)
            if r == len(datasets) - 1:
                ax.set_xlabel(r"MH transitions ($K$)", fontsize=FONT_SIZE - 1)
            pair = by_key.get((dataset, model))
            if pair is None:
                ax.text(0.5, 0.5, "no MultiTry run", transform=ax.transAxes, ha="center", va="center",
                        fontsize=FONT_SIZE - 2, color=GRAY)
                ax.set_yticks([])
                ax.tick_params(axis="x", labelbottom=r == len(datasets) - 1)
                continue
            for method, trajectories in cumulative_trajectories(pair).items():
                steps, mean, sd = mean_sd(trajectories)
                ax.plot(steps, mean, color=COLORS[method], linewidth=1.3)
                ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[method], alpha=0.15, linewidth=0)
                rows.append({"figure": "cumulative-time", "dataset": dataset, "model": model, "method": method,
                             "traces": len(trajectories), "metric": "terminal mean cumulative MH-step seconds",
                             "value": f"{mean[-1]:.3f}", "sd": f"{sd[-1]:.3f}", "mh_transitions": int(steps[-1])})
            ax.text(0.03, 0.95, f"$n={pair.n}$", transform=ax.transAxes, ha="left", va="top",
                    fontsize=FONT_SIZE - 2, color=COLORS[MULTI])
    method_legend(figure, y=1.0)
    figure.tight_layout(h_pad=0.4, w_pad=0.4)
    return save_figure(figure, output, pad_inches=0.02)


def first_n(values: Sequence, n: int) -> list:
    return list(values[:n]) if len(values) >= n else []


def accuracy_and_time(pair: Pair) -> dict[str, tuple[float, float] | None]:
    """Accuracy and mean seconds per prompt on the first ``n`` prompts, per method."""
    n = pair.n
    out: dict[str, tuple[float, float] | None] = {
        MULTI: (np.mean([p.correct for p in pair.multi]), np.mean([p.total_seconds for p in pair.multi]))}
    for method, log in ((POWER, pair.power_log), (PRESTO, pair.presto_log)):
        if log is None:
            out[method] = None
            continue
        correct, seconds = first_n(result_correctness(log), n), first_n(prompt_seconds(log), n)
        out[method] = (np.mean(correct), np.mean(seconds)) if correct and seconds else None
    return out


def dataset_axes(pairs: list[Pair], width: float, height: float):
    datasets = [dataset for dataset in DATASETS if any(pair.dataset == dataset for pair in pairs)]
    columns = 3
    rows = -(-len(datasets) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    return figure, list(zip(datasets, axes.flat))


def draw_accuracy(pairs: list[Pair], output: Path, rows: list[dict]) -> Path:
    figure, panels = dataset_axes(pairs, 3.2, 2.2)
    width = 0.26
    for index, (dataset, ax) in enumerate(panels):
        subset = [pair for pair in pairs if pair.dataset == dataset]
        for x, pair in enumerate(subset):
            stats = accuracy_and_time(pair)
            for offset, method in zip((-width, 0.0, width), METHODS):
                if stats[method] is None:
                    ax.text(x + offset, 0.01, "--", ha="center", va="bottom", fontsize=FONT_SIZE - 3, color=GRAY)
                    continue
                accuracy = 100 * stats[method][0]
                ax.bar(x + offset, accuracy, width, color=COLORS[method], edgecolor="none")
                rows.append({"figure": "accuracy", "dataset": dataset, "model": pair.model, "method": method,
                             "traces": pair.n, "metric": "accuracy on first n prompts (%)",
                             "value": f"{accuracy:.1f}", "sd": "", "mh_transitions": 100})
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels([f"{label(pair)} ($n={pair.n}$)" for pair in subset], fontsize=FONT_SIZE - 2.5,
                           rotation=25, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 105)
        ax.set_ylabel(r"accuracy (\%)", fontsize=FONT_SIZE - 1)
        ax.set_title(rf"\textbf{{{panel_letter(index)}}} {DATASET_LABELS.get(dataset, dataset)}", fontsize=FONT_SIZE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE - 2)
    figure.legend(handles=[Patch(color=COLORS[m], label=m) for m in METHODS], loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=FONT_SIZE)
    figure.tight_layout()
    return save_figure(figure, output, pad_inches=0.02)


def draw_accuracy_vs_time(pairs: list[Pair], output: Path, rows: list[dict]) -> Path:
    figure, panels = dataset_axes(pairs, 2.9, 2.3)
    for index, (dataset, ax) in enumerate(panels):
        for pair in (pair for pair in pairs if pair.dataset == dataset):
            stats = accuracy_and_time(pair)
            points = {method: value for method, value in stats.items() if value is not None}
            xs = [points[m][1] / 60 for m in METHODS if m in points]
            ys = [100 * points[m][0] for m in METHODS if m in points]
            ax.plot(xs, ys, color=GRAY, linewidth=0.6, zorder=1)
            for method, (accuracy, seconds) in points.items():
                ax.scatter(seconds / 60, 100 * accuracy, color=COLORS[method], marker=MODEL_MARKERS[pair.model],
                           s=26, edgecolor="black", linewidth=0.3, zorder=2)
                rows.append({"figure": "accuracy-vs-time", "dataset": dataset, "model": pair.model,
                             "method": method, "traces": pair.n, "metric": "mean sampling minutes per prompt",
                             "value": f"{seconds / 60:.3f}", "sd": "", "mh_transitions": 100})
        ax.set_xlabel("sampling time per prompt (min)", fontsize=FONT_SIZE - 1)
        ax.set_ylabel(r"accuracy (\%)", fontsize=FONT_SIZE - 1)
        ax.set_ylim(-5, 105)
        ax.set_title(rf"\textbf{{{panel_letter(index)}}} {DATASET_LABELS.get(dataset, dataset)}", fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE - 2)
    handles = [Patch(color=COLORS[m], label=m) for m in METHODS]
    handles += [Line2D([], [], color=DARK_GRAY, marker=MODEL_MARKERS[m], linestyle="", label=label(m))
                for m in MODELS if any(pair.model == m for pair in pairs)]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                  frameon=False, fontsize=FONT_SIZE - 1, columnspacing=0.9, handletextpad=0.3)
    figure.tight_layout()
    return save_figure(figure, output, pad_inches=0.02)


def draw_diagnostics(pairs: list[Pair], output: Path, rows: list[dict]) -> Path:
    figure, (rate_ax, prob_ax, split_ax) = plt.subplots(1, 3, figsize=(12.5, 3.3),
                                                          gridspec_kw={"width_ratios": (1.25, 1.0, 1.25)})
    names = [f"{DATASET_LABELS.get(p.dataset, p.dataset)} / {label(p)}" for p in pairs]
    y = np.arange(len(pairs))

    # (a) Realized acceptance rate: MultiTry versus the PowerMH kernel as run by PreSTO-PowerMH.
    for row_index, pair in enumerate(pairs):
        multi_rate = 100 * np.mean([np.mean(p.accepted) for p in pair.multi])
        rate_ax.scatter(multi_rate, row_index, color=COLORS[MULTI], s=18, zorder=3)
        rows.append({"figure": "diagnostics", "dataset": pair.dataset, "model": pair.model, "method": MULTI,
                     "traces": pair.n, "metric": "realized acceptance rate (%)", "value": f"{multi_rate:.2f}",
                     "sd": "", "mh_transitions": 100})
        if pair.presto_log is not None:
            ratios = first_n(presto_acceptance(pair.presto_log), pair.n)
            if ratios:
                kernel_rate = 100 * np.mean(ratios)
                rate_ax.scatter(kernel_rate, row_index, color=COLORS[PRESTO], s=18, marker="s", zorder=3)
                rate_ax.plot([kernel_rate, multi_rate], [row_index, row_index], color=GRAY, linewidth=0.6)
                rows.append({"figure": "diagnostics", "dataset": pair.dataset, "model": pair.model,
                             "method": PRESTO, "traces": len(ratios), "metric": "realized acceptance rate (%)",
                             "value": f"{kernel_rate:.2f}", "sd": "", "mh_transitions": 100})
    rate_ax.set_yticks(y)
    rate_ax.set_yticklabels([f"{name} ($n={p.n}$)" for name, p in zip(names, pairs)], fontsize=FONT_SIZE - 3)
    rate_ax.invert_yaxis()
    rate_ax.set_xlabel(r"realized acceptance rate (\%)", fontsize=FONT_SIZE - 1)
    rate_ax.set_title(r"\textbf{(a)} acceptance per MH step", fontsize=FONT_SIZE)
    rate_ax.legend(handles=[Line2D([], [], color=COLORS[MULTI], marker="o", linestyle="", label=MULTI),
                            Line2D([], [], color=COLORS[PRESTO], marker="s", linestyle="",
                                   label=f"{PRESTO} (PowerMH kernel)")],
                   fontsize=FONT_SIZE - 2.5, loc="lower right", frameon=True)

    # (b) epsilon-certainty of the MTM acceptance probability A, with the certainty figures' buckets and colors.
    buckets = (("certain reject", lambda a: a <= EPSILON, SUBTREE_BUCKET_COLORS[2]),
               ("uncertain", lambda a: EPSILON < a < 1 - EPSILON, SUBTREE_BUCKET_COLORS[1]),
               ("certain accept", lambda a: a >= 1 - EPSILON, SUBTREE_BUCKET_COLORS[0]))
    left = np.zeros(len(pairs))
    for name, test, color in buckets:
        shares = np.array([100 * np.mean([test(a) for p in pair.multi for a in p.acceptance_probability])
                           for pair in pairs])
        prob_ax.barh(y, shares, left=left, color=color, label=rf"$\epsilon$-{name}")
        for pair, share in zip(pairs, shares):
            rows.append({"figure": "diagnostics", "dataset": pair.dataset, "model": pair.model, "method": MULTI,
                         "traces": pair.n, "metric": f"MTM acceptance probability, epsilon-{name} share (%)",
                         "value": f"{share:.2f}", "sd": "", "mh_transitions": 100})
        left += shares
    prob_ax.set_yticks(y)
    prob_ax.set_yticklabels([])
    prob_ax.invert_yaxis()
    prob_ax.set_xlim(0, 100)
    prob_ax.set_xlabel(rf"share of MH decisions (\%), $\epsilon={EPSILON}$", fontsize=FONT_SIZE - 1)
    prob_ax.set_title(r"\textbf{(b)} MultiTry-MH acceptance probability", fontsize=FONT_SIZE)
    prob_ax.legend(fontsize=FONT_SIZE - 3, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3,
                   frameon=False, handlelength=1.0, columnspacing=0.8)

    # (c) Where a MultiTry MH step spends its time.
    generation = np.array([np.mean([p.generation_seconds / len(p.step_seconds) for p in pair.multi]) for pair in pairs])
    scoring = np.array([np.mean([p.scoring_seconds / len(p.step_seconds) for p in pair.multi]) for pair in pairs])
    step = np.array([np.mean([np.mean(p.step_seconds) for p in pair.multi]) for pair in pairs])
    split_ax.barh(y, generation, color=DARK_GRAY, label="suffix generation")
    split_ax.barh(y, scoring, left=generation, color=COLORS[MULTI], label="token scoring")
    split_ax.barh(y, np.clip(step - generation - scoring, 0, None), left=generation + scoring, color=GRAY,
                  label="other")
    for pair, g, s, t in zip(pairs, generation, scoring, step):
        rows.append({"figure": "diagnostics", "dataset": pair.dataset, "model": pair.model, "method": MULTI,
                     "traces": pair.n, "metric": "mean seconds per MH step (generation/scoring/total)",
                     "value": f"{g:.3f}/{s:.3f}/{t:.3f}", "sd": "", "mh_transitions": 100})
    split_ax.set_yticks(y)
    split_ax.set_yticklabels([])
    split_ax.invert_yaxis()
    split_ax.set_xlabel("seconds per MH step", fontsize=FONT_SIZE - 1)
    split_ax.set_title(r"\textbf{(c)} MultiTry-MH time per step", fontsize=FONT_SIZE)
    split_ax.legend(fontsize=FONT_SIZE - 2.5, loc="lower right", frameon=True)
    for ax in (rate_ax, prob_ax, split_ax):
        ax.tick_params(labelsize=FONT_SIZE - 2)
    figure.tight_layout(w_pad=0.8)
    return save_figure(figure, output, pad_inches=0.02)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    pairs = discover_pairs()
    if not pairs:
        raise SystemExit("no MultiTry logs with a finished prompt")
    rows: list[dict] = []
    for pair in pairs:
        print(f"{pair.dataset:10s} {pair.model:13s} n={pair.n:2d} multi={pair.multi_log.parent.parent.name} "
              f"power={pair.power_log.parent.parent.name if pair.power_log else '-'} "
              f"presto={pair.presto_log.parent.parent.name if pair.presto_log else '-'}")
        for method, log in ((MULTI, pair.multi_log), (POWER, pair.power_log), (PRESTO, pair.presto_log)):
            rows.append({"figure": "sources", "dataset": pair.dataset, "model": pair.model, "method": method,
                         "traces": pair.n, "metric": "source log", "value": str(log) if log else "",
                         "sd": "", "mh_transitions": ""})
    out = args.output_dir
    for path in (
        draw_cumulative_time(pairs, out / "multitry-mh.cumulative-time.pdf", rows),
        draw_accuracy(pairs, out / "multitry-mh.accuracy.pdf", rows),
        draw_accuracy_vs_time(pairs, out / "multitry-mh.accuracy-vs-time.pdf", rows),
        draw_diagnostics(pairs, out / "multitry-mh.diagnostics.pdf", rows),
        write_dict_rows(out / "multitry-mh.summary.csv", rows),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

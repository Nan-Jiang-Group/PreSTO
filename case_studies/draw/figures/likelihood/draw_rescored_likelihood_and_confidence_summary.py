#!/usr/bin/env python3
"""Summarize rescored log-likelihood and confidence for every sampler/PreSTO pair.

The pairs are the jobs of case_studies/score_log_likelihood_and_confidence/score_log_likelihood_and_confidence.sh, each
a baseline run (PowerMH or EntropyCut) and its PreSTO runs, rescored under p_0. For every pair this script

* checks that all runs were scored on the same prompts: identical per-prompt token counts across the runs, and equal to
  the per-prompt ``context_len`` the PreSTO log records;
* checks the rescoring itself where the sampler logged both statistics (PreSTO-EntropyCut): among rows whose re-encoded
  length matches the logged one, the rescored log-likelihood must agree within --agreement-tol. Re-encoding decoded
  text can split the same characters into different tokens (whitespace runs in code), so a (dataset, model) with more
  than --max-disagreement of such rows is dropped from both families; one with no logged reference is kept and marked
  unvalidated;
* keeps the prompts whose rows are exact in every run (see draw_rescored_likelihood_and_confidence.py) and drops pairs
  with fewer than --min-exact of them;
* draws the two-panel histogram of draw_rescored_likelihood_and_confidence.py;
* compares the baseline with one PreSTO run (N_pf=20, or N_pf=10 where 20 was not run);
* reports n, mean, and standard deviation per method, and for the paired difference PreSTO minus baseline per prompt:
  an exact two-sided sign-flip test of no difference, Holm-corrected over every test in the summary, and a paired
  equivalence test (TOST, t-based) with margin --equivalence-margin times the baseline's across-prompt standard
  deviation, with its 90% interval;
* pools each sampler's standardized differences (difference / baseline SD) with a fixed-effect average and tests
  that pooled difference for equivalence against the same margin;
* writes one LaTeX table per cut law with the means, standard deviations, and both test results.

It then draws one summary figure per cut law (uniform, entropy): rows are pairs, columns are the two metrics, markers
are means with 95% intervals (1.96 standard errors). Outputs go to --output-dir: the figures, per-pair histograms under
pairs/, and likelihood-confidence-summary.csv.

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_rescored_likelihood_and_confidence_summary.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt
import numpy as np

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence import (
    group_colors,
    one_presto_budget,
    output_prefix,
    plot,
    read_groups,
    restrict_to_exact,
)
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import PANEL_SPECS, add_panel_label
from case_studies.draw.figures.proposals.draw_dataset_proposal_certainty import DATASET_LABELS, MODEL_LABELS
from case_studies.plot_config import GRAY

CASE_STUDIES_DIR = paths.CASE_STUDIES_DIR
LAUNCHER = CASE_STUDIES_DIR / "score_log_likelihood_and_confidence/score_log_likelihood_and_confidence.sh"
DEFAULT_OUTPUT_DIR = CASE_STUDIES_DIR / "score_log_likelihood_and_confidence/figures"
JOB_LINE = re.compile(r'^\s*"(\S+) (\S+) ([^"]+)"\s*$', re.M)
CONTEXT_LEN = re.compile(r"context_len=(\d+), new_suffix_len=(\d+)")
MODEL_DISPLAY = {**MODEL_LABELS, "qwen": "Qwen2.5-7B"}
DATASET_DISPLAY = {**DATASET_LABELS, "aime": "AIME"}
FAMILIES = (("uniform", "logs/"), ("entropy", "logs-entropycut/"))
MARKERS = ("o", "s", "^", "D")
# Every pair ran with max_new_tokens=1024; a PreSTO log's first proposal per prompt spans that whole horizon.
HORIZON = "1024"
csv_rows = list[dict]


def launcher_jobs(launcher: Path, prefixes: tuple[str, ...] = tuple(prefix for _, prefix in FAMILIES)
                  ) -> list[tuple[str, str, list[Path]]]:
    """Return (dataset, model, CSV paths) for every launcher job whose runs live under one of ``prefixes``."""
    return [(dataset, model, [CASE_STUDIES_DIR / p.replace(".csv", ".rescored.csv") for p in paths.split()])
            for dataset, model, paths in JOB_LINE.findall(launcher.read_text()) if paths.startswith(prefixes)]


def check_same_prompts(groups: dict[str, csv_rows], name: str) -> None:
    """Fail unless every run scored the same prompts, and PreSTO's match the lengths its log records."""
    lengths = {label: [int(row["prompt_tokens"]) for row in rows] for label, rows in groups.items()}
    reference = next(iter(lengths.values()))
    for label, values in lengths.items():
        if values != reference:
            raise SystemExit(f"{name}: {label} was scored on different prompts ({values} vs {reference})")
    for label, rows in list(groups.items())[1:]:
        source = Path(rows[0]["source_csv"])
        for log in source.parent.glob(source.name.removesuffix(".rescored.csv") + ".*.log"):
            logged = [int(c) for c, h in CONTEXT_LEN.findall(log.read_text(errors="replace")) if h == HORIZON]
            if logged and logged[:len(reference)] != reference:
                raise SystemExit(f"{name}: {label} prompts differ from the lengths its log records")


def rescoring_agreement(jobs: list[tuple[str, str, list[Path]]], tol: float) -> dict[tuple[str, str], dict]:
    """Per (dataset, model): how many length-matched rescored rows disagree with the sampler's logged log-likelihood."""
    stats: dict[tuple[str, str], dict] = {}
    for dataset, model, paths in jobs:
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if not row.get("logged_log_likelihood") or row["num_response_tokens"] != row["logged_num_response_tokens"]:
                        continue
                    gap = abs(float(row["log_likelihood"]) - float(row["logged_log_likelihood"]))
                    entry = stats.setdefault((dataset, model), {"dataset": dataset, "model": model, "checked": 0,
                                                                "disagree": 0, "gaps": []})
                    entry["checked"] += 1
                    entry["disagree"] += gap > tol
                    entry["gaps"].append(gap)
    for entry in stats.values():
        entry["median_gap"] = float(np.median(entry.pop("gaps")))
        entry["disagree_fraction"] = entry["disagree"] / entry["checked"]
    return stats


def sign_flip_p_value(differences: np.ndarray) -> float:
    """Exact two-sided p-value of the mean paired difference under random sign flips."""
    n = len(differences)
    observed = abs(differences.mean())
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    means = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def t_cdf(x: float, df: int) -> float:
    """Student t cumulative distribution, by numerical integration of its density (no SciPy here)."""
    grid = np.linspace(0.0, abs(x), 20001)
    density = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi) \
        * (1 + grid ** 2 / df) ** (-(df + 1) / 2)
    half = float(np.trapezoid(density, grid))
    return 0.5 + half if x >= 0 else 0.5 - half


def t_quantile(q: float, df: int) -> float:
    """Inverse of t_cdf by bisection."""
    low, high = -50.0, 50.0
    for _ in range(80):
        middle = (low + high) / 2
        low, high = (middle, high) if t_cdf(middle, df) < q else (low, middle)
    return (low + high) / 2


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def equivalence_test(differences: np.ndarray, margin: float) -> tuple[float, float, float]:
    """Paired TOST: (p-value for |mean difference| < margin, 90% interval low, high)."""
    n = len(differences)
    se = differences.std(ddof=1) / math.sqrt(n)
    mean = differences.mean()
    p_value = max(1 - t_cdf((mean + margin) / se, n - 1), t_cdf((mean - margin) / se, n - 1))
    half_width = t_quantile(0.95, n - 1) * se
    return p_value, mean - half_width, mean + half_width


def summarize_pair(family: str, dataset: str, model: str, groups: dict[str, csv_rows],
                   margin_sd: float) -> list[dict]:
    """One summary row per method and metric, with the paired difference for PreSTO rows."""
    labels = list(groups)
    baseline = {row["sample_idx"]: row for row in groups[labels[0]]}
    out = []
    for label in labels:
        for key, _ in PANEL_SPECS:
            values = [float(row[key]) for row in groups[label]]
            row = {"family": family, "dataset": dataset, "model": model, "method": label, "metric": key,
                   "n": len(values), "mean": fmean(values), "sd": stdev(values) if len(values) > 1 else 0.0,
                   "correct": sum(r["is_correct"] == "True" for r in groups[label])}
            if label != labels[0]:
                diffs = np.array([float(r[key]) - float(baseline[r["sample_idx"]][key]) for r in groups[label]])
                baseline_sd = stdev(float(r[key]) for r in groups[labels[0]])
                margin = margin_sd * baseline_sd
                p_equiv, low, high = equivalence_test(diffs, margin)
                row.update(mean_diff=float(diffs.mean()), p_value=sign_flip_p_value(diffs), margin=margin,
                           ci90_low=low, ci90_high=high, p_equiv=p_equiv, baseline_sd=baseline_sd,
                           se_diff=float(diffs.std(ddof=1) / math.sqrt(len(diffs))))
            out.append(row)
    return out


def pooled_equivalence(summary: list[dict], margin_sd: float) -> list[dict]:
    """Fixed-effect pool of standardized differences (difference / baseline SD) per sampler and metric, with TOST."""
    pooled = []
    for family, _ in FAMILIES:
        for key, _ in PANEL_SPECS:
            rows = [r for r in summary if r["family"] == family and r["metric"] == key and "p_value" in r]
            if not rows:
                continue
            z = np.array([r["mean_diff"] / r["baseline_sd"] for r in rows])
            weights = np.array([(r["baseline_sd"] / r["se_diff"]) ** 2 for r in rows])
            mean = float((weights * z).sum() / weights.sum())
            se = float(weights.sum() ** -0.5)
            p_equiv = max(1 - normal_cdf((mean + margin_sd) / se), normal_cdf((mean - margin_sd) / se))
            pooled.append({"family": family, "metric": key, "pairs": len(rows), "std_diff": mean,
                           "ci90_low": mean - 1.645 * se, "ci90_high": mean + 1.645 * se, "margin_sd": margin_sd,
                           "p_equiv": p_equiv})
    return pooled


def format_p(value: float) -> str:
    return r"$<\!0.001$" if value < 0.001 else f"{value:.3f}"


def latex_table(family: str, summary: list[dict], pooled: list[dict]) -> str:
    """The tabular for one sampler: mean_sd per method, Holm-adjusted difference test, and equivalence test."""
    baseline, presto = {"uniform": ("PowerMH", r"\method"), "entropy": ("EntropyCut", r"\methodEnt")}[family]
    rows = [r for r in summary if r["family"] == family]
    pairs = list(dict.fromkeys((r["dataset"], r["model"]) for r in rows))
    lines = [r"\begin{tabular}{llr cccc cccc}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Token log-likelihood} & \multicolumn{4}{c}{Confidence} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
             rf"Dataset & Base LLM & $n$ & {baseline} & {presto} & $p_{{\mathrm{{diff}}}}$ & $p_{{\mathrm{{eq}}}}$"
             rf" & {baseline} & {presto} & $p_{{\mathrm{{diff}}}}$ & $p_{{\mathrm{{eq}}}}$ \\", r"\midrule"]
    for dataset, model in pairs:
        cells = []
        n = None
        dagger = ""
        for key, _ in PANEL_SPECS:
            base, pre = [r for r in rows if (r["dataset"], r["model"], r["metric"]) == (dataset, model, key)]
            n = base["n"]
            if re.search(r"=10\b", pre["method"]):
                dagger = r"$^{\dagger}$"
            p_eq = format_p(pre["p_equiv"])
            cells += [rf"${base['mean']:.3f}_{{\pm {base['sd']:.3f}}}$", rf"${pre['mean']:.3f}_{{\pm {pre['sd']:.3f}}}$",
                      format_p(pre["p_holm"]), rf"\textbf{{{p_eq}}}" if pre["p_equiv"] < 0.05 else p_eq]
        lines.append(f"{DATASET_DISPLAY.get(dataset, dataset)}{dagger} & {MODEL_DISPLAY.get(model, model)} & {n} & "
                     + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    pooled_cells = []
    for key, _ in PANEL_SPECS:
        entry = next(e for e in pooled if e["family"] == family and e["metric"] == key)
        p_eq = format_p(entry["p_equiv"])
        pooled_cells += [rf"\multicolumn{{3}}{{c}}{{$\Delta/\mathrm{{SD}}={entry['std_diff']:+.3f}$ "
                         rf"$[{entry['ci90_low']:+.3f}, {entry['ci90_high']:+.3f}]$}}",
                         rf"\textbf{{{p_eq}}}" if entry["p_equiv"] < 0.05 else p_eq]
    lines.append(rf"Pooled & \multicolumn{{2}}{{l}}{{{len(pairs)} pairs}} & " + " & ".join(pooled_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def draw_family(family: str, summary: list[dict], path: Path) -> Path:
    """Rows are pairs, columns are the two metrics; markers show means with 95% intervals."""
    rows = [r for r in summary if r["family"] == family]
    pairs = list(dict.fromkeys((r["dataset"], r["model"]) for r in rows))
    # Baseline first, then PreSTO by prefetch budget, so colours and legend order match across figures.
    all_methods = sorted(dict.fromkeys(r["method"] for r in rows),
                         key=lambda m: (m.startswith("PreSTO"), int(re.search(r"=(\d+)", m).group(1)) if "=" in m else 0))
    colors = group_colors(all_methods)
    figure, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(7.2, 0.34 * len(pairs) + 1.2), sharey=True,
                                layout="constrained")
    for column, (ax, (key, label)) in enumerate(zip(axes, PANEL_SPECS)):
        for method_index, method in enumerate(all_methods):
            points = [(i, r) for i, pair in enumerate(pairs) for r in rows
                      if (r["dataset"], r["model"]) == pair and r["method"] == method and r["metric"] == key]
            offset = (method_index - (len(all_methods) - 1) / 2) * 0.22
            ax.errorbar([r["mean"] for _, r in points], [i + offset for i, _ in points],
                        xerr=[1.96 * r["sd"] / math.sqrt(r["n"]) for _, r in points], fmt=MARKERS[method_index],
                        color=colors[method], markersize=3.5, elinewidth=0.9, capsize=1.5,
                        label=method if column == 0 else None)
        for i in range(len(pairs) - 1):
            ax.axhline(i + 0.5, color=GRAY, linewidth=0.3)
        ax.set_xlabel(label)
        add_panel_label(ax, f"({chr(ord('a') + column)})")
    n_by_pair = {(r["dataset"], r["model"]): r["n"] for r in rows}
    axes[0].set_yticks(range(len(pairs)),
                       [f"{DATASET_DISPLAY.get(d, d)} / {MODEL_DISPLAY.get(m, m)} ($n$={n_by_pair[(d, m)]})"
                        for d, m in pairs])
    axes[0].set_ylim(len(pairs) - 0.5, -0.5)
    for ax in axes:
        ax.tick_params(axis="y", which="minor", left=False, right=False)
    figure.legend(loc="outside upper center", ncols=len(all_methods), frameon=False)
    return save_figure(figure, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--launcher", type=Path, default=LAUNCHER, help="Launcher whose jobs define the pairs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-exact", type=int, default=10,
                        help="Drop pairs with fewer prompts exact in every run.")
    parser.add_argument("--agreement-tol", type=float, default=0.01,
                        help="Largest |rescored - logged| token log-likelihood that counts as agreement.")
    parser.add_argument("--max-disagreement", type=float, default=0.05,
                        help="Drop a (dataset, model) whose share of disagreeing rows exceeds this.")
    parser.add_argument("--equivalence-margin", type=float, default=0.2,
                        help="Equivalence margin in units of the baseline's across-prompt standard deviation.")
    return parser


def kept_pairs(jobs: list[tuple[str, str, list[Path]]], agreement: dict[tuple[str, str], dict], min_exact: int,
               max_disagreement: float) -> list[tuple[str, str, str, dict[str, csv_rows], dict | None]]:
    """Return (family, dataset, model, exact groups, rescoring check) for every pair that passes all checks."""
    kept = []
    for dataset, model, paths in jobs:
        family = next(name for name, prefix in FAMILIES if str(paths[0].relative_to(CASE_STUDIES_DIR)).startswith(prefix))
        name = f"{family} {dataset} {model}"
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"skip {name}: not rescored ({missing[0].name})")
            continue
        check = agreement.get((dataset, model))
        if check and check["disagree_fraction"] > max_disagreement:
            print(f"drop {name}: rescoring disagrees with logged values on {check['disagree']}/{check['checked']} rows")
            continue
        groups = read_groups(paths)
        check_same_prompts(groups, name)
        exact = restrict_to_exact(groups)
        n_exact = len(next(iter(exact.values())))
        if n_exact < min_exact:
            print(f"drop {name}: {n_exact} prompts exact in every run (< {min_exact})")
            continue
        kept.append((family, dataset, model, exact, check))
    return kept


def main() -> None:
    args = build_parser().parse_args()
    summary: list[dict] = []
    jobs = launcher_jobs(args.launcher)
    agreement = rescoring_agreement(jobs, args.agreement_tol)
    write_dict_rows(args.output_dir / "rescoring-agreement.csv", sorted(agreement.values(), key=lambda e: (e["dataset"], e["model"])))
    for family, dataset, model, exact, check in kept_pairs(jobs, agreement, args.min_exact, args.max_disagreement):
        exact = one_presto_budget(exact)
        prefix = output_prefix(exact, False, args.output_dir / "pairs")
        plot(exact, 12, prefix.with_name(prefix.name + ".pdf"),
             title=f"{DATASET_DISPLAY.get(dataset, dataset)} / {MODEL_DISPLAY.get(model, model)}")
        for row in summarize_pair(family, dataset, model, exact, args.equivalence_margin):
            row["rescoring_check"] = f"{check['checked'] - check['disagree']}/{check['checked']} agree" if check else "no reference"
            summary.append(row)

    tested = [r for r in summary if "p_value" in r]
    for row, adjusted in zip(tested, holm([r["p_value"] for r in tested])):
        row["p_holm"] = adjusted
    columns = ["family", "dataset", "model", "method", "metric", "n", "correct", "mean", "sd", "mean_diff", "p_value",
               "p_holm", "margin", "ci90_low", "ci90_high", "p_equiv", "rescoring_check"]
    write_dict_rows(args.output_dir / "likelihood-confidence-summary.csv",
                    [{c: r.get(c, "") for c in columns} for r in summary], fieldnames=columns)
    pooled = pooled_equivalence(summary, args.equivalence_margin)
    write_dict_rows(args.output_dir / "likelihood-confidence-pooled-equivalence.csv", pooled)
    for family, _ in FAMILIES:
        if any(r["family"] == family for r in summary):
            table = args.output_dir / f"likelihood-confidence-table.{family}-cut.tex"
            table.write_text(latex_table(family, summary, pooled))
            print(table)
    for family, _ in FAMILIES:
        if any(r["family"] == family for r in summary):
            print(draw_family(family, summary, args.output_dir / f"likelihood-confidence-summary.{family}-cut.pdf"))
    for r in tested:
        print(f"{r['family']:8s} {r['dataset']:10s} {r['model']:11s} {r['method'][:22]:22s} {r['metric']:15s} "
              f"n={r['n']:2d} diff={r['mean_diff']:+.4f} p={r['p_value']:.3f} holm={r['p_holm']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test whether vLLM and SGLang give equivalent rescored log-likelihood and confidence for the same sampler.

The engine jobs of score_log_likelihood_and_confidence.sh rescore each sampler (PowerMH, PreSTO-PowerMH) run once on
vLLM and once on SGLang with the same settings and prompts (case_studies/compare_vllm_sglang/). Under p_0 every response
has a token log-likelihood (1/n) log p_0(x | x_0) and a confidence (1/n) sum_t sum_u p_t(u) log p_t(u). For each job
this script

* checks that all runs were scored on the same prompts (identical per-prompt token counts);
* keeps, per sampler, the prompts whose rows are exact on both engines (see draw_rescored_likelihood_and_confidence.py)
  and skips a sampler with fewer than --min-exact of them;
* reports n, mean, and standard deviation per engine, and for the per-prompt difference SGLang minus vLLM: an exact
  two-sided sign-flip test of no difference, Holm-corrected over every test here, and a paired equivalence test (TOST,
  t-based) with margin --equivalence-margin times vLLM's across-prompt standard deviation, with its 90% interval. The
  engines are equivalent at level 0.05 when that interval lies inside (-margin, margin);
* pools each metric's standardized differences (difference / vLLM SD) over the samplers with a fixed-effect average
  and tests the pooled difference for equivalence against the same margin;
* draws, per sampler, the two-panel histogram of both engines;
* within each engine, runs the same paired tests on PreSTO-PowerMH minus PowerMH (margin from PowerMH's SD), to tell
  an engine effect from a sampler effect.

The engines run independent chains, so the per-prompt differences carry sampling noise as well as any engine effect:
TOST asks whether the mean difference is small next to the spread across prompts, not whether the samples coincide.

Outputs go to --output-dir: engine-equivalence.csv, engine-equivalence-pooled.csv, engine-equivalence-table.tex,
within-engine-equivalence.csv, and one histogram PDF per job and sampler.

Run from the repository root once the launcher's engine jobs (GROUP=engines) have written their .rescored.csv files:

    PYTHONPATH=$PWD src/.venv/bin/python -m case_studies.score_log_likelihood_and_confidence.compare_engines
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

from case_studies import paths
from case_studies.draw.common.figure_io import write_dict_rows
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence import (
    plot,
    read_run,
    restrict_to_exact,
)
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence_summary import (
    DATASET_DISPLAY,
    LAUNCHER,
    MODEL_DISPLAY,
    equivalence_test,
    format_p,
    holm,
    launcher_jobs,
    normal_cdf,
    sign_flip_p_value,
)
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import PANEL_SPECS
from case_studies.plot_config import POWER_METHOD_LABEL

DEFAULT_OUTPUT_DIR = paths.CASE_STUDIES_DIR / "score_log_likelihood_and_confidence/figures/engines"
# Launcher jobs whose runs live here are the engine comparisons.
ENGINE_PREFIX = "logs-compare-vllm-sglang/"
# The reference engine first: differences are SGLang minus vLLM, and the margin scales with vLLM's spread.
ENGINES = ("vllm", "sglang")
ENGINE_LABELS = {"vllm": "vLLM", "sglang": "SGLang"}
SAMPLER_LABELS = {"power_sampling_mcmc": POWER_METHOD_LABEL, "subtree_prefetching_MH": f"PreSTO-{POWER_METHOD_LABEL}"}
csv_rows = list[dict]


def read_job(csv_paths: list[Path]) -> dict[str, dict[str, csv_rows]]:
    """Return rescored rows by sampler, then engine; a run's engine is the directory it was written to."""
    runs: dict[str, dict[str, csv_rows]] = {}
    for path in csv_paths:
        rows = read_run(path)
        engine = path.parent.name
        if not rows or engine not in ENGINES:
            raise SystemExit(f"{path}: expected a non-empty run under a {' or '.join(ENGINES)} directory")
        sampler = SAMPLER_LABELS[rows[0]["method"]]
        if engine in runs.setdefault(sampler, {}):
            raise SystemExit(f"two {engine} runs of {sampler}: {path}")
        runs[sampler][engine] = rows
    for sampler, by_engine in runs.items():
        if set(by_engine) != set(ENGINES):
            raise SystemExit(f"{sampler} needs one run per engine, got {sorted(by_engine)}")
        runs[sampler] = {engine: by_engine[engine] for engine in ENGINES}
    return runs


def check_same_prompts(runs: dict[str, dict[str, csv_rows]], name: str) -> None:
    """Fail unless every run was scored on the same prompts."""
    lengths = {(sampler, engine): [row["prompt_tokens"] for row in rows]
               for sampler, by_engine in runs.items() for engine, rows in by_engine.items()}
    reference = next(iter(lengths.values()))
    for (sampler, engine), values in lengths.items():
        if values != reference:
            raise SystemExit(f"{name}: {sampler} on {engine} was scored on different prompts")


def compare(dataset: str, model: str, sampler: str, by_engine: dict[str, csv_rows], margin_sd: float) -> list[dict]:
    """One row per metric: both engines' mean and SD, and the tests on the per-prompt difference."""
    reference, other = (by_engine[engine] for engine in ENGINES)
    reference_by_prompt = {row["sample_idx"]: row for row in reference}
    out = []
    for key, _ in PANEL_SPECS:
        row = {"dataset": dataset, "model": model, "sampler": sampler, "metric": key, "n": len(reference)}
        for engine, rows in by_engine.items():
            values = [float(r[key]) for r in rows]
            row.update({f"{engine}_mean": fmean(values), f"{engine}_sd": stdev(values)})
        diffs = np.array([float(r[key]) - float(reference_by_prompt[r["sample_idx"]][key]) for r in other])
        margin = margin_sd * row[f"{ENGINES[0]}_sd"]
        p_equiv, low, high = equivalence_test(diffs, margin)
        row.update(mean_diff=float(diffs.mean()), se_diff=float(diffs.std(ddof=1) / math.sqrt(len(diffs))),
                   p_value=sign_flip_p_value(diffs), margin=margin, ci90_low=low, ci90_high=high, p_equiv=p_equiv)
        out.append(row)
    return out


def within_engine(dataset: str, model: str, runs: dict[str, dict[str, csv_rows]], margin_sd: float) -> list[dict]:
    """Per engine and metric: the paired tests on PreSTO-PowerMH minus PowerMH over prompts exact in both runs."""
    baseline, presto = SAMPLER_LABELS["power_sampling_mcmc"], SAMPLER_LABELS["subtree_prefetching_MH"]
    out = []
    for engine in ENGINES:
        exact = restrict_to_exact({baseline: runs[baseline][engine], presto: runs[presto][engine]})
        baseline_by_prompt = {row["sample_idx"]: row for row in exact[baseline]}
        for key, _ in PANEL_SPECS:
            baseline_sd = stdev(float(r[key]) for r in exact[baseline])
            diffs = np.array([float(r[key]) - float(baseline_by_prompt[r["sample_idx"]][key]) for r in exact[presto]])
            p_equiv, low, high = equivalence_test(diffs, margin_sd * baseline_sd)
            out.append({"dataset": dataset, "model": model, "engine": engine, "metric": key, "n": len(diffs),
                        "baseline_mean": fmean(float(r[key]) for r in exact[baseline]), "baseline_sd": baseline_sd,
                        "presto_mean": fmean(float(r[key]) for r in exact[presto]), "mean_diff": float(diffs.mean()),
                        "margin": margin_sd * baseline_sd, "ci90_low": low, "ci90_high": high, "p_equiv": p_equiv,
                        "p_value": sign_flip_p_value(diffs)})
    return out


def pooled_equivalence(summary: list[dict], margin_sd: float) -> list[dict]:
    """Fixed-effect pool of standardized differences (difference / vLLM SD) per metric, with TOST."""
    pooled = []
    for key, _ in PANEL_SPECS:
        rows = [r for r in summary if r["metric"] == key]
        if not rows:
            continue
        reference_sd = np.array([r[f"{ENGINES[0]}_sd"] for r in rows])
        z = np.array([r["mean_diff"] for r in rows]) / reference_sd
        weights = (reference_sd / np.array([r["se_diff"] for r in rows])) ** 2
        mean = float((weights * z).sum() / weights.sum())
        se = float(weights.sum() ** -0.5)
        p_equiv = max(1 - normal_cdf((mean + margin_sd) / se), normal_cdf((mean - margin_sd) / se))
        pooled.append({"metric": key, "comparisons": len(rows), "std_diff": mean, "ci90_low": mean - 1.645 * se,
                       "ci90_high": mean + 1.645 * se, "margin_sd": margin_sd, "p_equiv": p_equiv})
    return pooled


def latex_table(summary: list[dict], pooled: list[dict]) -> str:
    """Mean_sd per engine, Holm-adjusted difference test, and equivalence test, for both metrics."""
    vllm, sglang = (ENGINE_LABELS[engine] for engine in ENGINES)
    lines = [r"\begin{tabular}{lllr cccc cccc}", r"\toprule",
             r" & & & & \multicolumn{4}{c}{Token log-likelihood} & \multicolumn{4}{c}{Confidence} \\",
             r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
             rf"Dataset & Base LLM & Sampler & $n$ & {vllm} & {sglang} & $p_{{\mathrm{{diff}}}}$ & $p_{{\mathrm{{eq}}}}$"
             rf" & {vllm} & {sglang} & $p_{{\mathrm{{diff}}}}$ & $p_{{\mathrm{{eq}}}}$ \\", r"\midrule"]
    for dataset, model, sampler in dict.fromkeys((r["dataset"], r["model"], r["sampler"]) for r in summary):
        cells = []
        rows = {r["metric"]: r for r in summary if (r["dataset"], r["model"], r["sampler"]) == (dataset, model, sampler)}
        for key, _ in PANEL_SPECS:
            row = rows[key]
            p_eq = format_p(row["p_equiv"])
            cells += [rf"${row[f'{engine}_mean']:.3f}_{{\pm {row[f'{engine}_sd']:.3f}}}$" for engine in ENGINES]
            cells += [format_p(row["p_holm"]), rf"\textbf{{{p_eq}}}" if row["p_equiv"] < 0.05 else p_eq]
        lines.append(f"{DATASET_DISPLAY.get(dataset, dataset)} & {MODEL_DISPLAY.get(model, model)} & {sampler} & "
                     f"{rows[PANEL_SPECS[0][0]]['n']} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    pooled_cells = []
    for key, _ in PANEL_SPECS:
        entry = next(e for e in pooled if e["metric"] == key)
        p_eq = format_p(entry["p_equiv"])
        pooled_cells += [rf"\multicolumn{{3}}{{c}}{{$\Delta/\mathrm{{SD}}={entry['std_diff']:+.3f}$ "
                         rf"$[{entry['ci90_low']:+.3f}, {entry['ci90_high']:+.3f}]$}}",
                         rf"\textbf{{{p_eq}}}" if entry["p_equiv"] < 0.05 else p_eq]
    lines.append(rf"Pooled & \multicolumn{{3}}{{l}}{{{pooled[0]['comparisons']} comparisons}} & "
                 + " & ".join(pooled_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--launcher", type=Path, default=LAUNCHER, help="Launcher whose engine jobs to compare.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-exact", type=int, default=10,
                        help="Skip a sampler with fewer prompts exact on both engines.")
    parser.add_argument("--equivalence-margin", type=float, default=0.2,
                        help="Equivalence margin in units of vLLM's across-prompt standard deviation.")
    parser.add_argument("--bins", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    jobs = launcher_jobs(args.launcher, (ENGINE_PREFIX,))
    if not jobs:
        raise SystemExit(f"{args.launcher} has no jobs under {ENGINE_PREFIX}")
    summary: list[dict] = []
    within: list[dict] = []
    for dataset, model, csv_paths in jobs:
        name = f"{dataset} {model}"
        missing = [p for p in csv_paths if not p.exists()]
        if missing:
            print(f"skip {name}: not rescored ({missing[0].relative_to(paths.CASE_STUDIES_DIR)})")
            continue
        runs = read_job(csv_paths)
        check_same_prompts(runs, name)
        if len(runs) == len(SAMPLER_LABELS):
            within += within_engine(dataset, model, runs, args.equivalence_margin)
        for sampler, by_engine in runs.items():
            exact = restrict_to_exact(by_engine)
            n_exact = len(exact[ENGINES[0]])
            if n_exact < args.min_exact:
                print(f"skip {name} {sampler}: {n_exact} prompts exact on both engines (< {args.min_exact})")
                continue
            title = f"{DATASET_DISPLAY.get(dataset, dataset)} / {MODEL_DISPLAY.get(model, model)}"
            slug = sampler.lower().replace(" ", "-")
            print(plot({f"{sampler} ({ENGINE_LABELS[engine]})": rows for engine, rows in exact.items()}, args.bins,
                       args.output_dir / f"dataset-{dataset}.model-{model}.{slug}.vllm-vs-sglang.pdf", title=title))
            summary += compare(dataset, model, sampler, exact, args.equivalence_margin)
    if not summary:
        raise SystemExit("nothing to compare yet: run the launcher with GROUP=engines first")

    for row, adjusted in zip(summary, holm([r["p_value"] for r in summary])):
        row["p_holm"] = adjusted
    print(write_dict_rows(args.output_dir / "engine-equivalence.csv", summary))
    pooled = pooled_equivalence(summary, args.equivalence_margin)
    print(write_dict_rows(args.output_dir / "engine-equivalence-pooled.csv", pooled))
    table = args.output_dir / "engine-equivalence-table.tex"
    table.write_text(latex_table(summary, pooled))
    print(table)
    for r in summary:
        print(f"{r['dataset']:8s} {r['model']:11s} {r['sampler']:15s} {r['metric']:15s} n={r['n']:2d} "
              f"diff={r['mean_diff']:+.4f} 90% CI [{r['ci90_low']:+.4f}, {r['ci90_high']:+.4f}] "
              f"margin=±{r['margin']:.4f} p_eq={r['p_equiv']:.3f} p_holm={r['p_holm']:.3f}")
    if within:
        print(write_dict_rows(args.output_dir / "within-engine-equivalence.csv", within))
    for r in within:
        print(f"within {r['engine']:6s} {r['metric']:15s} n={r['n']:2d} PreSTO-PowerMH diff={r['mean_diff']:+.4f} "
              f"90% CI [{r['ci90_low']:+.4f}, {r['ci90_high']:+.4f}] margin=±{r['margin']:.4f} p_eq={r['p_equiv']:.3f} "
              f"p_diff={r['p_value']:.3f}")
    for e in pooled:
        print(f"pooled {e['metric']:15s} Δ/SD={e['std_diff']:+.3f} 90% CI [{e['ci90_low']:+.3f}, {e['ci90_high']:+.3f}] "
              f"margin=±{e['margin_sd']} p_eq={e['p_equiv']:.3f}")


if __name__ == "__main__":
    main()

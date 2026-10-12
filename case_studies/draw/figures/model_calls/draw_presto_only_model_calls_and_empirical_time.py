#!/usr/bin/env python3
"""Render complete PreSTO-only Gemma runs and package their PDFs.

Run with:
    MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    XDG_CACHE_HOME=/private/tmp/power-sharpening-xdg-cache \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_presto_only_model_calls_and_empirical_time.py

Keep the reference three-panel style and use all 20 complete traces. LCB is
excluded because the available runs contain V5 data.
Use the standard comparison filenames so paired figures can replace them later.

A pair whose PowerMH baseline has since finished is skipped and recorded under
``superseded`` in the manifest: its paired comparison figure beside the source
logs is the current figure, and any stale PreSTO-only copy here is removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from case_studies import paths
from case_studies.draw.figures.model_calls import draw_accept_first_power_mh_model_calls_and_empirical_time as combined
from case_studies.draw.figures.model_calls.draw_lcb_v6_available_model_call_time import single_method_figure
from case_studies.extract.logs.model_call_traces import extract_model_call_traces

ROOT = paths.REPO_ROOT
DATASETS = ("math500", "aime", "mbpp", "gpqa", "human_eval", "mmlu")
MODELS = ("gemma-12b-it",)
DEFAULT_OUTPUT = ROOT / "case_studies/presto-only-gemma-12b-it-figures"


def complete_run(path: Path) -> bool:
    """Accept only a finished run whose 20 traces all reach 100 MH transitions."""
    if "INFO: output saved to" not in path.read_text(errors="replace"):
        return False
    traces = extract_model_call_traces(path)
    return len(traces) == 20 and all(t.complete and t.total_mh_steps == 100 for t in traces)


def latest_complete_log(dataset: str, model: str) -> Path:
    """Select the newest dated full run with the requested shared settings."""
    name = (
        f"dataset-{dataset}.model-{model}.alpha4.0.steps100.blocks-1."
        "prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024."
        "seed10086.subtreePrefetch.vllm.log"
    )
    for path in sorted((ROOT / "case_studies/logs" / dataset).glob(f"*/vllm/{name}"), reverse=True):
        if complete_run(path):
            return path
    raise ValueError(f"No complete 20-question PreSTO run for {dataset}/{model}")


def matching_power_mh_log(dataset: str, model: str, subtree: Path) -> Path | None:
    """Find a finished PowerMH baseline this subtree run may legitimately be drawn against.

    A pair is usable only when the baseline run finished and shares the subtree run's configuration, so a crashed or
    differently configured PowerMH log leaves the pair PreSTO-only rather than being overlaid.
    """
    name = (
        f"dataset-{dataset}.model-{model}.alpha4.0.steps100.blocks-1."
        "samples20.maxnew1024.seed10086.powerMH.vllm.log"
    )
    for path in sorted((ROOT / "case_studies/logs" / dataset).glob(f"*/vllm/{name}"), reverse=True):
        if complete_run(path) and combined._power_mh_config_matches(subtree, path):
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    sources = [(dataset, model, latest_complete_log(dataset, model)) for model in MODELS for dataset in DATASETS]
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    superseded = []
    for dataset, model, source in sources:
        # A pair that has since gained a finished, configuration-matched PowerMH run belongs in the paired comparison
        # figure beside its logs, not in this PreSTO-only set.
        baseline = matching_power_mh_log(dataset, model, source)
        if baseline is not None:
            paired = combined.default_output_path(
                combined.single_budget_comparison(source, traces_per_rank=10)
            )
            stale = output_directory / paired.name
            if stale.exists():
                stale.unlink()
            superseded.append({
                "dataset": dataset, "model": model,
                "power_mh_log": str(baseline), "paired_figure": str(paired),
            })
            print(f"Superseded by paired figure: {dataset}/{model} -> {paired}", flush=True)
            continue
        comparison = combined.single_budget_comparison(source, traces_per_rank=20)
        analysis = comparison.batch_analyses[0][1]
        assert analysis.traces_per_rank == 20
        output = output_directory / combined.default_output_path(comparison).name
        note = single_method_figure(
            comparison, None, source, output,
            note="PreSTO-PowerMH only: 20 complete traces; 100 MH transitions.",
        )
        records.append({
            "dataset": dataset, "model": model, "method": "PreSTO-PowerMH",
            "source_log": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "pdf": str(output), "traces_plotted": 20, "mh_transitions": 100,
            "prefetch_budget": 20, "rank_fn": "bfs_accept_first",
            "max_new_tokens": 1024, "note": note,
        })
        print(f"Wrote {output}", flush=True)
    archive = output_directory.parent / f"{output_directory.name}.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for record in records:
            path = Path(record["pdf"])
            bundle.write(path, f"{output_directory.name}/{path.name}")
    manifest = {
        "description": "PreSTO-only model calls and empirical time; no PowerMH timing baseline.",
        "figures": records,
        "superseded": superseded,
        "superseded_note": (
            "These pairs now have a finished, configuration-matched PowerMH run, so their paired comparison figure "
            "beside the source logs replaces the PreSTO-only version, which is no longer written here."
        ),
        "excluded": {
            "lcb_v6": "Available logs use V5 problems, excluded as requested.",
        },
        "archive": {"path": str(archive), "pdf_count": len(records), "nested_subfolders": False},
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {archive}", flush=True)


if __name__ == "__main__":
    main()

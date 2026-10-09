"""Collect total sampling time per method from the standard-configuration timing runs.

Run from the repository root:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening \
        python -m case_studies.scaling_baselines.collect_total_time [--logs-dir DIR] [--output FILE]

Reads every ``logs_total_time/<dataset>/<date>/vllm/*.log`` written by ``total_time.standard_config.punakha.sh`` and
writes one row per run to ``total_time.csv``. Each runner logs one ``<sampler> took <s> seconds for <i>-th prompts``
line per batch (one prompt per batch here); a run's total time is the sum of those lines, so it covers generation only,
not model loading or grading. The methods are told apart by the log suffix:

    low_temp   .low_temp.vllm.log          one sample at the low temperature
    best_of_n  .best_of_n.vllm.log         N samples (setting = N)
    power_mh   .powerMH.vllm.log           blockwise PowerMH
    presto     .subtreePrefetch.vllm.log   PreSTO-PowerMH (setting = rank and prefetch budget)
    power_smc  .smc.vllm.log               Power-SMC on the custom engine (setting = number of particles)
    power_smc_standard  .smc.vllm.log      Power-SMC on stock vLLM (run name has .power_smc_standard.)

PowerMH and PreSTO settings start with the MH configuration (``steps10.blocks-8``). ``vs_power_mh`` divides each MH
run's total by the PowerMH total with the same MH configuration, and every other run's total by the standard
``steps10.blocks-8`` PowerMH, for the same dataset, model, and date. Accuracy comes from the
run's CSV and is empty while a run is still going.

KV-cache usage comes from the run's ``<run>.resources.json`` (written when the resource probe is on):

    peak_kv_cache_occupancy      run peak fraction of the KV-block pool in use
    peak_kv_cache_gib            that peak in GiB (pool size x occupancy); kv_pool_gib is the pool size
    mean_prompt_peak_kv          mean over prompts of each prompt's peak occupancy (from the CSV's per-prompt windows)
    kv_cache_eviction_rate       cached blocks evicted / blocks allocated (exact only with KV_CACHE_MODE=hooks)
    prefix_cache_hit_rate        prefix-cache hit tokens / queried tokens
    peak_demand_gib              peak GPU memory with the idle part of the preallocated KV pool taken out
    kv_source                    scheduler-hook (exact) or metrics-poll (sampled; can miss short peaks)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCALING_DIR = Path(__file__).resolve().parent
LOG_SUFFIX = {
    ".low_temp.vllm.log": "low_temp",
    ".best_of_n.vllm.log": "best_of_n",
    ".powerMH.vllm.log": "power_mh",
    ".subtreePrefetch.vllm.log": "presto",
    ".smc.vllm.log": "power_smc",
}
TOOK = re.compile(
    r"(?:low_temp_sampler|best_of_n_sampler|mcmc_power_sampler|subtree_prefetching_sampling|smc_power_sampler) took "
    r"(?P<seconds>[\d.]+) seconds for (?P<sample>\d+)-th prompts"
)
RUN_NAME = re.compile(r"dataset-(?P<dataset>[^.]+)\.model-(?P<model>.+?)\.(?:low_temp|best_of_n|alpha|power_smc)")
SETTING = {
    "best_of_n": re.compile(r"\.(?P<setting>n\d+)\."),
    "presto": re.compile(r"\.prefetch-budget-(?P<budget>\d+)\.rank-(?P<rank>[^.]+)\."),
    "power_smc": re.compile(r"\.(?P<setting>nparticles\d+)\."),
    "power_smc_standard": re.compile(r"\.(?P<setting>nparticles\d+)\."),
}
MH_CONFIG = re.compile(r"\.(?P<config>steps\d+\.blocks-\d+)\.")
STANDARD_MH_CONFIG = "steps10.blocks-8"
METHOD_ORDER = ["low_temp", "best_of_n", "power_smc", "power_smc_standard", "power_mh", "presto"]
# Run-level KV-cache and memory metrics copied from <run>.resources.json.
RESOURCE_KEYS = [
    "peak_kv_cache_occupancy", "peak_kv_cache_gib", "kv_pool_gib", "kv_cache_eviction_rate", "prefix_cache_hit_rate",
    "peak_demand_gib", "kv_source",
]
COLUMNS = [
    "method", "dataset", "model", "setting", "n_prompts", "total_sec", "sec_per_prompt", "vs_power_mh", "accuracy",
    "peak_kv_cache_occupancy", "peak_kv_cache_gib", "kv_pool_gib", "mean_prompt_peak_kv", "kv_cache_eviction_rate",
    "prefix_cache_hit_rate", "peak_demand_gib", "kv_source", "run_date", "source",
]


def _setting(method: str, run_name: str) -> str:
    """Return the per-run knob that tells runs of one method apart (N for best-of-N, rank/budget for PreSTO)."""
    if method in ("power_mh", "presto"):
        config = MH_CONFIG.search(run_name)
        prefix = config["config"] if config else ""
        match = SETTING["presto"].search(run_name) if method == "presto" else None
        return f"{prefix}.{match['rank']}.p{match['budget']}" if match else prefix
    match = SETTING[method].search(run_name) if method in SETTING else None
    if match is None:
        return ""
    return match["setting"]


def collect(logs_dir: Path) -> pd.DataFrame:
    """Return one timing row per run found under ``logs_dir``."""
    rows = []
    for path in sorted(logs_dir.glob("*/*/vllm/*.log")):
        suffix = next((s for s in LOG_SUFFIX if path.name.endswith(s)), None)
        if suffix is None:
            continue
        method = LOG_SUFFIX[suffix]
        run_name = path.name[: -len(suffix)]
        if method == "power_smc" and ".power_smc_standard." in run_name:
            method = "power_smc_standard"
        match = RUN_NAME.match(run_name)
        if match is None:
            continue
        seconds = [float(m["seconds"]) for m in TOOK.finditer(path.read_text(errors="replace"))]
        if not seconds:
            continue
        csv_path = path.with_name(f"{run_name}.csv")
        accuracy = mean_prompt_peak_kv = float("nan")
        n_prompts = len(seconds)
        if csv_path.exists():
            frame = pd.read_csv(csv_path, usecols=lambda c: c in {"is_correct", "peak_kv_cache_occupancy"})
            if not frame.empty:
                accuracy = frame["is_correct"].astype(bool).mean()
                n_prompts = len(frame)
                if "peak_kv_cache_occupancy" in frame:
                    mean_prompt_peak_kv = frame["peak_kv_cache_occupancy"].mean()
        resources_path = path.with_name(f"{run_name}.resources.json")
        resources = json.loads(resources_path.read_text()) if resources_path.exists() else {}
        total = sum(seconds)
        rows.append({
            "method": method, "dataset": match["dataset"], "model": match["model"],
            "setting": _setting(method, run_name), "n_prompts": n_prompts, "total_sec": total,
            "sec_per_prompt": total / n_prompts, "accuracy": accuracy, "mean_prompt_peak_kv": mean_prompt_peak_kv,
            **{key: resources.get(key) for key in RESOURCE_KEYS},
            "run_date": path.parent.parent.name, "source": str(path.relative_to(logs_dir)),
        })
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    table = pd.DataFrame(rows)
    keys = ["dataset", "model", "run_date"]
    power_mh = table[table["method"] == "power_mh"]
    power_mh = dict(zip(zip(power_mh["dataset"], power_mh["model"], power_mh["run_date"], power_mh["setting"]),
                        power_mh["total_sec"]))

    def reference(row) -> float:
        config = row.setting.split(".rank")[0] if row.method in ("power_mh", "presto") else ""
        config = ".".join(row.setting.split(".")[:2]) if config else STANDARD_MH_CONFIG
        return power_mh.get((row.dataset, row.model, row.run_date, config), float("nan"))

    table["vs_power_mh"] = [row.total_sec / reference(row) for row in table.itertuples()]
    table["order"] = table["method"].map(METHOD_ORDER.index)
    table = table.sort_values(["dataset", "model", "run_date", "order", "setting"], ignore_index=True)
    return table[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs-dir", type=Path, default=SCALING_DIR / "logs_total_time")
    parser.add_argument("--output", type=Path, default=SCALING_DIR / "total_time.csv")
    args = parser.parse_args()
    table = collect(args.logs_dir)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} rows -> {args.output}")
    if not table.empty:
        shown = ["method", "dataset", "model", "setting", "n_prompts", "total_sec", "vs_power_mh", "accuracy",
                 "peak_kv_cache_occupancy", "peak_kv_cache_gib", "mean_prompt_peak_kv", "prefix_cache_hit_rate"]
        print(table[shown].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

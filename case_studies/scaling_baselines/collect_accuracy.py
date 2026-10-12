"""Collect accuracy against the scaling axis from the scaling-baseline run CSVs.

Run from the repository root:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening \
        python -m case_studies.scaling_baselines.collect_accuracy [--logs-dir DIR] [--output FILE]

Reads every ``logs/<dataset>/<date>/*.csv`` written by the scripts in ``scripts/`` and writes one row per
(method, dataset, model, scaling value) to ``accuracy_scaling.csv``. The methods and their axes are:

    low_temp   temperature   (one sample per prompt)
    best_of_n  best_of_n     (best of the first n samples by sequence log-probability; one run holds the whole ladder)
    power_smc  n_particles   (Power-SMC population per prompt)

When a cell appears in more than one dated run, the run with the most graded prompts is kept (ties: the latest date).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

SCALING_DIR = Path(__file__).resolve().parent
RUN_NAME = re.compile(
    r"dataset-(?P<dataset>[^.]+)\.model-(?P<model>.+?)\.(?P<method>low_temp|best_of_n|power_smc)\."
)
# The scaling axis of each method, and the per-run settings that identify one line on a figure.
AXIS = {"low_temp": "temperature", "best_of_n": "best_of_n", "power_smc": "n_particles"}
SETTING = {"low_temp": None, "best_of_n": "temperature", "power_smc": None}
COLUMNS = [
    "method", "dataset", "model", "axis", "value", "setting", "max_new_tokens", "n_problems", "n_correct", "accuracy",
    "stderr", "run_date", "source",
]


def collect(logs_dir: Path) -> pd.DataFrame:
    """Return one accuracy row per method, dataset, model, and scaling value found under ``logs_dir``."""
    rows = []
    for path in sorted(logs_dir.glob("*/*/*.csv")):
        match = RUN_NAME.match(path.name)
        if match is None:
            continue
        method = match["method"]
        frame = pd.read_csv(path, usecols=lambda c: c in {"is_correct", "temperature", "n_particles", "best_of_n",
                                                         "max_new_tokens"})
        if frame.empty or "is_correct" not in frame:
            continue
        frame["is_correct"] = frame["is_correct"].astype(bool)
        axis = AXIS[method]
        setting = SETTING[method]
        for value, group in frame.groupby(axis):
            n = len(group)
            correct = int(group["is_correct"].sum())
            accuracy = correct / n
            rows.append({
                "method": method, "dataset": match["dataset"], "model": match["model"], "axis": axis,
                "value": float(value),
                "setting": float(group[setting].iloc[0]) if setting else math.nan,
                "max_new_tokens": int(group["max_new_tokens"].iloc[0]) if "max_new_tokens" in group else math.nan,
                "n_problems": n, "n_correct": correct, "accuracy": accuracy,
                "stderr": math.sqrt(accuracy * (1 - accuracy) / n),
                "run_date": path.parent.name, "source": str(path.relative_to(logs_dir)),
            })
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    table = pd.DataFrame(rows)
    keys = ["method", "dataset", "model", "value", "setting"]
    table = table.sort_values(["n_problems", "run_date"]).drop_duplicates(keys, keep="last", ignore_index=True)
    return table.sort_values(["method", "dataset", "model", "setting", "value"], ignore_index=True)[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs-dir", type=Path, default=SCALING_DIR / "logs")
    parser.add_argument("--output", type=Path, default=SCALING_DIR / "accuracy_scaling.csv")
    args = parser.parse_args()
    table = collect(args.logs_dir)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} rows -> {args.output}")
    if not table.empty:
        print(table.groupby("method").size().to_string())


if __name__ == "__main__":
    main()

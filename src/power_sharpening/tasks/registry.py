"""Shared benchmark-runner configuration for all power-sharpening runners.

Single source of truth for the pieces that every experiment runner needs,
across both the HuggingFace and vLLM backends:

* the ``TASKS`` registry and ``build_benchmark`` factory,
* the benchmark-selection CLI (``--task`` / ``--batch_size``) plus the task-conditional flags (aime / lcb) via
  ``add_benchmark_args``,
* ``resolve_task`` — the pre-parser idiom that discovers ``--task`` before the task-specific flags are registered,
* reproducibility (``set_random_seed``).

Runners import from here so the config lives in one place:

    from power_sharpening.tasks.registry import (
        add_benchmark_args, add_benchmark_selection_args, build_benchmark, resolve_task, set_random_seed,
    )

    task = resolve_task()
    parser = argparse.ArgumentParser()
    add_benchmark_selection_args(parser)     # shared --task / --batch_size
    # ... runner-specific args ...
    add_benchmark_args(parser, task)         # only the chosen task's flags
    args = parser.parse_args()

Imports use relative submodule paths (not ``power_sharpening.tasks`` package-level) so this module never round-trips
through ``power_sharpening/tasks/__init__`` — no circular
import even though it lives inside the package.
"""
import argparse
import os
import random

import numpy as np
import torch

from .aime_benchmark import AIMEBenchmark
from .gpqa_benchmark import GPQABenchmark
from .human_eval_benchmark import HumanEvalBenchmark
from .lcb_benchmark import LCB_CLASSES, LiveCodeBenchBenchmark
from .math_benchmark import MATHBenchmark
from .mbpp_benchmark import MBPPBenchmark
from .mmlu_benchmark import MMLUBenchmark


TASKS = {
    "math500": MATHBenchmark,
    "aime": AIMEBenchmark,
    "aime2024": AIMEBenchmark,
    "aime2025": AIMEBenchmark,
    "aime2024-2025": AIMEBenchmark,
    "gpqa": GPQABenchmark,
    "human_eval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "lcb": LiveCodeBenchBenchmark,
    "mmlu": MMLUBenchmark,
}


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and vLLM subprocesses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_benchmark(task: str, args, model_str: str):
    """Instantiate the selected benchmark with its dataset options."""
    if task.startswith("aime"):
        return AIMEBenchmark(
            batch_size=args.batch_size,
            name=args.aime_dataset if task == "aime" else task,
        )
    if task == "human_eval":
        return HumanEvalBenchmark(
            batch_size=args.batch_size,
            model_name=args.model_str,
            model_str=model_str,
        )
    if task == "lcb":
        benchmark_cls = LCB_CLASSES[args.difficulty]
        return benchmark_cls(
            batch_size=args.batch_size,
            version=args.lcb_version,
            max_prompt_chars=args.max_prompt_chars,
        )
    return TASKS[task](batch_size=args.batch_size)


def add_benchmark_selection_args(
    parser,
    default_task: str = "math500",
    batch_size_default: int = 1,
) -> None:
    """Register the benchmark-selection args common to every runner."""
    parser.add_argument(
        "--task", type=str, default=default_task, choices=sorted(TASKS)
    )
    parser.add_argument("--batch_size", type=int, default=batch_size_default)


def add_benchmark_args(parser, task: str) -> None:
    """Register only the CLI args the selected ``task`` actually consumes.

    Keeps the flat ``--task`` interface but avoids cluttering every run with flags for other benchmarks:
    aime/lcb-specific options are added only when their task is chosen, under a titled group so ``--help`` shows which
    dataset each flag belongs to. Kept in lockstep with ``build_benchmark`` above, which reads exactly these attributes
    for the same tasks.
    """
    if task.startswith("aime"):
        group = parser.add_argument_group("aime benchmark options")
        group.add_argument(
            "--aime_dataset",
            type=str,
            default="aime2024-2025",
            help="AIME split to load (only used when --task aime).",
        )
    if task == "lcb":
        group = parser.add_argument_group("lcb (LiveCodeBench) benchmark options")
        group.add_argument(
            "--difficulty",
            type=str,
            default="all",
            choices=["all", "easy", "medium", "hard"],
            help="LiveCodeBench difficulty filter (only used when --task lcb).",
        )
        group.add_argument(
            "--lcb_version",
            type=str,
            default="release_v6",
            help="LiveCodeBench release version (only used when --task lcb).",
        )
        group.add_argument(
            "--max_prompt_chars",
            type=int,
            default=3500,
            help="Max prompt length in characters (only used when --task lcb).",
        )


def resolve_task(argv=None, default_task: str = "math500") -> str:
    """Discover ``--task`` before task-specific flags are registered.

    Uses a throwaway parser with ``add_help=False`` so ``--help`` still fires on the real parser (where it can list the
    chosen task's flags), and ``parse_known_args`` so unknown/task-specific flags don't error here.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--task", type=str, default=default_task, choices=sorted(TASKS)
    )
    return pre.parse_known_args(argv)[0].task


__all__ = [
    "TASKS",
    "set_random_seed",
    "build_benchmark",
    "add_benchmark_selection_args",
    "add_benchmark_args",
    "resolve_task",
]

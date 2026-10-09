"""Run subtree-prefetched HF Multi-Try MH with benchmark YAML settings.

Example (GPU):
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 \
      --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.hf.run_subtree_prefetching_multi_try_mh \
      --dataset math500 --algorithm subtree_prefetching_multi_try_mh \
      --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry \
      --override num_tries=4 prefetch_budget=16 'proposal_temperatures=[0.25,0.5,1.0]'
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from power_sharpening.backends.hf.samplers.subtree_prefetching_multi_try_mh import (
    subtree_prefetching_multi_try_sampling,
)
from power_sharpening.config import add_config_selection_args, resolve_config
from power_sharpening.common.prefetch_rank import RANK_FNS
from power_sharpening.runners.hf import run_multi_try_mh as baseline
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.tasks.registry import build_benchmark, set_random_seed


logger = logging.getLogger("[run_subtree_prefetching_multi_try_hf]")
REQUIRED_KEYS = baseline.REQUIRED_MULTI_TRY_KEYS + (
    "prefetch_budget", "rank_fn", "print_tree", "stop_on_eos",
)


def _validate_and_resolve(args):
    """Reject unsupported prefetched kernels before loading a model."""
    temperature = baseline._validate_and_resolve(args)
    if args.temperature_schedule_type != "const":
        raise SystemExit("Subtree Multi-Try MH requires temperature_schedule_type='const'.")
    if args.rank_fn not in RANK_FNS:
        raise SystemExit(f"Unknown rank_fn {args.rank_fn!r}; choose from {sorted(RANK_FNS)}.")
    if (
        isinstance(args.prefetch_budget, bool)
        or not isinstance(args.prefetch_budget, int)
        or args.prefetch_budget < args.num_tries
    ):
        raise SystemExit("prefetch_budget must be an integer >= num_tries (one complete bundle).")
    for name in ("print_tree", "stop_on_eos"):
        if not isinstance(getattr(args, name), bool):
            raise SystemExit(f"{name} must be a boolean.")
    return temperature


def run_subtree_multi_try(model_str, benchmark, args, proposal_temperature):
    """Reuse benchmark formatting, context caps, grading, and output metadata."""
    def sample(wrapper, prefix, *, sample_index, context_length, **kwargs):
        tokens, stats = subtree_prefetching_multi_try_sampling(
            wrapper,
            prefix,
            **kwargs,
            prefetch_budget=args.prefetch_budget,
            rank_fn=args.rank_fn,
            print_tree=args.print_tree,
            stop_on_eos=args.stop_on_eos,
            seed=args.seed + sample_index,
            max_model_len=context_length,
        )
        target_scores = [args.alpha * value for value in stats.final_base_logprobs]
        acceptance_rate = (
            stats.total_acceptances / stats.total_walked_steps
            if stats.total_walked_steps else 0.0
        )
        return tokens, None, target_scores, acceptance_rate, stats

    return baseline.run_multi_try(
        model_str, benchmark, args, proposal_temperature,
        sample_fn=sample, method="subtree_prefetching_multi_try_mh",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Config-driven HF subtree Multi-Try MH runner.")
    add_config_selection_args(
        parser,
        algorithm_default="subtree_prefetching_multi_try_mh",
        algorithm_choices=["subtree_prefetching_multi_try_mh"],
        save_str_default="result/subtree_prefetching_multi_try/",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    task = resolve_config(args, args.algorithm, REQUIRED_KEYS)
    proposal_temperature = _validate_and_resolve(args)
    set_random_seed(args.seed)
    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(task, args, model_str)
    benchmark.should_use_chat_template(args.model_str)
    logger.info("resolved configuration: %s", vars(args))
    results, all_stats = run_subtree_multi_try(model_str, benchmark, args, proposal_temperature)
    temperatures_tag = (
        "scalar" if args.proposal_temperatures is None
        else "-".join(str(value) for value in args.proposal_temperatures)
    )
    out_stem = args.run_name or (
        f"{Path(model_str).name}_{benchmark.name}_subtree_multi_try_hf"
        f"_mcmc{args.mcmc_steps}_tries{args.num_tries}_budget{args.prefetch_budget}"
        f"_blocks{args.num_blocks}_maxnew{args.max_new_tokens}_alpha{args.alpha}"
        f"_temp{proposal_temperature}_proposals-{temperatures_tag}_rank-{args.rank_fn}"
        f"_seed{args.seed}"
    )
    output_dir = Path(args.save_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{out_stem}.csv"
    pd.DataFrame(results).to_csv(output_filename, index=False)
    stats_filename = output_dir / f"{out_stem}.stats.json"
    with stats_filename.open("w") as handle:
        json.dump(all_stats, handle, indent=2)
    logger.info("output saved to %s", output_filename)
    logger.info("stats saved to %s", stats_filename)
    return results, all_stats


if __name__ == "__main__":
    main()

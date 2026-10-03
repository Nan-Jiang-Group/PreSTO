"""Run vLLM subtree-prefetching MultiTryMH and save per-sample work statistics.

Run on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      python -m power_sharpening.runners.vllm.run_subtree_prefetching_multi_try_mh \
      --dataset math500 --algorithm subtree_prefetching_multi_try_mh \
      --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry \
      --override num_tries=4 prefetch_budget=16

One live engine processes all prompts, including every prompt in a dataset batch. Each tree belongs to one prompt.
Outputs include a CSV, per-sample ``.stats.json``, and optional run-level ``.resources.json``. Prefetch budgets count
generated suffixes; a complete MultiTry bundle costs ``num_tries``.
"""

import argparse
import json
import time
from pathlib import Path

# The baseline runner imports the shared driver before loading the backend, preserving the environment setup and
# dynamic-loader ordering for vLLM.
from power_sharpening.runners.vllm import run_multi_try_mh as baseline
from power_sharpening.backends.vllm.samplers.subtree_prefetching_multi_try_mh import (
    subtree_prefetching_multi_try_sampling,
)
from power_sharpening.common.prefetch_rank import RANK_FNS


driver = baseline.driver
REQUIRED_SUBTREE_MULTI_TRY_KEYS = baseline.REQUIRED_MULTI_TRY_KEYS + (
    "prefetch_budget", "rank_fn", "print_tree", "stop_on_eos",
)


def build_parser():
    """Build the config CLI for fixed-temperature subtree MultiTryMH."""
    parser = argparse.ArgumentParser(description="Subtree-prefetching vLLM MultiTryMH.")
    driver.add_config_selection_args(
        parser,
        algorithm_default="subtree_prefetching_multi_try_mh",
        algorithm_choices=["subtree_prefetching_multi_try_mh"],
        save_str_default="result/subtree_prefetching_multi_try_mh/",
    )
    parser.add_argument(
        "--resource_probe", action=argparse.BooleanOptionalAction, default=True,
        help="Save run-level GPU and KV-cache measurements to <run>.resources.json.",
    )
    return parser


def validate_args(args):
    """Reject unsupported cuts, changing mixtures, and incomplete bundles."""
    init_temperature = baseline.validate_args(args)
    if args.temperature_schedule_type != "const":
        raise ValueError("subtree MultiTryMH requires temperature_schedule_type=const")
    if (
        isinstance(args.prefetch_budget, bool)
        or not isinstance(args.prefetch_budget, int)
        or args.prefetch_budget < args.num_tries
    ):
        raise ValueError("prefetch_budget must be an integer >= num_tries (suffix requests)")
    for key in ("print_tree", "stop_on_eos"):
        if not isinstance(getattr(args, key), bool):
            raise ValueError(f"{key} must be a boolean")
    if args.rank_fn not in RANK_FNS:
        raise ValueError(f"rank_fn must be one of {sorted(RANK_FNS)}")
    return init_temperature


def _subtree_fields(args):
    """Metadata required to interpret suffix-budget and scoring measurements."""
    return {
        **baseline._multi_try_fields(args),
        "prefetch_budget": args.prefetch_budget,
        "prefetch_budget_units": "suffix_requests",
        "prefetch_bundle_budget": args.prefetch_budget // args.num_tries,
        "rank_fn": args.rank_fn,
        "stop_on_eos": args.stop_on_eos,
    }


def run_subtree_multi_try_sampling(
    benchmark, model_str, args, max_model_len, init_temperature,
    *, method="subtree_multi_try_mh", fields=None, draw_proposals=None, wrapper_kwargs=None,
):
    """Reuse the shared benchmark loop and engine/probe exception cleanup.

    The keyword arguments let the v2 runner swap in engine-cached proposal scoring without copying this loop.
    """
    fields = fields or _subtree_fields
    sample_records = []

    def sample_batch(mh_llm, prompts, sampling_params, scheduler, max_new_tokens):
        aligned_tokens = max_new_tokens // args.num_blocks * args.num_blocks
        if aligned_tokens <= 0:
            raise ValueError("remaining context is too short for one token per block")
        if aligned_tokens != max_new_tokens:
            driver.logger.info(
                "aligned generation budget from %d to %d tokens for %d blocks",
                max_new_tokens, aligned_tokens, args.num_blocks,
            )
        completions = []
        for prompt in prompts:
            index = len(sample_records)
            # Give each prompt a reproducible seed based on its dataset index.
            sample_seed = None if args.seed is None else (args.seed + index) % (2**63 - 1)
            params = driver.SamplingParams(
                alpha=sampling_params.alpha, temperature=sampling_params.temperature,
                seed=sample_seed,
            )
            started = time.perf_counter()
            response_ids, stats = subtree_prefetching_multi_try_sampling(
                mh_llm, list(mh_llm.tokenizer.encode(prompt)), params,
                num_of_blocks=args.num_blocks, max_new_tokens=aligned_tokens,
                mcmc_steps=args.mcmc_steps, num_tries=args.num_tries,
                proposal_temperatures=args.proposal_temperatures,
                scoring_batch_size=args.scoring_batch_size,
                prefetch_budget=args.prefetch_budget, rank_fn=args.rank_fn,
                stop_on_eos=args.stop_on_eos, print_tree=args.print_tree,
                verbose=args.verbose, draw_proposals=draw_proposals,
            )
            elapsed = time.perf_counter() - started
            record = stats.to_json(mcmc_steps=args.mcmc_steps, num_blocks=args.num_blocks)
            record.update(
                idx=index, sample_seed=sample_seed, elapsed_seconds=elapsed,
                effective_max_new_tokens=aligned_tokens,
                method=method, **fields(args),
            )
            sample_records.append(record)
            completions.append(mh_llm.tokenizer.decode(response_ids, skip_special_tokens=True))
            driver.logger.info("sample %d subtree MultiTryMH stats: %s", index, json.dumps(record))
        return completions

    results, resources = driver._run_mcmc_benchmark(
        benchmark, model_str, args, max_model_len, init_temperature,
        method=method, cut_power=None, sample_batch=sample_batch,
        wrapper_kwargs=wrapper_kwargs,
    )
    if len(results) != len(sample_records):
        raise RuntimeError("benchmark must return one grading result per sampled prompt")
    for row, record in zip(results, sample_records):
        row.update(fields(args))
        row.update({key: value for key, value in record.items() if not isinstance(value, (dict, list))})
        row["prefetch_stats"] = json.dumps(record)
    return results, resources


def _default_run_name(args, benchmark, model_str, max_model_len):
    return (
        baseline._default_run_name(args, benchmark, model_str, max_model_len)
        + f"_prefetch{args.prefetch_budget}_rank-{args.rank_fn}"
    )


def main(argv=None):
    args, results, run_name = baseline.run_benchmark_main(
        argv, parser=build_parser(), required_keys=REQUIRED_SUBTREE_MULTI_TRY_KEYS,
        validate=validate_args, run_sampling=run_subtree_multi_try_sampling,
        method="subtree_multi_try_mh", extra_fields=_subtree_fields,
        default_run_name=_default_run_name,
    )
    stats_path = Path(args.save_str) / f"{run_name}.stats.json"
    stats_path.write_text(json.dumps([json.loads(row["prefetch_stats"]) for row in results], indent=2) + "\n")
    driver.logger.info("stats saved to %s", stats_path)


if __name__ == "__main__":
    main()

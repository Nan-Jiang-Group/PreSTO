"""Run vLLM subtree-prefetching MultiTryMH v2 (engine-cached proposal scores) and save per-sample work statistics.

Run on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.vllm.run_subtree_prefetching_multi_try_mh_v2 \
      --dataset math500 --model_str qwen \
      --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry \
      --override num_tries=4 prefetch_budget=16 proposal_temperatures=[0.25,0.5,1.0]

Tree scheduling, selection, and acceptance match ``run_subtree_prefetching_multi_try_mh``. Only proposal scoring
differs: the engine is built with ``log_z_temperatures=proposal_temperatures`` and every suffix is scored at all mixture
temperatures during generation (``multi_try_mh_v2._draw_proposals_v2``), so no one-token scoring requests are made.
"""

import json
from pathlib import Path

# The v1 runner imports the shared driver before loading the backend, preserving vLLM's environment setup.
from power_sharpening.runners.vllm import run_subtree_prefetching_multi_try_mh as subtree
from power_sharpening.runners.vllm import run_multi_try_mh_v2 as standalone_v2
from power_sharpening.backends.vllm.samplers.multi_try_mh_v2 import _draw_proposals_v2


baseline = subtree.baseline
driver = subtree.driver
REQUIRED_SUBTREE_MULTI_TRY_V2_KEYS = standalone_v2.REQUIRED_MULTI_TRY_V2_KEYS + (
    "prefetch_budget", "rank_fn", "print_tree", "stop_on_eos",
)


def build_parser():
    """Build the subtree MultiTryMH CLI with the v2 algorithm selected."""
    parser = subtree.build_parser()
    parser.description = "Subtree-prefetching vLLM MultiTryMH v2 with engine-cached proposal scores."
    parser.set_defaults(
        algorithm="subtree_prefetching_multi_try_mh_v2",
        save_str="result/subtree_prefetching_multi_try_mh_v2/",
    )
    for action in parser._actions:
        if action.dest == "algorithm":
            action.choices = ["subtree_prefetching_multi_try_mh_v2"]
    return parser


def validate_args(args):
    """Apply subtree MultiTry validation, then require the explicit temperature list the engine is built with."""
    # The baseline validator checks scoring_batch_size, which v2 does not use.
    args.scoring_batch_size = getattr(args, "scoring_batch_size", None) or 1
    init_temperature = subtree.validate_args(args)
    if args.proposal_temperatures is None:
        raise ValueError("MultiTryMH v2 requires an explicit proposal_temperatures list")
    return init_temperature


def _subtree_v2_fields(args):
    fields = subtree._subtree_fields(args)
    fields.pop("scoring_batch_size", None)
    fields["proposal_scoring"] = "engine_log_z"
    return fields


def run_subtree_multi_try_v2_sampling(benchmark, model_str, args, max_model_len, init_temperature):
    """Run the shared subtree loop with engine-cached proposal scores."""
    return subtree.run_subtree_multi_try_sampling(
        benchmark, model_str, args, max_model_len, init_temperature,
        method="subtree_multi_try_mh_v2", fields=_subtree_v2_fields,
        draw_proposals=_draw_proposals_v2,
        wrapper_kwargs={"log_z_temperatures": sorted(set(map(float, args.proposal_temperatures)))},
    )


def _default_run_name(args, benchmark, model_str, max_model_len):
    return (
        standalone_v2._default_run_name(args, benchmark, model_str, max_model_len)
        + f"_prefetch{args.prefetch_budget}_rank-{args.rank_fn}"
    )


def main(argv=None):
    args, results, run_name = baseline.run_benchmark_main(
        argv, parser=build_parser(), required_keys=REQUIRED_SUBTREE_MULTI_TRY_V2_KEYS,
        validate=validate_args, run_sampling=run_subtree_multi_try_v2_sampling,
        method="subtree_multi_try_mh_v2", extra_fields=_subtree_v2_fields,
        default_run_name=_default_run_name,
    )
    stats_path = Path(args.save_str) / f"{run_name}.stats.json"
    stats_path.write_text(json.dumps([json.loads(row["prefetch_stats"]) for row in results], indent=2) + "\n")
    driver.logger.info("stats saved to %s", stats_path)


if __name__ == "__main__":
    main()

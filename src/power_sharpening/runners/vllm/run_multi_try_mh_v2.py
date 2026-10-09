"""Run the vLLM MultiTryMH v2 sampler (engine-cached proposal scores) on a configured benchmark.

Run on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.vllm.run_multi_try_mh_v2 \
      --dataset math500 --model_str qwen \
      --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry \
      --override num_tries=4 proposal_temperatures=[0.25,0.5,1.0]

Transitions match ``run_multi_try_mh``; the engine is built with ``log_z_temperatures=proposal_temperatures`` so every
suffix is scored at all mixture temperatures during generation instead of by extra one-token requests.
"""

from pathlib import Path

# Import the shared drivers first: they set the vLLM environment before loading the backend.
from power_sharpening.runners.vllm import run_power_mh as driver
from power_sharpening.runners.vllm import run_multi_try_mh as baseline
from power_sharpening.backends.vllm.samplers.multi_try_mh_v2 import (
    multi_try_mcmc_power_sampler_v2,
)


REQUIRED_MULTI_TRY_V2_KEYS = driver.REQUIRED_MCMC_KEYS + ("num_tries", "proposal_temperatures")


def build_parser():
    """Build the shared config CLI with the MultiTry v2 algorithm selected."""
    parser = baseline.build_parser()
    parser.description = "Config-driven vLLM MultiTryMH v2 runner with engine-cached proposal scores."
    parser.set_defaults(algorithm="multi_try_v2", save_str="result/multi_try_v2/")
    for action in parser._actions:
        if action.dest == "algorithm":
            action.choices = ["multi_try_v2"]
    return parser


def validate_args(args):
    """Apply MultiTry validation, then require the explicit temperature list the engine is built with."""
    # The baseline validator checks scoring_batch_size, which v2 does not use.
    args.scoring_batch_size = getattr(args, "scoring_batch_size", None) or 1
    init_temperature = baseline.validate_args(args)
    if args.proposal_temperatures is None:
        raise ValueError("MultiTryMH v2 requires an explicit proposal_temperatures list")
    return init_temperature


def _multi_try_v2_fields(args):
    return {
        "num_tries": args.num_tries,
        "proposal_temperatures": args.proposal_temperatures,
        "proposal_scoring": "engine_log_z",
    }


def run_multi_try_v2_sampling(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
    init_temperature: float,
):
    """Run MultiTryMH v2 through the shared vLLM benchmark and resource lifecycle."""
    def sample_batch(mh_llm, prompts, sampling_params, scheduler, max_new_tokens):
        # All candidates in a block share its fixed horizon, so discard a partial block left by a context cap.
        aligned_tokens = max_new_tokens // args.num_blocks * args.num_blocks
        if aligned_tokens <= 0:
            raise ValueError("remaining context is too short for one token per block")
        if aligned_tokens != max_new_tokens:
            driver.logger.info(
                "aligned generation budget from %d to %d tokens for %d blocks",
                max_new_tokens, aligned_tokens, args.num_blocks,
            )
        return multi_try_mcmc_power_sampler_v2(
            mh_llm,
            prompts,
            sampling_params=sampling_params,
            num_of_blocks=args.num_blocks,
            max_new_tokens=aligned_tokens,
            mcmc_steps=args.mcmc_steps,
            num_tries=args.num_tries,
            proposal_temperatures=args.proposal_temperatures,
        )

    results, resources = driver._run_mcmc_benchmark(
        benchmark, model_str, args, max_model_len, init_temperature,
        method="multi_try_mh_v2", cut_power=None, sample_batch=sample_batch,
        wrapper_kwargs={"log_z_temperatures": sorted(set(map(float, args.proposal_temperatures)))},
    )
    fields = _multi_try_v2_fields(args)
    for row in results:
        row.update(fields)
    return results, resources


def _default_run_name(args, benchmark, model_str, max_model_len):
    mixture_tag = "_proposal-" + "-".join(map(str, args.proposal_temperatures))
    return (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_mcmc{args.mcmc_steps}_blocks{args.num_blocks}_tries{args.num_tries}"
        f"_maxlen{max_model_len}_temp{args.temperature}_alpha{args.alpha}"
        f"{mixture_tag}_sched-{args.temperature_schedule_type}_seed{args.seed}"
    )


def main(argv=None):
    baseline.run_benchmark_main(
        argv, parser=build_parser(), required_keys=REQUIRED_MULTI_TRY_V2_KEYS,
        validate=validate_args, run_sampling=run_multi_try_v2_sampling,
        method="multi_try_mh_v2", extra_fields=_multi_try_v2_fields,
        default_run_name=_default_run_name,
    )


if __name__ == "__main__":
    main()

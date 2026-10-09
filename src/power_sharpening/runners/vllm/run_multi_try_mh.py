"""Run the existing vLLM MultiTryMH sampler on a configured benchmark.

Run on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      python -m power_sharpening.runners.vllm.run_multi_try_mh \
      --dataset math500 --algorithm multi_try --model_str qwen \
      --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry \
      --override num_tries=4

Use ``--override proposal_temperatures=[0.25,0.5,1.0]`` for a uniform mixture of whole-suffix proposal temperatures.
Explicit mixtures require a constant schedule; scalar proposal temperatures support the sampler's schedules. The cyclic
schedule is unavailable because this runner does not expose its required minimum and maximum temperatures.
"""

import argparse
import json
import math
import logging
from pathlib import Path

# Import the shared driver first: it sets the vLLM environment before loading the backend and owns engine/probe cleanup,
# batching, context caps, and grading.
from power_sharpening.runners.vllm import run_power_mh as driver
from power_sharpening.backends.vllm.samplers.multi_try_mh import (
    multi_try_mcmc_power_sampler,
)


REQUIRED_MULTI_TRY_KEYS = driver.REQUIRED_MCMC_KEYS + (
    "num_tries",
    "scoring_batch_size",
)


def build_parser():
    """Build the shared config CLI with the MultiTry algorithm selected."""
    parser = argparse.ArgumentParser(
        description="Config-driven vLLM MultiTryMH runner with uniform cuts.",
    )
    driver.add_config_selection_args(
        parser,
        algorithm_default="multi_try",
        algorithm_choices=["multi_try"],
        save_str_default="result/multi_try/",
    )
    parser.add_argument(
        "--resource_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save GPU and KV-cache measurements to <run>.resources.json. "
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for exact occupancy and "
            "eviction metrics. --override resource_probe=false wins over "
            "this flag."
        ),
    )
    return parser


def validate_args(args):
    """Validate MultiTry inputs before constructing the dataset or engine."""
    args.proposal_temperatures = getattr(args, "proposal_temperatures", None)
    args.max_samples = getattr(args, "max_samples", None)
    if args.cut_dist_type != "uniform":
        raise ValueError("MultiTryMH requires cut_dist_type=uniform")
    if args.temperature_schedule_type == "cyclic":
        raise ValueError(
            "temperature_schedule_type=cyclic requires min_temp and max_temp, "
            "which this runner does not expose"
        )
    for key, minimum in (
        ("num_tries", 1), ("scoring_batch_size", 1), ("num_blocks", 1),
        ("max_new_tokens", 1), ("mcmc_steps", 0), ("batch_size", 1),
    ):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}")
    if args.max_new_tokens % args.num_blocks:
        raise ValueError("max_new_tokens must be divisible by num_blocks")
    if args.max_samples is not None and (
        isinstance(args.max_samples, bool)
        or not isinstance(args.max_samples, int)
        or args.max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer or null")
    requested_context = getattr(args, "max_model_len", None)
    if requested_context is not None and (
        isinstance(requested_context, bool)
        or not isinstance(requested_context, int)
        or requested_context <= 0
    ):
        raise ValueError("max_model_len must be a positive integer or null")
    for key in ("alpha", "temperature"):
        value = getattr(args, key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 and not (key == "temperature" and value == -1))
        ):
            allowed = "-1 or a positive finite number" if key == "temperature" else "finite and positive"
            raise ValueError(f"{key} must be {allowed}")
    temperatures = args.proposal_temperatures
    if temperatures is not None:
        if (
            not isinstance(temperatures, list)
            or not temperatures
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in temperatures
            )
        ):
            raise ValueError(
                "proposal_temperatures must be a nonempty list of positive finite numbers"
            )
        if args.temperature_schedule_type != "const":
            raise ValueError(
                "proposal_temperatures requires temperature_schedule_type=const; "
                "override proposal_temperatures=null to anneal the scalar temperature."
            )
    return 1.0 / args.alpha if args.temperature == -1 else args.temperature


def _multi_try_fields(args):
    return {
        "num_tries": args.num_tries,
        "proposal_temperatures": args.proposal_temperatures,
        "scoring_batch_size": args.scoring_batch_size,
    }


def run_multi_try_sampling(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
    init_temperature: float,
):
    """Run MultiTryMH through the shared vLLM benchmark and resource lifecycle."""
    def sample_batch(mh_llm, prompts, sampling_params, scheduler, max_new_tokens):
        # A context cap can leave a partial block. All candidates in a block need the same fixed horizon, so discard the
        # remainder before sampling.
        aligned_tokens = max_new_tokens // args.num_blocks * args.num_blocks
        if aligned_tokens <= 0:
            raise ValueError("remaining context is too short for one token per block")
        if aligned_tokens != max_new_tokens:
            driver.logger.info(
                "aligned generation budget from %d to %d tokens for %d blocks",
                max_new_tokens, aligned_tokens, args.num_blocks,
            )
        return multi_try_mcmc_power_sampler(
            mh_llm,
            prompts,
            sampling_params=sampling_params,
            num_of_blocks=args.num_blocks,
            max_new_tokens=aligned_tokens,
            mcmc_steps=args.mcmc_steps,
            num_tries=args.num_tries,
            # With no refinement steps, keep the initial temperature; cosine and linear schedules cannot advance over a
            # zero-step horizon.
            temperature_scheduler=scheduler if args.mcmc_steps else None,
            proposal_temperatures=args.proposal_temperatures,
            scoring_batch_size=args.scoring_batch_size,
        )

    results, resources = driver._run_mcmc_benchmark(
        benchmark, model_str, args, max_model_len, init_temperature,
        method="multi_try_mh", cut_power=None, sample_batch=sample_batch,
    )
    fields = _multi_try_fields(args)
    for row in results:
        row.update(fields)
    return results, resources


def _default_run_name(args, benchmark, model_str, max_model_len):
    mixture_tag = ""
    if args.proposal_temperatures is not None:
        mixture_tag = "_proposal-" + "-".join(map(str, args.proposal_temperatures))
    return (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_mcmc{args.mcmc_steps}_blocks{args.num_blocks}_tries{args.num_tries}"
        f"_maxlen{max_model_len}_temp{args.temperature}_alpha{args.alpha}"
        f"{mixture_tag}_scorebatch{args.scoring_batch_size}"
        f"_sched-{args.temperature_schedule_type}_seed{args.seed}"
    )


def run_benchmark_main(
    argv=None, *, parser, required_keys, validate, run_sampling,
    method, extra_fields, default_run_name,
):
    """Resolve a MultiTry-family run and save its CSV/resource artifacts.

    The standalone and prefetching runners share the same model setup, validation lifecycle, output naming, and resource
    metadata conventions. Returns the resolved arguments, rows, and run name for optional artifacts.
    """
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    task = driver.resolve_config(args, args.algorithm, required_keys)
    try:
        init_temperature = validate(args)
    except ValueError as error:
        parser.error(str(error))

    driver.set_random_seed(args.seed)
    model_str = driver.MODEL_MAP[args.model_str]
    benchmark = driver.build_benchmark(task, args, model_str)
    driver.logger.info("loading %s of length %d", benchmark.name, benchmark.size)
    use_chat_template = benchmark.should_use_chat_template(args.model_str)
    driver.logger.info(
        "prompt style for %s: %s", args.model_str,
        "chat template" if use_chat_template else "raw",
    )
    max_model_len = None
    if benchmark.uses_max_model_len or getattr(args, "max_model_len", None) is not None:
        max_model_len, checkpoint_limit = driver.resolve_max_model_len(
            model_str, requested=getattr(args, "max_model_len", None),
        )
        driver.logger.info(
            "checkpoint context limit=%s; using max_model_len=%s",
            checkpoint_limit, max_model_len,
        )
    driver._log_run_configuration(
        args, method=method, benchmark_name=benchmark.name,
        benchmark_size=benchmark.size, model_str=model_str, task=task,
        max_model_len=max_model_len, init_temperature=init_temperature,
    )
    results, resources = run_sampling(
        benchmark, model_str, args, max_model_len, init_temperature,
    )

    # Keep pandas after inference to preserve vLLM's dynamic-loader ordering.
    import pandas as pd

    output_dir = Path(args.save_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or default_run_name(args, benchmark, model_str, max_model_len)
    output_path = output_dir / f"{run_name}.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    driver.logger.info("output saved to %s", output_path)
    if resources.get("enabled"):
        resource_path = output_dir / f"{run_name}.resources.json"
        metadata = {
            key: getattr(args, key)
            for key in (
                "algorithm", "dataset", "alpha", "num_blocks", "mcmc_steps",
                "max_new_tokens", "max_samples", "seed", "prefix_cache",
                "temperature", "temperature_schedule_type", "cut_dist_type",
            )
        }
        metadata.update(
            run=run_name, method=method, model=args.model_str,
            base_llm=model_str, dtype=getattr(args, "dtype", "auto"),
            max_model_len=max_model_len, init_temperature=init_temperature,
            **extra_fields(args), **resources,
        )
        resource_path.write_text(json.dumps(metadata, indent=2) + "\n")
        driver.logger.info("resource metrics saved to %s", resource_path)
    return args, results, run_name


def main(argv=None):
    run_benchmark_main(
        argv, parser=build_parser(), required_keys=REQUIRED_MULTI_TRY_KEYS,
        validate=validate_args, run_sampling=run_multi_try_sampling,
        method="multi_try_mh", extra_fields=_multi_try_fields,
        default_run_name=_default_run_name,
    )


if __name__ == "__main__":
    main()

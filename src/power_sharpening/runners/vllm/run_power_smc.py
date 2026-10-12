"""Unified Power-SMC runner using the shared benchmark interfaces.

Show configuration options with ``python -m power_sharpening.runners.vllm.run_power_smc --help``.

The benchmark classes own dataset loading, prompt formatting, completion extraction, grading, and prediction formatting.
This file only configures Power-SMC and iterates over the selected benchmark.

Power-SMC draws samples from the *sharpened* distribution p(x)^alpha using Sequential Monte Carlo: a population of
`n_particles` candidate continuations is grown block by block, periodically resampled toward higher-probability
particles. The same driver works for every benchmark because each benchmark implements the same
``get_question_and_answer`` / ``evaluate_completions`` interface.

Each batch logs ``smc_power_sampler took <s> seconds for <i>-th prompts``, the same generation-time line the other
runners write. ``--resource_probe`` (on by default) measures KV-cache occupancy, KV-block eviction, prefix-cache hits,
and GPU memory as the other runners do: per batch in the CSV rows and for the whole run in ``<run>.resources.json``. Set
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` for exact occupancy and eviction counts.

``run_standard_smc.py`` runs the same loop on stock vLLM (``engine="standard"``): base-model log p comes from a
``prompt_logprobs`` rescoring pass (``standard_smc_sampler.py``) instead of the custom engine's ``power_logprobs``.
"""

import os


import argparse
import json
import time
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

# Model alias -> checkpoint mapping.
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.config.model_context import (
    batch_max_new_tokens,
    resolve_max_model_len,
)

# Shared helpers: benchmark factory and seeding.
from power_sharpening.tasks.registry import (
    build_benchmark,
    set_random_seed,
)

from power_sharpening.common.temp_scheduler import set_schedule
from power_sharpening.config import add_config_selection_args, resolve_config

from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper
from power_sharpening.backends.vllm.samplers import smc_power_sampler
from power_sharpening.backends.vllm.samplers.standard_smc_sampler import standard_vllm_smc_power_sampler
from power_sharpening.backends.vllm.engine_patch import SamplingParams
# ResourceProbe measures GPU memory and KV-cache usage from the live engine, as in run_power_mh.py and run_low_temp.py.
from power_sharpening.common.resource_probe import ResourceProbe

import logging

logger = logging.getLogger("[run_power_smc_vllm]")


def _log_run_configuration(args, **derived_config):
    """Log the resolved YAML/CLI configuration before the run starts."""
    config = dict(vars(args))
    config.update(derived_config)
    lines = [f"  {key}: {value!r}" for key, value in sorted(config.items())]
    logger.info("resolved configuration:\n%s", "\n".join(lines))


def _get_prompts_and_problems(benchmark, batch, tokenizer):
    """Call get_question_and_answer while preserving raw problems for grading."""
    prompts, _ = benchmark.get_question_and_answer(batch, tokenizer)
    problems = benchmark.batch_to_problems(batch)
    return prompts, problems[: len(prompts)]


def _run_size(benchmark, args) -> int:
    """Number of prompts to run: the whole benchmark, or the first ``max_samples`` of it."""
    max_samples = getattr(args, "max_samples", None)
    if max_samples is None:
        return benchmark.size
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
        raise SystemExit(f"max_samples must be a positive integer or null, got {max_samples!r}.")
    return min(benchmark.size, max_samples)


def run_power_smc(
    benchmark,
    model_str: str,
    args,
    max_model_len: int,
    init_temperature: float,
    engine: str = "custom",
):
    """Run Power-SMC over any benchmark implementation on the custom or the stock (``engine="standard"``) vLLM."""
    logger.info("Benchmark on %s (Power-SMC)...", benchmark.name)
    results = []
    begin = time.time()
    seen = 0  # number of prompts processed so far
    total_accuracy = 0  # running count of correct predictions
    run_size = _run_size(benchmark, args)

    # Only pass max_model_len to vLLM when we resolved one; otherwise let vLLM use the checkpoint default.
    vllm_kwargs = {}
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    # Prefix-cache counters only exist when the engine keeps statistics, and the offline LLM entrypoint switches them
    # off by default.
    if getattr(args, "resource_probe", True):
        vllm_kwargs["disable_log_stats"] = False
    if engine == "standard":
        # Generation logprobs must be the proposal q (after temperature and top-k/top-p); stock vLLM defaults to raw
        # logprobs, which are the base model p. The rescoring pass's prompt logprobs are raw p in either mode.
        vllm_kwargs["logprobs_mode"] = "processed_logprobs"

    common_fields = {
        "method": f"power_smc_{engine}_vllm",
        "base_llm": model_str,
        "alpha": args.alpha,
        "n_particles": args.n_particles,
        "block_size": args.block_size,
        "ess_threshold": args.ess_threshold,
        "alpha_ramp_tokens": args.alpha_ramp_tokens,
        "init_temperature": init_temperature,
        "max_model_len": max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "prefix_cache": args.prefix_cache,
        "temperature_schedule_type": args.temperature_schedule_type,
    }
    if engine == "standard":
        common_fields["smc_score_batch_size"] = args.smc_score_batch_size

    logger.info("Power-SMC engine=%s prefix caching=%s", engine, args.prefix_cache)

    # Start before model construction so the probe can tell the loaded engine footprint from generation memory.
    probe = ResourceProbe(enabled=bool(getattr(args, "resource_probe", True)))
    probe.start()

    # Load the vLLM engine once and reuse it for every batch; closing() also releases the probe hooks if the run
    # fails.
    with closing(probe), vLLM_Wrapper(
        model=model_str,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.prefix_cache,
        engine_type=engine,
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **vllm_kwargs,
    ) as mh_llm:
        probe.attach(mh_llm.llm)
        tokenizer = mh_llm.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # alpha is the power-sharpening exponent (target p(x)^alpha); the initial temperature seeds the temperature
        # schedule below.
        sampling_params = SamplingParams(
            temperature=init_temperature,
            alpha=args.alpha,
            seed=args.seed,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        # SMC advances one "chunk" per block_size tokens, and the temperature schedule is stepped once per chunk, so
        # total_steps == #chunks.
        total_chunks = max(args.max_new_tokens // args.block_size, 1)
        scheduler = set_schedule(
            init_temperature,
            args.temperature_schedule_type,
            total_steps=total_chunks,
        )
        logger.info(
            "temperature schedule=%s total_chunks=%d",
            args.temperature_schedule_type,
            total_chunks,
        )

        # Run configuration stamped onto every result row, so a single CSV is self-describing for downstream analysis.

        # Iterate over the benchmark's batches (DataLoader yields raw rows).
        for batch in benchmark.dataset_loader:
            if seen >= run_size:
                break
            benchmark.seen = seen
            # The benchmark turns raw rows into model-ready prompts plus the per-problem objects it later needs for
            # grading.
            prompts, problems = _get_prompts_and_problems(
                benchmark,
                batch,
                tokenizer=tokenizer,
            )
            prompts = prompts[:run_size - seen]
            problems = problems[:run_size - seen]
            if not prompts:
                continue

            started = time.time()
            logger.info(
                "BATCH %d-%d\tFIRST_INPUT: %s",
                seen,
                seen + len(prompts) - 1,
                prompts[0],
            )

            # Cap the generation budget so prompt + output fits the context window; without a known limit, use the
            # configured budget.
            if max_model_len is None:
                max_new_tokens = args.max_new_tokens
            else:
                max_new_tokens = batch_max_new_tokens(
                    tokenizer,
                    prompts,
                    max_model_len,
                    args.max_new_tokens,
                )

            # Core Power-SMC call: maintains n_particles per prompt, resamples every block_size tokens when ESS drops
            # below ess_threshold, and ramps alpha from 1 -> args.alpha over the first alpha_ramp_tokens.
            sampler_started = time.time()
            smc_kwargs = dict(
                sampling_params=sampling_params,
                max_new_tokens=max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                block_size=args.block_size,
                n_particles=args.n_particles,
                ess_threshold=args.ess_threshold,
                alpha_ramp_tokens=args.alpha_ramp_tokens,
                temperature_scheduler=scheduler,
            )
            if engine == "standard":
                completions = standard_vllm_smc_power_sampler(
                    mh_llm, prompts, score_batch_size=args.smc_score_batch_size, **smc_kwargs
                )
            else:
                completions = smc_power_sampler(mh_llm, prompts, **smc_kwargs)
            # Same wording as the MH and low-temperature runners' timing line, so one parser reads all of them; it covers
            # generation only, not grading.
            logger.info(
                "smc_power_sampler took %.3f seconds for %d-th prompts",
                time.time() - sampler_started,
                seen,
            )
            # Close the resource window on this batch before grading, which touches no GPU.
            resources = probe.mark()
            # The benchmark extracts predictions and grades each completion.
            batch_results = benchmark.evaluate_completions(problems, completions)

            for index, result in enumerate(batch_results):
                logger.info(
                    "sample %d response_length=%d prediction=%s "
                    "is_correct=%s justification=%s",
                    seen + index,
                    len(result["completion"]),
                    result["prediction"],
                    result["is_correct"],
                    result["justification"],
                )
                # Merge run config with the per-sample grading result and the batch's resource window.
                results.append({**common_fields, **result, **resources})
                total_accuracy += int(result["is_correct"])

            # Progress + ETA accounting.
            seen += len(prompts)
            total_elapsed = time.time() - begin
            avg_per_sample = total_elapsed / seen
            remaining = avg_per_sample * max(run_size - seen, 0)
            logger.info(
                "[%d/%d] correct=%d/%d acc=%.4f | "
                "batch=%.1fs avg/sample=%.1fs total=%.4fhours eta=%.1fsec",
                seen,
                run_size,
                total_accuracy,
                seen,
                total_accuracy / seen,
                time.time() - started,
                avg_per_sample,
                total_elapsed / 3600,
                remaining,
            )

    # Run-level peaks remain available after engine teardown and probe cleanup.
    logger.info("resources: %s", probe.summary_line())
    return results, probe.summary()


# Config keys the SMC run requires after the YAML files are merged.
REQUIRED_SMC_KEYS = (
    "task",
    "batch_size",
    "seed",
    "alpha",
    "n_particles",
    "block_size",
    "ess_threshold",
    "alpha_ramp_tokens",
    "min_new_tokens",
    "temperature",
    "temperature_schedule_type",
    "gpu_memory_utilization",
    "top_p",
    "top_k",
    "repetition_penalty",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven Power-SMC runner. Selects a dataset + algorithm "
            "from dataset.yaml / algorithms.yaml and runs Power-SMC; use "
            "--override to change any merged value from the CLI."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="smc",
        algorithm_choices=["smc"],
        save_str_default="power_smc_results/",
    )
    probe_group = parser.add_argument_group("resource measurement")
    probe_group.add_argument(
        "--resource_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure peak KV-cache occupancy and prefix-cache hit rate, plus "
            "GPU-memory and KV-eviction metrics, as run_power_mh.py does. "
            "Results are written to <run>.resources.json and per-batch CSV "
            "columns. Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for exact occupancy "
            "and eviction metrics. Pass --no-resource_probe to disable "
            "measurement. (--override resource_probe=false wins over this flag.)"
        ),
    )
    return parser


def main(engine: str = "custom"):
    """Run the Power-SMC command using the shared CLI and result format."""
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Merge dataset.yaml + algorithms.yaml for the chosen (dataset, smc), apply --override, project onto args, and
    # validate the SMC hyperparameters.
    task = resolve_config(
        args,
        args.algorithm,
        REQUIRED_SMC_KEYS + (("smc_score_batch_size",) if engine == "standard" else ()),
    )

    # Seed Python/NumPy/Torch/CUDA (and vLLM workers) for reproducibility.
    set_random_seed(args.seed)

    model_str = MODEL_MAP[args.model_str]
    # Instantiate the selected benchmark with its dataset-specific options.
    benchmark = build_benchmark(task, args, model_str)
    logger.info("loading %s of length %d", benchmark.name, benchmark.size)

    # Decide the context window: some benchmarks (e.g. long-context code) need an explicit cap derived from the
    # checkpoint; others leave it to vLLM.
    if not benchmark.uses_max_model_len:
        max_model_len = None
    else:
        max_model_len, checkpoint_limit = resolve_max_model_len(
            model_str, requested=None
        )
        logger.info(
            "checkpoint context limit=%s; using max_model_len=%s",
            checkpoint_limit,
            max_model_len,
        )

    # Convention: --temperature -1 means "use 1/alpha" (the natural starting
    # temperature for sampling from p(x)^alpha).
    init_temperature = (
        1.0 / args.alpha if args.temperature == -1.0 else args.temperature
    )
    _log_run_configuration(
        args,
        benchmark_name=benchmark.name,
        benchmark_size=benchmark.size,
        init_temperature=init_temperature,
        max_model_len=max_model_len,
        model_str=model_str,
        task=task,
    )
    results, resource_summary = run_power_smc(
        benchmark, model_str, args, max_model_len, init_temperature, engine=engine
    )

    # Persist results to a CSV whose name encodes the full run configuration.
    os.makedirs(args.save_str, exist_ok=True)
    output_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_{engine}-vllm"
        f"_nparticles{args.n_particles}_block{args.block_size}"
        f"_ess{args.ess_threshold}_ramp{args.alpha_ramp_tokens}"
        f"_maxlen{max_model_len}_temp{args.temperature}_alpha{args.alpha}"
        f"_sched-{args.temperature_schedule_type}_seed{args.seed}.csv"
    )
    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name below, which no log matches.
    if getattr(args, "run_name", None):
        output_name = f"{args.run_name}.csv"
    output_filename = os.path.join(
        args.save_str,
        output_name,
    )
    pd.DataFrame(results).to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

    if resource_summary.get("enabled"):
        resource_filename = output_filename.replace(".csv", ".resources.json")
        with open(resource_filename, "w") as handle:
            json.dump(
                {
                    "run": Path(output_filename).stem,
                    "method": "power_smc" if engine == "custom" else f"power_smc_{engine}",
                    "dataset": args.dataset,
                    "model": args.model_str,
                    "base_llm": model_str,
                    "alpha": args.alpha,
                    "n_particles": args.n_particles,
                    "block_size": args.block_size,
                    "init_temperature": init_temperature,
                    "batch_size": args.batch_size,
                    "max_new_tokens": args.max_new_tokens,
                    "max_samples": getattr(args, "max_samples", None),
                    "seed": args.seed,
                    "prefix_cache": args.prefix_cache,
                    "dtype": getattr(args, "dtype", "auto"),
                    **resource_summary,
                },
                handle,
                indent=2,
            )
        logger.info("resource metrics saved to %s", resource_filename)


if __name__ == "__main__":
    main()

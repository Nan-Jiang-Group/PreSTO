"""Unified power-MCMC runner using the shared benchmark interfaces.

The benchmark classes own dataset loading, prompt formatting, completion extraction, grading, and prediction formatting.
This file only configures blockwise Metropolis-Hastings sampling and iterates over the benchmark.

Power-MCMC draws samples from the *sharpened* distribution p(x)^alpha by running a blockwise Metropolis-Hastings chain:
a completion is split into `num_blocks` blocks and refined over `mcmc_steps` propose/accept rounds. The same driver
works for every benchmark because each benchmark implements the same ``get_question_and_answer`` /
``evaluate_completions`` interface.

``cut_dist_type`` selects the cut law and therefore the sampler:
``uniform`` runs ``mcmc_power_sampler``, while ``entropy`` runs ``mcmc_power_sampler_entropycut``, which draws cuts from
positive jumps in the model's predictive entropy and carries the state-dependent cut-law ratio in every MH correction
(``cut_power`` is its exponent beta).

Run with ``python -m power_sharpening.runners.vllm.run_power_mh --dataset math500 --resource_probe`` to save run-level
KV-cache measurements.
"""

import os

# Disable tqdm progress bars (vLLM/HF emit them per request, which is noisy in batch logs). Must be set before
# vLLM/transformers are imported.
os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import logging
import math
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np

# Model alias -> checkpoint mapping.
from power_sharpening.tasks.constants import MODEL_MAP
# vLLM_Wrapper: vLLM wrapper (patched engine); mcmc_power_sampler: blockwise MH
# power sampler; mcmc_power_sampler_entropycut: same chain with entropy cuts.
from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper
from power_sharpening.backends.vllm.samplers import (
    mcmc_power_sampler,
    mcmc_power_sampler_entropycut,
)
# ResourceProbe measures run-level GPU and KV-cache usage from the live engine.
from power_sharpening.common.resource_probe import ResourceProbe
# set_schedule builds the temperature schedule stepped once per MH step.
from power_sharpening.common.temp_scheduler import set_schedule
from power_sharpening.config.model_context import (
    batch_max_new_tokens,
    resolve_max_model_len,
)
from power_sharpening.backends.vllm.engine_patch import SamplingParams
# Shared helpers: benchmark factory and seeding.
from power_sharpening.tasks.registry import (
    build_benchmark,
    set_random_seed,
)
# Shared config-driven CLI: --dataset/--algorithm/--override + YAML merge.
from power_sharpening.config import add_config_selection_args, resolve_config



logger = logging.getLogger("[run_powersampling_mh_vllm]")


def _log_run_configuration(args, **derived_config):
    """Log the resolved YAML/CLI configuration before the run starts."""
    config = dict(vars(args))
    config.update(derived_config)
    lines = [f"  {key}: {value!r}" for key, value in sorted(config.items())]
    logger.info("resolved configuration:\n%s", "\n".join(lines))


def _get_prompts_and_problems(
    benchmark,
    batch,
    tokenizer,
):
    """Call get_question_and_answer while preserving raw problems for grading."""
    prompts, _ = benchmark.get_question_and_answer(batch, tokenizer)
    problems = benchmark.batch_to_problems(batch)
    return prompts, problems[:len(prompts)]


def _run_mcmc_benchmark(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
    init_temperature: float,
    *,
    method: str,
    cut_power: float | None,
    sample_batch,
    wrapper_kwargs: dict | None = None,
):
    """Drive one MH chain over a benchmark; ``sample_batch`` picks the sampler.

    Everything outside the sampler call is identical for the uniform and entropy-cut chains -- engine construction, the
    resource probe, batching, grading, and the ETA accounting -- so the two public entry points below differ only in
    what they pass here.

    Args:
        method: Value stamped into every result row's ``method`` field.
        cut_power: Exponent beta for entropy cuts, or None for uniform cuts.
        sample_batch: Called as ``sample_batch(mh_llm, prompts, sampling_params, scheduler, max_new_tokens)`` and
            returns one completion per prompt.
        wrapper_kwargs: Extra ``vLLM_Wrapper`` arguments, such as ``log_z_temperatures`` for MultiTryMH v2.
    """
    logger.info(
        "Benchmark on %s (power-MCMC, %s cuts)...",
        benchmark.name,
        args.cut_dist_type,
    )
    results = []
    begin = time.time()
    seen = 0              # number of prompts processed so far
    total_accuracy = 0    # running count of correct predictions
    run_size = (
        benchmark.size
        if args.max_samples is None
        else min(benchmark.size, args.max_samples)
    )

    # Only pass max_model_len to vLLM when we resolved one; otherwise let vLLM use the checkpoint default.
    vllm_kwargs = {}
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
    if gpu_memory_utilization is not None:
        vllm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    resource_probe_enabled = bool(getattr(args, "resource_probe", True))
    # Prefix-cache counters only exist when the engine keeps statistics, and the offline LLM entrypoint switches them
    # off by default.
    if resource_probe_enabled:
        vllm_kwargs.setdefault("disable_log_stats", False)

    # Start before model construction so the probe can distinguish the loaded engine footprint from memory consumed
    # while generating.
    probe = ResourceProbe(enabled=resource_probe_enabled)
    probe.start()

    # Release the engine and probe hooks even if construction or sampling fails.
    with closing(probe), vLLM_Wrapper(
        model=model_str,
        engine_type="custom",
        seed=args.seed,
        enable_prefix_caching=args.prefix_cache,
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **(wrapper_kwargs or {}),
        **vllm_kwargs,
    ) as mh_llm:
        probe.attach(mh_llm.llm)
        # alpha is the power-sharpening exponent (target p(x)^alpha); the initial temperature seeds the temperature
        # schedule below.
        # No per-request seed: vLLM's Gumbel noise is a function of (request seed, position) only, so a shared seed makes
        # every proposal replay the current suffix's randomness and the chain never moves. The engine seed above still
        # makes the run reproducible.
        sampling_params = SamplingParams(
            temperature=init_temperature,
            alpha=args.alpha,
        )
        # The temperature schedule is stepped once per MH step.
        scheduler = set_schedule(
            init_temperature,
            args.temperature_schedule_type,
            total_steps=args.mcmc_steps,
        )
        logger.info(
            "temperature schedule=%s mcmc_steps=%d",
            args.temperature_schedule_type,
            args.mcmc_steps,
        )

        # Run configuration stamped onto every result row, so a single CSV is self-describing for downstream analysis.
        common_fields = {
            "method": method,
            "base_llm": model_str,
            "alpha": args.alpha,
            "mcmc_steps": args.mcmc_steps,
            "num_blocks": args.num_blocks,
            "init_temperature": init_temperature,
            "max_model_len": max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature_schedule_type": args.temperature_schedule_type,
            "cut_dist_type": args.cut_dist_type,
            "cut_power": cut_power,
        }

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
                mh_llm.tokenizer,
            )
            remaining_samples = run_size - seen
            prompts = prompts[:remaining_samples]
            problems = problems[:remaining_samples]
            # Skip empty batches (e.g. a benchmark may filter a whole batch).
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
                    mh_llm.tokenizer,
                    prompts,
                    max_model_len,
                    args.max_new_tokens,
                )

            # Core power-MCMC call: splits each completion into num_blocks blocks and runs mcmc_steps blockwise
            # propose/accept rounds under the temperature schedule.
            sampler_started = time.time()
            completions = sample_batch(
                mh_llm,
                prompts,
                sampling_params,
                scheduler,
                max_new_tokens,
            )
            # Wording held identical for both chains: the case-study parser closes a sample on this line.
            logger.info(
                "mcmc_power_sampler took %.3f seconds for %d-th prompts, "
                "response lengths: %s",
                time.time() - sampler_started,
                seen,
                [len(completion) for completion in completions],
            )
            # Close the resource window on this batch before grading, which touches no GPU.
            resources = probe.mark()
            # The benchmark extracts predictions and grades each completion.
            batch_results = benchmark.evaluate_completions(
                problems, completions
            )

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
            remaining = avg_per_sample * max(
                run_size - seen, 0
            )
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
    resource_summary = probe.summary()
    logger.info("resources: %s", probe.summary_line())

    return results, resource_summary


def run_power_sampling_mcmc(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
    init_temperature: float,
):
    """Run blockwise power-MCMC with the state-independent uniform cut law."""
    # numpy's default_rng ignores the global seed set by set_random_seed.
    rng = np.random.default_rng(args.seed)

    def sample_batch(mh_llm, prompts, sampling_params, scheduler, max_new_tokens):
        return mcmc_power_sampler(
            mh_llm,
            prompts,
            sampling_params=sampling_params,
            num_of_blocks=args.num_blocks,
            max_new_tokens=max_new_tokens,
            mcmc_steps=args.mcmc_steps,
            temperature_scheduler=scheduler,
            rng=rng,
        )

    return _run_mcmc_benchmark(
        benchmark,
        model_str,
        args,
        max_model_len,
        init_temperature,
        method="power_sampling_mcmc",
        cut_power=None,
        sample_batch=sample_batch,
    )


def run_entropycut_power_sampling_mcmc(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
    init_temperature: float,
):
    """Run blockwise power-MCMC with state-dependent entropy cuts.

    Cuts are drawn from positive jumps in the model's predictive entropy raised to ``cut_power``, and every acceptance
    ratio carries the reverse/forward cut-law term the uniform chain does not need.
    """
    beta = float(getattr(args, "cut_power", 4.0))
    # numpy's default_rng ignores the global seed set by set_random_seed.
    rng = np.random.default_rng(args.seed)
    if args.temperature_schedule_type != "const":
        # EntropyCut reuses the proposal log-probs it cached for the current state, so the kernel has to stay put; the
        # sampler takes the schedule's first temperature and holds it there.
        logger.warning(
            "entropy cuts need a fixed proposal kernel: schedule %r is "
            "pinned at its first temperature (%.6f)",
            args.temperature_schedule_type,
            init_temperature,
        )

    def sample_batch(mh_llm, prompts, sampling_params, scheduler, max_new_tokens):
        return mcmc_power_sampler_entropycut(
            mh_llm,
            prompts,
            sampling_params=sampling_params,
            num_of_blocks=args.num_blocks,
            max_new_tokens=max_new_tokens,
            mcmc_steps=args.mcmc_steps,
            beta=beta,
            temperature_scheduler=scheduler,
            rng=rng,
        )

    return _run_mcmc_benchmark(
        benchmark,
        model_str,
        args,
        max_model_len,
        init_temperature,
        method="entropycut_power_sampling_mcmc",
        cut_power=beta,
        sample_batch=sample_batch,
    )


# Config keys the power-MCMC run requires after the YAML files are merged.
REQUIRED_MCMC_KEYS = (
    "task",
    "batch_size",
    "seed",
    "alpha",
    "mcmc_steps",
    "num_blocks",
    "max_new_tokens",
    "temperature",
    "temperature_schedule_type",
    "cut_dist_type",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven power-MCMC runner. Selects a dataset + algorithm "
            "from dataset.yaml / algorithms.yaml and runs blockwise "
            "Metropolis-Hastings; --override changes any merged value. "
            "--override cut_dist_type=entropy runs the EntropyCut chain "
            "instead, with cut_power as its exponent beta."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="power_mcmc",
        algorithm_choices=["power_mcmc"],
        save_str_default="wilcoxon_test_mcmc_vs_standard/",
    )
    probe_group = parser.add_argument_group("resource measurement")
    probe_group.add_argument(
        "--resource_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure peak KV-cache occupancy and prefix-cache hit rate, plus "
            "GPU-memory and KV-eviction metrics. Results are written to "
            "<run>.resources.json. This also enables vLLM engine statistics. "
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for exact occupancy and "
            "eviction metrics; otherwise occupancy is polled and eviction is "
            "unavailable. Pass --no-resource_probe to disable measurement. "
            "(--override resource_probe=false wins over this flag.)"
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Merge dataset.yaml + algorithms.yaml for (dataset, power_mcmc), apply
    # --override, project onto args, and validate the MCMC hyperparameters.
    task = resolve_config(args, args.algorithm, REQUIRED_MCMC_KEYS)
    if args.cut_dist_type not in ("uniform", "entropy"):
        raise SystemExit(
            f"cut_dist_type={args.cut_dist_type!r} not in ['entropy', 'uniform']."
        )
    if args.cut_dist_type == "entropy":
        cut_power = float(getattr(args, "cut_power", 4.0))
        if not math.isfinite(cut_power) or cut_power < 0.0:
            raise SystemExit(
                "cut_power must be a finite nonnegative value, got "
                f"{cut_power!r}."
            )
    if not hasattr(args, "max_samples"):
        args.max_samples = None
    if args.max_samples is not None and (
        isinstance(args.max_samples, bool)
        or not isinstance(args.max_samples, int)
        or args.max_samples <= 0
    ):
        raise SystemExit(
            "max_samples must be a positive integer or null, got "
            f"{args.max_samples!r}."
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

    # Whether to wrap prompts in the model's chat template (model-dependent).
    use_chat_template = benchmark.should_use_chat_template(args.model_str)
    logger.info(
        "prompt style for %s: %s",
        args.model_str,
        "chat template" if use_chat_template else "raw",
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
        model_str=model_str,
        task=task,
        max_model_len=max_model_len,
        init_temperature=init_temperature,
    )

    if args.cut_dist_type == "entropy":
        results, resource_summary = run_entropycut_power_sampling_mcmc(
            benchmark,
            model_str,
            args,
            max_model_len,
            init_temperature,
        )
    else:
        results, resource_summary = run_power_sampling_mcmc(
            benchmark,
            model_str,
            args,
            max_model_len,
            init_temperature,
        )

    # Persist results to a CSV whose name encodes the full run configuration.
    os.makedirs(args.save_str, exist_ok=True)
    cut_tag = ""
    if args.cut_dist_type != "uniform":
        cut_tag = f"_cut-{args.cut_dist_type}"
    output_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_mcmc{args.mcmc_steps}_blocks{args.num_blocks}"
        f"_maxlen{max_model_len}_temp{args.temperature}_alpha{args.alpha}"
        f"{cut_tag}"
        f"_sched-{args.temperature_schedule_type}_seed{args.seed}.csv"
    )
    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name below, which no log matches.
    if getattr(args, "run_name", None):
        output_name = f"{args.run_name}.csv"
    output_filename = os.path.join(args.save_str, output_name)
    # Import pandas only after vLLM has completed sampling.  On Vista's glibc/Conda stack, loading pandas before vLLM's
    # compiled dependencies can abort the dynamic loader while it initializes static TLS.
    import pandas as pd

    pd.DataFrame(results).to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

    if resource_summary.get("enabled"):
        resource_filename = output_filename.replace(".csv", ".resources.json")
        with open(resource_filename, "w") as handle:
            json.dump(
                {
                    "run": Path(output_filename).stem,
                    "method": (
                        "entropycut_power_mh"
                        if args.cut_dist_type == "entropy"
                        else "power_mh"
                    ),
                    "algorithm": args.algorithm,
                    "dataset": args.dataset,
                    "model": args.model_str,
                    "base_llm": model_str,
                    "alpha": args.alpha,
                    "num_blocks": args.num_blocks,
                    "mcmc_steps": args.mcmc_steps,
                    "cut_dist_type": args.cut_dist_type,
                    "cut_power": (
                        float(getattr(args, "cut_power", 4.0))
                        if args.cut_dist_type == "entropy"
                        else None
                    ),
                    "max_new_tokens": args.max_new_tokens,
                    "max_samples": args.max_samples,
                    "seed": args.seed,
                    "prefix_cache": args.prefix_cache,
                    "dtype": getattr(args, "dtype", "auto"),
                    **resource_summary,
                },
                handle,
                indent=2,
            )
        logger.info("resource metrics saved to %s", resource_filename)

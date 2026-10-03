"""Unified Calderhead parallel power-MCMC runner using the shared benchmark interfaces.

The benchmark classes own dataset loading, prompt formatting, completion extraction, grading, and prediction formatting.
This file only configures Calderhead's parallel Metropolis-Hastings sampling and iterates over the benchmark.

Calderhead-MH draws samples from the *sharpened* distribution p(x)^alpha with a parallel multi-proposal MCMC chain
(Calderhead 2014,
https://www.pnas.org/doi/10.1073/pnas.1408184111): at each of ``mcmc_steps``
blockwise steps it generates ``num_proposals`` candidate blocks, builds the transition matrix, and samples the next
state -- enabling parallel generation while preserving correct MH dynamics. The same driver works for every benchmark
because each benchmark implements the same ``get_question_and_answer`` / ``evaluate_completions`` interface.
"""

import os

# Disable tqdm progress bars (vLLM/HF emit them per request, which is noisy in batch logs). Must be set before
# vLLM/transformers are imported.
os.environ["TQDM_DISABLE"] = "1"

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Model alias -> checkpoint mapping.
from power_sharpening.tasks.constants import MODEL_MAP
# vLLM_Wrapper: vLLM wrapper (patched engine); calderhead_mcmc_power_sampler:
# parallel multi-proposal blockwise MH power sampler.
from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper
from power_sharpening.backends.vllm.samplers import calderhead_mcmc_power_sampler
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



logger = logging.getLogger("[run_calderhead_mh_vllm]")


def _get_prompts_and_problems(
    benchmark,
    batch,
    tokenizer,
):
    """Call get_question_and_answer while preserving raw problems for grading."""
    prompts, _ = benchmark.get_question_and_answer(batch, tokenizer)
    problems = benchmark.batch_to_problems(batch)
    return prompts, problems[:len(prompts)]


def run_calderhead_mcmc(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
    init_temperature: float,
):
    """Run Calderhead parallel power-MCMC over a shared benchmark implementation."""
    logger.info("Benchmark on %s (Calderhead parallel MCMC)...", benchmark.name)
    results = []
    begin = time.time()
    seen = 0              # number of prompts processed so far
    total_accuracy = 0    # running count of correct predictions

    # Only pass max_model_len to vLLM when we resolved one; otherwise let vLLM use the checkpoint default.
    vllm_kwargs = {}
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len

    # Load the model once and reuse it for every batch (the context manager tears vLLM down on exit).
    with vLLM_Wrapper(
        model=model_str,
        engine_type="custom",
        seed=args.seed,
        enable_prefix_caching=args.prefix_cache,
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **vllm_kwargs,
    ) as mh_llm:
        # alpha is the power-sharpening exponent (target p(x)^alpha); the initial temperature seeds the temperature
        # schedule below.
        sampling_params = SamplingParams(
            temperature=init_temperature,
            alpha=args.alpha,
            seed=args.seed,
        )
        # The temperature schedule is stepped once per MH step.
        scheduler = set_schedule(
            init_temperature,
            args.temperature_schedule_type,
            total_steps=args.mcmc_steps,
        )
        logger.info(
            "temperature schedule=%s mcmc_steps=%d num_proposals=%d",
            args.temperature_schedule_type,
            args.mcmc_steps,
            args.num_proposals,
        )

        # Run configuration stamped onto every result row, so a single CSV is self-describing for downstream analysis.
        common_fields = {
            "method": "calderhead_power_sampling_mcmc",
            "base_llm": model_str,
            "alpha": args.alpha,
            "mcmc_steps": args.mcmc_steps,
            "num_blocks": args.num_blocks,
            "num_proposals": args.num_proposals,
            "init_temperature": init_temperature,
            "max_model_len": max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature_schedule_type": args.temperature_schedule_type,
        }

        # Iterate over the benchmark's batches (DataLoader yields raw rows).
        for batch in benchmark.dataset_loader:
            benchmark.seen = seen
            # The benchmark turns raw rows into model-ready prompts plus the per-problem objects it later needs for
            # grading.
            prompts, problems = _get_prompts_and_problems(
                benchmark,
                batch,
                mh_llm.tokenizer,
            )
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

            # Core Calderhead call: at each of mcmc_steps blockwise steps it proposes num_proposals candidate blocks and
            # samples the next state from the resulting transition matrix.
            completions = calderhead_mcmc_power_sampler(
                mh_llm,
                prompts,
                sampling_params=sampling_params,
                num_of_blocks=args.num_blocks,
                max_new_tokens=max_new_tokens,
                mcmc_steps=args.mcmc_steps,
                num_proposals=args.num_proposals,
                temperature_scheduler=scheduler,
            )
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
                # Merge run config with the per-sample grading result.
                results.append({**common_fields, **result})
                total_accuracy += int(result["is_correct"])

            # Progress + ETA accounting.
            seen += len(prompts)
            total_elapsed = time.time() - begin
            avg_per_sample = total_elapsed / seen
            remaining = avg_per_sample * max(
                benchmark.size - seen, 0
            )
            logger.info(
                "[%d/%d] correct=%d/%d acc=%.4f | "
                "batch=%.1fs avg/sample=%.1fs total=%.4fhours eta=%.1fsec",
                seen,
                benchmark.size,
                total_accuracy,
                seen,
                total_accuracy / seen,
                time.time() - started,
                avg_per_sample,
                total_elapsed / 3600,
                remaining,
            )

    return results


# Config keys the Calderhead run requires after the YAML files are merged.
REQUIRED_CALDERHEAD_KEYS = (
    "task",
    "batch_size",
    "seed",
    "alpha",
    "mcmc_steps",
    "num_blocks",
    "num_proposals",
    "temperature",
    "temperature_schedule_type",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven Calderhead parallel power-MCMC runner. Selects a "
            "dataset + algorithm from dataset.yaml / algorithms.yaml; "
            "--override changes any merged value from the CLI."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="calderhead_power_mcmc",
        algorithm_choices=["calderhead_power_mcmc"],
        save_str_default="wilcoxon_test_mcmc_vs_standard/",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Merge dataset.yaml + algorithms.yaml for (dataset, calderhead_power_mcmc), apply --override, project onto args,
    # and validate the hyperparameters.
    task = resolve_config(args, args.algorithm, REQUIRED_CALDERHEAD_KEYS)

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
    results = run_calderhead_mcmc(
        benchmark,
        model_str,
        args,
        max_model_len,
        init_temperature,
    )

    # Persist results to a CSV whose name encodes the full run configuration.
    os.makedirs(args.save_str, exist_ok=True)
    output_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_mcmc{args.mcmc_steps}_blocks{args.num_blocks}_nprop{args.num_proposals}"
        f"_maxlen{max_model_len}_temp{args.temperature}_alpha{args.alpha}"
        f"_sched-{args.temperature_schedule_type}_seed{args.seed}.csv"
    )
    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name below, which no log matches.
    if getattr(args, "run_name", None):
        output_name = f"{args.run_name}.csv"
    output_filename = os.path.join(args.save_str, output_name)
    pd.DataFrame(results).to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

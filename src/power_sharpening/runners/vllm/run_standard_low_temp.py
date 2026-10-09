"""Unified low-temperature runner using the shared benchmark interfaces.

The benchmark classes own dataset loading, prompt formatting, completion extraction, grading, and prediction formatting.
This file only configures low-temperature sampling and iterates over the selected benchmark.

This is the "standard" low-temperature baseline compared against Power-SMC (``run_power_smc.py``) and Power-MCMC
(``run_power_mh.py``): draw a single sample per prompt at a fixed low temperature and grade it.

The same drivers work for every benchmark because each benchmark implements
the same ``get_question_and_answer`` / ``evaluate_completions`` interface:
``get_question_and_answer(batch, tokenizer)`` returns the formatted prompts plus the problem dicts that
``evaluate_completions`` grades against.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from power_sharpening.config.model_context import (
    batch_max_new_tokens,
    resolve_max_model_len,
)
# Shared config-driven CLI: --dataset/--algorithm/--override + YAML merge.
from power_sharpening.config import (
    add_config_selection_args,
    resolve_config,
)

# Model alias -> checkpoint mapping.
from power_sharpening.tasks.constants import MODEL_MAP

from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper

# Shared helpers: benchmark factory and seeding.
from power_sharpening.tasks.registry import (
    build_benchmark,
    set_random_seed,
)

def _engine_max_model_len(llm_wrapper, fallback: int | None) -> int:
    if fallback is not None:
        return fallback
    try:
        return llm_wrapper.llm.llm_engine.model_config.max_model_len
    except Exception:
        return 4096


def evaluate_with_extraction(
    benchmark,
    llm_wrapper,
    tokenizer,
    prompts: list[str],
    problems: list[dict],
    completions: list[str],
    max_model_len: int | None,
    seed: int,
) -> list[dict]:
    """Grade completions, then recover unreliable parsed answers if needed."""
    results = benchmark.evaluate_completions(problems, completions)
    pending = []
    for index, result in enumerate(results):
        if benchmark.needs_extraction(
            result["prediction"],
            completion=completions[index],
            problem=problems[index],
        ):
            cue = benchmark.extract_answer_text(
                completions[index], problems[index]
            )
            if cue:
                pending.append((index, cue))

    if not pending:
        return results

    ext_max_tokens = benchmark.extract_max_tokens()
    ext_budget = _engine_max_model_len(llm_wrapper, max_model_len)
    ext_budget = max(ext_budget - ext_max_tokens - 16, 1)

    ext_prompts = []
    for index, cue in pending:
        ext_prompt = prompts[index] + (completions[index] or "") + cue
        token_ids = tokenizer.encode(ext_prompt, add_special_tokens=False)
        if len(token_ids) > ext_budget:
            head = ext_budget // 2
            token_ids = token_ids[:head] + token_ids[-(ext_budget - head):]
            ext_prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        ext_prompts.append(ext_prompt)

    ext_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=ext_max_tokens,
        stop=benchmark.extract_stop_tokens(),
        seed=seed,
    )
    try:
        ext_outputs = llm_wrapper.llm.generate(
            ext_prompts,
            sampling_params=ext_params,
            use_tqdm=False,
        )
    except Exception as exc:
        logger.warning("two-stage extraction skipped (error): %s", exc)
        return results

    recovered = 0
    for (index, _), ext_output in zip(pending, ext_outputs):
        extracted_text = ext_output.outputs[0].text
        extracted = benchmark.parse_extraction(
            extracted_text, problems[index]
        )
        if not extracted:
            continue
        is_correct, justification = benchmark.check_correctness(
            problems[index], extracted
        )
        results[index] = {
            "completion": completions[index],
            "prediction": extracted,
            "is_correct": is_correct,
            "justification": justification,
        }
        recovered += 1

    logger.info(
        "two-stage extraction ran on %d/%d rows, recovered %d answers",
        len(pending),
        len(completions),
        recovered,
    )
    return results


def run_low_temperature(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
):
    """Run single-sample low-temperature decoding over a shared benchmark."""
    logger.info("Benchmark on %s (low-temp)...", benchmark.name)
    results = []
    begin = time.time()
    total_accuracy = 0    # running count of correct predictions

    # Only pass max_model_len to vLLM when we resolved one; otherwise let vLLM use the checkpoint default.
    vllm_kwargs = {}
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len

    common_fields = {
            "method": "standard_low_temperature",
            "base_llm": model_str,
            "temperature": args.temperature,
            "max_model_len": max_model_len,
            "max_new_tokens": args.max_new_tokens,
    }

    # Load the stock vLLM engine once and reuse it for every batch.
    with vLLM_Wrapper(
        model=model_str,
        seed=args.seed,
        enable_prefix_caching=args.prefix_cache,
        engine_type="standard",
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **vllm_kwargs,
    ) as llm_wrapper:
        tokenizer = llm_wrapper.tokenizer

        # Iterate over the benchmark's batches (DataLoader yields raw rows).
        for batch in benchmark.dataset_loader:
            batch_start = benchmark.seen
            # The benchmark turns raw rows into model-ready prompts plus the per-problem objects it later needs for
            # grading.
            prompts, problems = benchmark.get_question_and_answer(
                batch, tokenizer
            )
            # Skip empty batches (e.g. a benchmark may filter a whole batch).
            if not prompts:
                continue

            started = time.time()
            logger.info(
                "BATCH %d-%d\tFIRST_INPUT: %s",
                batch_start,
                benchmark.seen - 1,
                prompts[0],
            )

            # Cap the generation budget so prompt + output fits the context window; without a known limit, use the
            # configured budget.
            max_new_tokens = batch_max_new_tokens(
                tokenizer, prompts, max_model_len, args.max_new_tokens
            )
            sampling_params = SamplingParams(
                n=1,
                temperature=args.temperature,
                max_tokens=max_new_tokens,
                top_p=args.top_p,
                stop=benchmark.stop_tokens(),
                seed=args.seed,
            )

            request_outputs = llm_wrapper.llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            completions = [out.outputs[0].text for out in request_outputs]

            # The benchmark extracts predictions and grades each completion.
            batch_results = evaluate_with_extraction(
                benchmark,
                llm_wrapper,
                tokenizer,
                prompts,
                problems,
                completions,
                max_model_len,
                args.seed,
            )
            for index, result in enumerate(batch_results):
                logger.info(
                    "sample %d response_length=%d prediction=%s "
                    "is_correct=%s justification=%s",
                    batch_start + index,
                    len(result["completion"]),
                    result["prediction"],
                    result["is_correct"],
                    result["justification"],
                )
                # Merge run config with the per-sample grading result.
                results.append({**common_fields, **result})
                total_accuracy += int(result["is_correct"])


            # Progress + ETA accounting.
            total_elapsed = time.time() - begin
            avg_per_sample = total_elapsed / benchmark.seen
            remaining = avg_per_sample * max(benchmark.size - benchmark.seen, 0)

            logger.info(
                "[%d/%d] correct=%d/%d acc=%.4f | "
                "batch=%.1fs avg/sample=%.1fs total=%.4fhours eta=%.1fsec",
                benchmark.seen,
                benchmark.size,
                total_accuracy,
                benchmark.seen,
                total_accuracy / benchmark.seen,
                time.time() - started,
                avg_per_sample,
                total_elapsed / 3600,
                remaining,
            )

        return results


# Config keys every low-temp run needs.
REQUIRED_LOW_TEMP_KEYS = ("task", "batch_size", "seed", "temperature")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven low-temperature runner. Selects a dataset from "
            "dataset.yaml and reads the 'low_temp' block of algorithms.yaml; "
            "--override changes any merged value."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="low_temp",
        algorithm_choices=["low_temp"],
        save_str_default="wilcoxon_test_mcmc_vs_standard/",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger = logging.getLogger(f"[run_{args.algorithm}_vllm]")

    task = resolve_config(args, "low_temp", REQUIRED_LOW_TEMP_KEYS)

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

    results = run_low_temperature(
        benchmark, model_str, args, max_model_len,
    )
    # Persist results to a CSV whose name encodes the full run configuration.
    os.makedirs(args.save_str, exist_ok=True)
    output_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_temp{args.temperature}"
    )
    output_name += f"_maxlen{max_model_len}_seed{args.seed}.csv"

    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name below, which no log matches.
    if getattr(args, "run_name", None):
        output_name = f"{args.run_name}.csv"
    output_filename = os.path.join(args.save_str, output_name)
    pd.DataFrame(results).to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

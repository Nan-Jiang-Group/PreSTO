"""Unified low-temperature runner using the shared benchmark interfaces.

The benchmark classes own dataset loading, prompt formatting, completion extraction, grading, and prediction formatting.
This file only configures low-temperature sampling and iterates over the selected benchmark.

These are the "standard" baselines compared against Power-SMC (``run_power_smc.py``) and Power-MCMC
(``run_power_mh.py``). Two algorithms are
provided, selected with ``--algorithm``:

* ``low_temp``  -- draw a single sample per prompt at a fixed (low) temperature and grade it.
* ``best_of_n`` -- draw ``max(best_of_N)`` i.i.d. samples per prompt, then for every ``n`` in ``best_of_N`` keep the
  candidate with the highest sequence log-probability (best-of-n) and grade it.

The same drivers work for every benchmark because each benchmark implements the same ``get_question_and_answer`` /
``evaluate_completions`` interface.

Run (GPU):
    python -m power_sharpening.runners.vllm.run_low_temp --dataset math500 --algorithm low_temp --save_str /tmp/out \
        --override temperature=0.25 max_samples=20 batch_size=1

``max_samples`` (unset: the whole dataset) caps the number of prompts, as in ``run_power_mh.py``. Each batch logs
``<algorithm>_sampler took <s> seconds for <i>-th prompts``, the same generation-time line the MH runners write, so total
sampling time can be compared across methods.

``--resource_probe`` (on by default) measures KV-cache occupancy, KV-block eviction, prefix-cache hits, and GPU memory
as the MH runners do: per batch in the CSV rows and for the whole run in ``<run>.resources.json``. Set
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` for exact occupancy and eviction counts.
"""

import os



import argparse
import json
import logging
import sys
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

# Model alias -> checkpoint mapping.
from power_sharpening.tasks.constants import MODEL_MAP
# vLLM_Wrapper: vLLM wrapper (patched engine). Low-temp uses plain generation
# (no power sampler).
from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper
from power_sharpening.backends.vllm.engine_patch import SamplingParams
# ResourceProbe measures GPU memory and KV-cache usage from the live engine, as in run_power_mh.py.
from power_sharpening.common.resource_probe import ResourceProbe
from power_sharpening.config.model_context import (
    batch_max_new_tokens,
    resolve_max_model_len,
)
# Shared helpers: benchmark factory and seeding.
from power_sharpening.tasks.registry import (
    build_benchmark,
    set_random_seed,
)
# Shared config-driven CLI: --dataset/--algorithm/--override + YAML merge.
from power_sharpening.config import add_config_selection_args, resolve_config






def _generation_budget(mh_llm, prompts, max_model_len, max_new_tokens):
    """Cap the generation budget so prompt + output fits the context window."""
    if max_model_len is None:
        return max_new_tokens
    return batch_max_new_tokens(
        mh_llm.tokenizer, prompts, max_model_len, max_new_tokens,
    )


def _get_prompts_and_problems(
    benchmark,
    batch,
    tokenizer,
):
    """Call get_question_and_answer while preserving raw problems for grading."""
    prompts, _ = benchmark.get_question_and_answer(batch, tokenizer)
    problems = benchmark.batch_to_problems(batch)
    return prompts, problems[:len(prompts)]


def _engine_kwargs(args, max_model_len: int | None) -> dict:
    """vLLM engine settings shared with the MH runners, so the KV pool is sized the same way."""
    # Only pass max_model_len to vLLM when we resolved one; otherwise let vLLM use the checkpoint default.
    vllm_kwargs = {}
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
    if gpu_memory_utilization is not None:
        vllm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    # Prefix-cache counters only exist when the engine keeps statistics, and the offline LLM entrypoint switches them
    # off by default.
    if getattr(args, "resource_probe", True):
        vllm_kwargs["disable_log_stats"] = False
    return vllm_kwargs


def _run_size(benchmark, args) -> int:
    """Number of prompts to run: the whole benchmark, or the first ``max_samples`` of it."""
    max_samples = getattr(args, "max_samples", None)
    if max_samples is None:
        return benchmark.size
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
        raise SystemExit(f"max_samples must be a positive integer or null, got {max_samples!r}.")
    return min(benchmark.size, max_samples)


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
    seen = 0              # number of prompts processed so far
    total_accuracy = 0    # running count of correct predictions
    run_size = _run_size(benchmark, args)

    vllm_kwargs = _engine_kwargs(args, max_model_len)
    # Start before model construction so the probe can tell the loaded engine footprint from generation memory.
    probe = ResourceProbe(enabled=bool(getattr(args, "resource_probe", True)))
    probe.start()

    # Load the model once and reuse it for every batch; closing() also releases the probe hooks if the run fails.
    with closing(probe), vLLM_Wrapper(
        model=model_str,
        engine_type="custom",
        seed=args.seed,
        enable_prefix_caching=args.prefix_cache,
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **vllm_kwargs,
    ) as mh_llm:
        probe.attach(mh_llm.llm)
        # Run configuration stamped onto every result row, so a single CSV is self-describing for downstream analysis.
        common_fields = {
            "method": "low_temperature",
            "base_llm": model_str,
            "temperature": args.temperature,
            "max_model_len": max_model_len,
            "max_new_tokens": args.max_new_tokens,
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
            prompts = prompts[:run_size - seen]
            problems = problems[:run_size - seen]
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

            # One sample per prompt (n defaults to 1) at the fixed temperature.
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=_generation_budget(
                    mh_llm, prompts, max_model_len, args.max_new_tokens
                ),
                seed=args.seed,
            )
            sampler_started = time.time()
            request_outputs = mh_llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )
            # Same wording as the MH runners' timing line, so one parser reads all of them.
            logger.info(
                "low_temp_sampler took %.3f seconds for %d-th prompts",
                time.time() - sampler_started,
                seen,
            )
            # Close the resource window on this batch before grading, which touches no GPU.
            resources = probe.mark()
            completions = [req_out.outputs[0].text for req_out in request_outputs]

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


def run_best_of_n_with_low_temp(
    benchmark,
    model_str: str,
    args,
    max_model_len: int | None,
):
    """Run low-temperature best-of-N over a shared benchmark implementation."""
    logger.info("Benchmark on %s (low-temp best-of-N)...", benchmark.name)
    results = []
    begin = time.time()
    seen = 0  # number of prompts processed so far
    run_size = _run_size(benchmark, args)

    # best_of_N values to report; generate max_N candidates once and reuse the first n of them for each requested n.
    best_of_N = args.best_of_N
    max_N = max(best_of_N)
    total_accuracy = {n: 0 for n in best_of_N}  # running correct count per n

    vllm_kwargs = _engine_kwargs(args, max_model_len)
    # Start before model construction so the probe can tell the loaded engine footprint from generation memory.
    probe = ResourceProbe(enabled=bool(getattr(args, "resource_probe", True)))
    probe.start()

    # Load the model once and reuse it for every batch; closing() also releases the probe hooks if the run fails.
    with closing(probe), vLLM_Wrapper(
        model=model_str,
        engine_type="custom",
        seed=args.seed,
        enable_prefix_caching=args.prefix_cache,
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **vllm_kwargs,
    ) as mh_llm:
        probe.attach(mh_llm.llm)
        # Run configuration stamped onto every result row, so a single CSV is self-describing for downstream analysis.
        common_fields = {
            "method": "best_of_n",
            "base_llm": model_str,
            "temperature": args.temperature,
            "max_model_len": max_model_len,
            "max_new_tokens": args.max_new_tokens,
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
            prompts = prompts[:run_size - seen]
            problems = problems[:run_size - seen]
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

            # Draw max_N i.i.d. candidates per prompt at the fixed temperature.
            # logprobs=0 returns no per-token logprobs but still populates each
            # candidate's cumulative_logprob, which we rank by below.
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=_generation_budget(
                    mh_llm, prompts, max_model_len, args.max_new_tokens
                ),
                n=max_N,
                logprobs=0,
                seed=args.seed,
            )
            sampler_started = time.time()
            request_outputs = mh_llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )
            logger.info(
                "best_of_n_sampler took %.3f seconds for %d-th prompts",
                time.time() - sampler_started,
                seen,
            )
            resources = probe.mark()

            # For each requested n, keep the best of the first n candidates by sequence log-prob, then grade via the
            # shared benchmark interface.
            for n in best_of_N:
                best_completions = [
                    max(
                        req_out.outputs[:n],
                        key=lambda o: o.cumulative_logprob,
                    ).text
                    for req_out in request_outputs
                ]
                batch_results = benchmark.evaluate_completions(
                    problems, best_completions
                )
                for index, result in enumerate(batch_results):
                    logger.info(
                        "sample %d [N=%d] response_length=%d prediction=%s "
                        "is_correct=%s justification=%s",
                        seen + index,
                        n,
                        len(result["completion"]),
                        result["prediction"],
                        result["is_correct"],
                        result["justification"],
                    )
                    # Merge run config + the best-of-n setting + grading result.
                    results.append({**common_fields, "best_of_n": n, **result, **resources})
                    total_accuracy[n] += int(result["is_correct"])

            # Progress + ETA accounting (accuracy reported per best-of-n).
            seen += len(prompts)
            total_elapsed = time.time() - begin
            avg_per_sample = total_elapsed / seen
            remaining = avg_per_sample * max(run_size - seen, 0)
            acc_str = " ".join(
                f"N={n}:{total_accuracy[n] / seen:.4f}" for n in best_of_N
            )
            logger.info(
                "[%d/%d] acc(%s) | "
                "batch=%.1fs avg/sample=%.1fs total=%.4fhours eta=%.1fsec",
                seen,
                run_size,
                acc_str,
                time.time() - started,
                avg_per_sample,
                total_elapsed / 3600,
                remaining,
            )

    # Run-level peaks remain available after engine teardown and probe cleanup.
    logger.info("resources: %s", probe.summary_line())
    return results, probe.summary()


# Config keys every low-temp run needs; best_of_n additionally needs best_of_N.
REQUIRED_LOW_TEMP_KEYS = ("task", "batch_size", "seed", "temperature")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven low-temperature / best-of-N runner. Selects a "
            "dataset from dataset.yaml and reads the 'low_temp' block of "
            "algorithms.yaml; --override changes any merged value. --algorithm "
            "picks single-sample (low_temp) or best-of-N (best_of_n)."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="low_temp",
        algorithm_choices=["low_temp", "best_of_n"],
        save_str_default="wilcoxon_test_mcmc_vs_standard/",
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


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger = logging.getLogger(f"[run_{args.algorithm}_vllm]")

    # best_of_n additionally needs a best_of_N ladder from the config.
    required_keys = REQUIRED_LOW_TEMP_KEYS
    if args.algorithm == "best_of_n":
        required_keys = (*REQUIRED_LOW_TEMP_KEYS, "best_of_N")

    # Both modes read the 'low_temp' block of algorithms.yaml.
    task = resolve_config(args, "low_temp", required_keys)

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

    # Dispatch to the selected algorithm's driver.
    if args.algorithm == "low_temp":
        results, resource_summary = run_low_temperature(
            benchmark, model_str, args, max_model_len,
        )
    elif args.algorithm == "best_of_n":
        results, resource_summary = run_best_of_n_with_low_temp(
            benchmark, model_str, args, max_model_len,
        )
    else:
        raise ValueError(
            f"Unknown algorithm: {args.algorithm!r}. "
            "Choose from: low_temp, best_of_n"
        )

    # Persist results to a CSV whose name encodes the full run configuration. best_of_n additionally records which
    # best-of-N set was evaluated.
    os.makedirs(args.save_str, exist_ok=True)
    output_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_temp{args.temperature}"
    )
    if args.algorithm == "best_of_n":
        best_of_n_tag = "-".join(str(n) for n in args.best_of_N)
        output_name += f"_bestofN-{best_of_n_tag}"
    output_name += f"_maxlen{max_model_len}_seed{args.seed}.csv"

    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name below, which no log matches.
    if getattr(args, "run_name", None):
        output_name = f"{args.run_name}.csv"
    output_filename = os.path.join(args.save_str, output_name)
    pd.DataFrame(results).to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

    if resource_summary.get("enabled"):
        resource_filename = output_filename.replace(".csv", ".resources.json")
        with open(resource_filename, "w") as handle:
            json.dump(
                {
                    "run": Path(output_filename).stem,
                    "method": args.algorithm,
                    "dataset": args.dataset,
                    "model": args.model_str,
                    "base_llm": model_str,
                    "temperature": args.temperature,
                    "best_of_N": args.best_of_N if args.algorithm == "best_of_n" else None,
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

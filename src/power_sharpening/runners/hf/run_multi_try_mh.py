"""Run the existing HF Multi-Try MH sampler with benchmark YAML settings.

Example (GPU):
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 \
      --cpus-per-task=24 --time=03:00:00 \
      uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      python -m power_sharpening.runners.hf.run_multi_try_mh \
      --dataset math500 --algorithm multi_try \
      --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry \
      --override num_tries=4 'proposal_temperatures=[0.25,0.5,1.0]'
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import pandas as pd

from power_sharpening.backends.hf.samplers.multi_try_mh import (
    multi_try_mcmc_power_sampler,
)
from power_sharpening.backends.hf.wrapper import (
    HF_LLM_Wrapper,
    hf_load_model_and_tokenizer,
)
from power_sharpening.config import add_config_selection_args, resolve_config
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.common.multi_try import DEFAULT_PROPOSAL_TEMPERATURES
from power_sharpening.tasks.registry import build_benchmark, set_random_seed


logger = logging.getLogger("[run_multi_try_mh_hf]")

REQUIRED_MULTI_TRY_KEYS = (
    "task", "batch_size", "seed", "alpha", "temperature", "mcmc_steps",
    "num_blocks", "max_new_tokens", "num_tries", "temperature_schedule_type",
    "cut_dist_type",
)


def _validate_and_resolve(args) -> float:
    """Validate sampler settings before loading a model; resolve temperature."""
    if (
        isinstance(args.alpha, bool) or not isinstance(args.alpha, (int, float))
        or not math.isfinite(args.alpha) or args.alpha <= 0
    ):
        raise SystemExit("alpha must be finite and positive.")
    for name, minimum in (
        ("mcmc_steps", 0), ("num_blocks", 1), ("max_new_tokens", 1),
        ("num_tries", 1), ("batch_size", 1),
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise SystemExit(f"{name} must be an integer >= {minimum}, got {value}.")
    if args.max_new_tokens % args.num_blocks:
        raise SystemExit("max_new_tokens must be divisible by num_blocks.")
    if args.cut_dist_type != "uniform":
        raise SystemExit("The existing HF Multi-Try MH sampler requires uniform cuts.")
    if args.temperature_schedule_type == "cyclic":
        raise SystemExit(
            "temperature_schedule_type='cyclic' is unsupported by the HF wrapper: "
            "the cyclic schedule requires min_temp and max_temp."
        )
    args.max_samples = getattr(args, "max_samples", None)
    if args.max_samples is not None and (
        isinstance(args.max_samples, bool)
        or not isinstance(args.max_samples, int)
        or args.max_samples <= 0
    ):
        raise SystemExit("max_samples must be a positive integer or null.")
    max_model_len = getattr(args, "max_model_len", None)
    if max_model_len is not None and (
        isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len <= 0
    ):
        raise SystemExit("max_model_len must be a positive integer or null.")

    if (
        isinstance(args.temperature, bool)
        or not isinstance(args.temperature, (int, float))
    ):
        raise SystemExit("temperature must be numeric: -1 or a finite positive value.")
    proposal_temperature = 1.0 / args.alpha if args.temperature == -1 else args.temperature
    if not math.isfinite(proposal_temperature) or proposal_temperature <= 0:
        raise SystemExit("proposal temperature must be finite and positive.")
    temperatures = getattr(args, "proposal_temperatures", None)
    if temperatures is None:
        # The HF sampler proposes from the mixture only; null selects the default.
        temperatures = list(DEFAULT_PROPOSAL_TEMPERATURES)
    if temperatures is not None:
        if not isinstance(temperatures, (list, tuple)) or not temperatures:
            raise SystemExit("proposal_temperatures must be a nonempty list.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            for value in temperatures
        ):
            raise SystemExit("proposal_temperatures must contain finite positive values.")
        if args.temperature_schedule_type != "const":
            raise SystemExit(
                "proposal_temperatures requires temperature_schedule_type='const'."
            )
        temperatures = [float(value) for value in temperatures]
    args.proposal_temperatures = temperatures
    return proposal_temperature


def _model_context_length(model, requested=None):
    """Read the loaded HF model's limit and honor any smaller configured cap."""
    config = getattr(model, "config", None)
    for field in (
        "max_position_embeddings", "max_sequence_length", "seq_length", "n_positions",
    ):
        value = getattr(config, field, None)
        if isinstance(value, (int, float)) and 0 < value < 10**9:
            return int(value) if requested is None else min(int(value), requested)
    return requested


def _fit_generation_budget(prompt_tokens, requested_tokens, num_blocks, context_length):
    """Keep generation and full-sequence HF rescoring inside the context window."""
    available = requested_tokens
    if context_length is not None:
        available = min(available, context_length - prompt_tokens)
    available -= available % num_blocks
    if available <= 0:
        raise RuntimeError(
            "No positive Multi-Try MH generation budget remains: "
            f"prompt={prompt_tokens}, context_length={context_length}, "
            f"num_blocks={num_blocks}."
        )
    return available


def run_multi_try(
    model_str, benchmark, args, proposal_temperature, *, sample_fn=None,
    method="multi_try_mh",
):
    """Sample prompts; optional ``sample_fn`` also returns prefetch statistics.

    The callback receives the shared sampler arguments plus ``sample_index`` and ``context_length`` and returns the
    existing four outputs followed by a stats object. The standalone runner keeps its original sampler contract.
    """
    device, tokenizer, model = hf_load_model_and_tokenizer(model_str)
    sampler_wrapper = HF_LLM_Wrapper(
        model, tokenizer, device, temperature=proposal_temperature, alpha=args.alpha,
    )
    context_length = _model_context_length(model, getattr(args, "max_model_len", None))
    results, all_stats = [], []
    seen = 0
    begin = time.time()
    logger.info("Benchmark on %s (%s)", benchmark.name, method)

    for batch in benchmark.dataset_loader:
        if args.max_samples is not None and seen >= args.max_samples:
            break
        benchmark.seen = seen
        # Preserve reconstructed grading data returned by each benchmark, including GPQA/MMLU choice order and
        # benchmark-specific answer keys.
        prompts, problems = benchmark.get_question_and_answer(batch, tokenizer)
        if len(prompts) != len(problems):
            raise RuntimeError("Benchmark must return one grading problem per prompt.")
        for input_text, problem in zip(prompts, problems):
            if args.max_samples is not None and seen >= args.max_samples:
                break
            started = time.time()
            # Preserve the existing HF tokenization contract. The benchmark's chat flag does not guarantee that its
            # formatter emitted a chat template (MATH/GPQA/MMLU can still return plain prompts).
            prefix = list(tokenizer.encode(input_text))
            max_new_tokens = _fit_generation_budget(
                len(prefix), args.max_new_tokens, args.num_blocks, context_length,
            )
            if max_new_tokens != args.max_new_tokens:
                logger.warning(
                    "Capped max_new_tokens from %d to %d for prompt length %d "
                    "and HF context length %d.",
                    args.max_new_tokens, max_new_tokens, len(prefix), context_length,
                )
            sampler_kwargs = dict(
                mcmc_steps=args.mcmc_steps,
                max_new_tokens=max_new_tokens,
                num_of_blocks=args.num_blocks,
                num_tries=args.num_tries,
                proposal_temperatures=args.proposal_temperatures,
                verbose=args.verbose,
            )
            prefetch_json = {}
            if sample_fn is None:
                sampled_tokens, _, target_scores, acceptance_ratio = multi_try_mcmc_power_sampler(
                    sampler_wrapper, prefix, **sampler_kwargs,
                    temperature_schedule_type=args.temperature_schedule_type,
                )
            else:
                sampled_tokens, _, target_scores, acceptance_ratio, prefetch_stats = sample_fn(
                    sampler_wrapper, prefix, **sampler_kwargs,
                    sample_index=seen, context_length=context_length,
                )
                prefetch_json = prefetch_stats.to_json(
                    mcmc_steps=args.mcmc_steps, num_blocks=args.num_blocks,
                )
            sampled_tokens = list(sampled_tokens)
            if sampled_tokens[:len(prefix)] != prefix:
                raise RuntimeError("Multi-Try MH must return the unchanged prompt prefix.")
            response_ids = sampled_tokens[len(prefix):]
            completion = tokenizer.decode(response_ids, skip_special_tokens=True)
            result = benchmark.evaluate_completions([problem], [completion])[0]
            log_likelihood = math.fsum(target_scores) / args.alpha
            confidence = math.exp(log_likelihood / len(response_ids)) if response_ids else 0.0
            elapsed = time.time() - started
            metadata = {
                "idx": seen,
                "method": method,
                "base_llm": model_str,
                "dataset": args.dataset,
                "alpha": args.alpha,
                "temperature": proposal_temperature,
                "temperature_schedule_type": args.temperature_schedule_type,
                "mcmc_steps": args.mcmc_steps,
                "num_blocks": args.num_blocks,
                "num_tries": args.num_tries,
                "proposal_temperatures": args.proposal_temperatures,
                "cut_dist_type": args.cut_dist_type,
                "max_new_tokens": max_new_tokens,
                "requested_max_new_tokens": args.max_new_tokens,
                "num_response_tokens": len(response_ids),
                "acceptance_ratio": float(acceptance_ratio),
                "log_likelihood": log_likelihood,
                "confidence": confidence,
                "elapsed_seconds": elapsed,
            }
            if sample_fn is not None:
                metadata.update({
                    "logprob_sum": log_likelihood,
                    "log_likelihood": log_likelihood / len(response_ids) if response_ids else None,
                    "mean_token_probability": metadata.pop("confidence"),
                    "prefetch_budget": args.prefetch_budget,
                    "prefetch_budget_units": "suffix_requests",
                    "bundle_capacity": args.prefetch_budget // args.num_tries,
                    "sample_seed": args.seed + seen,
                    "rank_fn": args.rank_fn,
                    "stop_on_eos": args.stop_on_eos,
                    "total_workload": prefetch_stats.total_workload,
                    "total_nfe": prefetch_stats.total_nfe,
                    "total_acceptances": prefetch_stats.total_acceptances,
                    "total_walked_steps": prefetch_stats.total_walked_steps,
                })
            results.append({
                "question": problem.get("prompt", problem.get("question", input_text)),
                "correct_answer": problem.get("answer", problem.get("correct_letter", "")),
                "pred_answer": result.get("prediction", ""),
                **result,
                **metadata,
            })
            all_stats.append({
                **prefetch_json, **metadata, "acceptance_rate": float(acceptance_ratio),
            })
            print(f"IDX {seen}\n>>> completion:\n{completion}", flush=True)
            print(
                f"response tokens: {len(response_ids)}, acceptance ratio: {acceptance_ratio:.6f}, "
                f"used time: each {elapsed:.3f} sec, total {time.time() - begin:.3f} sec",
                flush=True,
            )
            seen += 1
    return results, all_stats


def build_parser():
    parser = argparse.ArgumentParser(description="Config-driven HuggingFace Multi-Try MH runner.")
    add_config_selection_args(
        parser,
        algorithm_default="multi_try",
        algorithm_choices=["multi_try"],
        save_str_default="result/multi_try/",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    task = resolve_config(args, args.algorithm, REQUIRED_MULTI_TRY_KEYS)
    proposal_temperature = _validate_and_resolve(args)
    set_random_seed(args.seed)
    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(task, args, model_str)
    benchmark.should_use_chat_template(args.model_str)
    logger.info("resolved configuration: %s", vars(args))
    results, all_stats = run_multi_try(model_str, benchmark, args, proposal_temperature)

    temperatures_tag = (
        "scalar" if args.proposal_temperatures is None
        else "-".join(str(value) for value in args.proposal_temperatures)
    )
    out_stem = args.run_name or (
        f"{Path(model_str).name}_{benchmark.name}_multi_try_hf"
        f"_mcmc{args.mcmc_steps}_tries{args.num_tries}_blocks{args.num_blocks}"
        f"_maxnew{args.max_new_tokens}_alpha{args.alpha}_temp{proposal_temperature}"
        f"_proposals-{temperatures_tag}_schedule-{args.temperature_schedule_type}_seed{args.seed}"
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

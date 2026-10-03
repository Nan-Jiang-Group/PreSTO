"""Run the config-driven SGLang EntropyCut MH sampler.

Example:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src --extra sglang \
      python -m power_sharpening.runners.sglang.run_entropy_cut_mh \
      --dataset math500 --algorithm entropy_cut_mh
"""

import os

# SGLang/transformers can emit a progress bar for every request. Set this before importing either package so batch logs
# stay readable.
os.environ.setdefault("TQDM_DISABLE", "1")
# MH needs output_token_logprobs under the actual low-temperature proposal q. SGLang's opt-in original-logprob mode
# would instead report base-p scores and invalidate the proposal ratio.
os.environ["RETURN_ORIGINAL_LOGPROB"] = "0"

import argparse
import importlib.metadata
import json
import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from power_sharpening.backends.sglang.samplers import (
    SGL_LLM_Wrapper,
    sglang_load_model_and_tokenizer,
)
from power_sharpening.backends.sglang.samplers.entropy_cut_mh import (
    entropy_cut_mh_sampler,
)
from power_sharpening.config import add_config_selection_args, resolve_config
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.tasks.registry import build_benchmark, set_random_seed


logger = logging.getLogger("[run_entropy_cut_mh_sglang]")


def _sglang_version() -> str:
    """Return the installed SGLang version for result provenance."""
    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_safe(value):
    """Convert non-finite diagnostics into explicit JSON strings."""
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    return value


def _get_prompts_and_problems(benchmark, batch, tokenizer):
    """Format prompts and retain each benchmark's grading-ready problems."""
    prompts, problems = benchmark.get_question_and_answer(batch, tokenizer)
    return prompts, problems[: len(prompts)]


def _engine_context_length(engine) -> int | None:
    """Read SGLang's resolved context length without assuming one wrapper API."""
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    context_length = getattr(tokenizer_manager, "context_len", None)
    if context_length is None:
        server_args = getattr(engine, "server_args", None)
        context_length = getattr(server_args, "context_length", None)
    if context_length is None:
        return None
    return int(context_length)


def _fit_generation_budget(
    prompt_tokens: int,
    requested_tokens: int,
    num_blocks: int,
    context_length: int | None,
) -> int:
    """Reserve SGLang's strict one-token scoring margin and whole blocks."""
    if context_length is None:
        return requested_tokens

    # Every proposal is rescored by passing the complete fixed-length state back as input and requesting one throwaway
    # token. SGLang 0.5.2 requires input_len + max_new_tokens < context_length, hence a two-token margin relative to the
    # final state.
    available = context_length - prompt_tokens - 2
    capped = min(requested_tokens, available)
    capped -= capped % num_blocks
    if capped <= 0:
        raise RuntimeError(
            "No positive EntropyCut generation budget remains after reserving "
            f"the scoring margin: prompt={prompt_tokens}, "
            f"context_length={context_length}, num_blocks={num_blocks}."
        )
    return capped


def _truncate_at_eos(token_ids, eos_token_id):
    """Return the visible completion prefix through its first EOS token."""
    token_ids = list(token_ids)
    if eos_token_id is None:
        return token_ids
    try:
        eos_index = token_ids.index(eos_token_id)
    except ValueError:
        return token_ids
    return token_ids[: eos_index + 1]


def _validate_and_resolve(args) -> float:
    """Validate the paper-faithful configuration and return proposal temperature."""
    if not math.isfinite(float(args.alpha)) or args.alpha <= 0.0:
        raise SystemExit(f"alpha must be positive, got {args.alpha}.")
    if (
        isinstance(args.mcmc_steps, bool)
        or not isinstance(args.mcmc_steps, int)
        or args.mcmc_steps < 0
    ):
        raise SystemExit(f"mcmc_steps must be non-negative, got {args.mcmc_steps}.")
    if (
        isinstance(args.num_blocks, bool)
        or not isinstance(args.num_blocks, int)
        or args.num_blocks <= 0
    ):
        raise SystemExit(f"num_blocks must be positive, got {args.num_blocks}.")
    if (
        isinstance(args.max_new_tokens, bool)
        or not isinstance(args.max_new_tokens, int)
        or args.max_new_tokens <= 0
    ):
        raise SystemExit(
            f"max_new_tokens must be positive, got {args.max_new_tokens}."
        )
    if args.max_new_tokens % args.num_blocks != 0:
        raise SystemExit(
            "EntropyCut currently requires max_new_tokens to be divisible by "
            f"num_blocks, got {args.max_new_tokens} and {args.num_blocks}."
        )
    if args.temperature_schedule_type != "const":
        raise SystemExit(
            "EntropyCut caches proposal scores for a fixed proposal kernel; "
            "temperature_schedule_type must be 'const'."
        )

    paper_temperature = 1.0 / args.alpha
    proposal_temperature = (
        paper_temperature if args.temperature == -1.0 else args.temperature
    )
    if (
        not math.isfinite(float(proposal_temperature))
        or proposal_temperature <= 0.0
    ):
        raise SystemExit(
            f"proposal temperature must be positive, got {proposal_temperature}."
        )
    if not math.isclose(
        proposal_temperature,
        paper_temperature,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            "The EntropyCut runner implements the paper proposal temperature "
            f"1/alpha={paper_temperature}; got temperature={args.temperature}. "
            "Use --override temperature=-1."
        )

    if (
        not math.isfinite(float(args.cut_power))
        or args.cut_power < 0.0
    ):
        raise SystemExit(f"cut_power must be non-negative, got {args.cut_power}.")
    if args.entropy_mode not in {"topk", "exact"}:
        raise SystemExit(
            f"entropy_mode={args.entropy_mode!r}; choose 'topk' or 'exact'."
        )
    if args.entropy_mode == "topk":
        if (
            isinstance(args.entropy_top_k, bool)
            or not isinstance(args.entropy_top_k, int)
            or args.entropy_top_k <= 0
        ):
            raise SystemExit(
                "entropy_top_k must be a positive integer when "
                "entropy_mode='topk'."
            )
    else:
        if args.entropy_top_k is not None:
            raise SystemExit(
                "Set entropy_top_k=null when entropy_mode='exact'."
            )
        raise SystemExit(
            "entropy_mode='exact' is not available through SGLang 0.5.2's "
            "public Engine.generate response. Use entropy_mode='topk', or add "
            "a server-side scalar full-vocabulary entropy output first."
        )
    if args.max_samples is not None:
        if (
            isinstance(args.max_samples, bool)
            or not isinstance(args.max_samples, int)
            or args.max_samples <= 0
        ):
            raise SystemExit(
                "max_samples must be a positive integer or null, got "
                f"{args.max_samples}."
            )

    return proposal_temperature


def run_entropy_cut(
    model_str: str,
    benchmark,
    args,
    proposal_temperature: float,
):
    """Run EntropyCut MH over every selected benchmark prompt."""
    engine_kwargs = {"random_seed": args.seed}
    if getattr(args, "gpu_memory_utilization", None) is not None:
        # SGLang names the equivalent memory fraction differently from vLLM.
        engine_kwargs["mem_fraction_static"] = args.gpu_memory_utilization
    engine, tokenizer = sglang_load_model_and_tokenizer(
        model_str,
        **engine_kwargs,
    )

    results = []
    all_stats = []
    begin = time.time()
    seen = 0
    sglang_version = _sglang_version()
    context_length = _engine_context_length(engine)
    entropy_estimator = (
        "topk_tail_bucket" if args.entropy_mode == "topk" else "exact"
    )
    rng = np.random.default_rng(args.seed)

    with SGL_LLM_Wrapper(
        engine,
        tokenizer,
        temperature=proposal_temperature,
        alpha=args.alpha,
    ) as sampler_wrapper:
        logger.info("Benchmark on %s (EntropyCut MH)", benchmark.name)
        for batch in benchmark.dataset_loader:
            if args.max_samples is not None and seen >= args.max_samples:
                break

            benchmark.seen = seen
            prompts, problems = _get_prompts_and_problems(
                benchmark,
                batch,
                tokenizer,
            )
            for input_text, problem in zip(prompts, problems):
                if args.max_samples is not None and seen >= args.max_samples:
                    break

                started = time.time()
                prefix = list(tokenizer.encode(input_text))
                max_new_tokens = _fit_generation_budget(
                    prompt_tokens=len(prefix),
                    requested_tokens=args.max_new_tokens,
                    num_blocks=args.num_blocks,
                    context_length=context_length,
                )
                if max_new_tokens != args.max_new_tokens:
                    logger.warning(
                        "Capped max_new_tokens from %d to %d for prompt length "
                        "%d and SGLang context length %d.",
                        args.max_new_tokens,
                        max_new_tokens,
                        len(prefix),
                        context_length,
                    )
                sampled_tokens, stats = entropy_cut_mh_sampler(
                    sampler_wrapper,
                    prefix,
                    mcmc_steps=args.mcmc_steps,
                    num_of_blocks=args.num_blocks,
                    max_new_tokens=max_new_tokens,
                    cut_power=args.cut_power,
                    entropy_mode=args.entropy_mode,
                    entropy_top_k=args.entropy_top_k,
                    verbose=args.verbose,
                    rng=rng,
                )
                sampled_tokens = list(sampled_tokens)
                if sampled_tokens[: len(prefix)] != prefix:
                    raise RuntimeError(
                        "entropy_cut_mh_sampler must return the full sequence "
                        "with the input prefix unchanged."
                    )
                continuation_ids = sampled_tokens[len(prefix) :]
                visible_continuation_ids = _truncate_at_eos(
                    continuation_ids,
                    tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    visible_continuation_ids,
                    skip_special_tokens=True,
                )
                result = benchmark.evaluate_completions(
                    [problem],
                    [completion],
                )[0]

                if not isinstance(stats, Mapping):
                    raise TypeError(
                        "entropy_cut_mh_sampler stats must be a mapping, got "
                        f"{type(stats).__name__}."
                    )
                stats_row = dict(stats)
                stats_row.update(
                    {
                        "idx": seen,
                        "dataset": args.dataset,
                        "sglang_version": sglang_version,
                        "entropy_mode": args.entropy_mode,
                        "entropy_estimator": entropy_estimator,
                        "entropy_top_k": args.entropy_top_k,
                        "cut_power": args.cut_power,
                    }
                )
                all_stats.append(stats_row)

                elapsed = time.time() - started
                results.append(
                    {
                        "method": "entropy_cut_mh",
                        "backend": "sglang",
                        "sglang_version": sglang_version,
                        "base_llm": model_str,
                        "dataset": args.dataset,
                        "alpha": args.alpha,
                        "proposal_temperature": proposal_temperature,
                        "temperature_schedule_type": (
                            args.temperature_schedule_type
                        ),
                        "mcmc_steps": args.mcmc_steps,
                        "num_blocks": args.num_blocks,
                        "configured_max_new_tokens": args.max_new_tokens,
                        "max_new_tokens": max_new_tokens,
                        "cut_power": args.cut_power,
                        "entropy_mode": args.entropy_mode,
                        "entropy_estimator": entropy_estimator,
                        "entropy_top_k": args.entropy_top_k,
                        "seed": args.seed,
                        "acceptance_rate": stats_row.get("acceptance_rate"),
                        "mean_cut_position": stats_row.get("mean_cut_position"),
                        "mean_cut_fraction": stats_row.get("mean_cut_fraction"),
                        "mean_token_entropy": stats_row.get(
                            "mean_token_entropy"
                        ),
                        "output_tokens": stats_row.get(
                            "output_tokens",
                            len(continuation_ids),
                        ),
                        "visible_output_tokens": len(
                            visible_continuation_ids
                        ),
                        "sequence_base_logprob": stats_row.get(
                            "sequence_base_logprob"
                        ),
                        "sequence_target_log_score": stats_row.get(
                            "sequence_target_log_score"
                        ),
                        "sequence_proposal_logprob": stats_row.get(
                            "sequence_proposal_logprob"
                        ),
                        "runtime_seconds": elapsed,
                        **result,
                    }
                )

                seen += 1
                total_elapsed = time.time() - begin
                logger.info(
                    "[%d/%d] response_tokens=%d prediction=%s "
                    "is_correct=%s sample=%.1fs total=%.2fh",
                    seen,
                    benchmark.size,
                    len(continuation_ids),
                    result["prediction"],
                    result["is_correct"],
                    elapsed,
                    total_elapsed / 3600.0,
                )

    return results, all_stats


REQUIRED_ENTROPY_CUT_KEYS = (
    "task",
    "batch_size",
    "seed",
    "alpha",
    "temperature",
    "temperature_schedule_type",
    "mcmc_steps",
    "num_blocks",
    "max_new_tokens",
    "cut_power",
    "entropy_mode",
)


def build_parser():
    """Build the YAML-config-driven EntropyCut CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven SGLang EntropyCut MH runner. Select a dataset from "
            "dataset.yaml and use --override KEY=VALUE for run-specific values."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="entropy_cut_mh",
        algorithm_choices=["entropy_cut_mh"],
        save_str_default="entropy_cut_mh_results/",
    )
    return parser


def main():
    """Resolve configuration, run the benchmark, and save CSV/JSONL artifacts."""
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    task = resolve_config(
        args,
        args.algorithm,
        REQUIRED_ENTROPY_CUT_KEYS,
    )
    if not hasattr(args, "entropy_top_k"):
        args.entropy_top_k = None
    if not hasattr(args, "max_samples"):
        args.max_samples = None
    proposal_temperature = _validate_and_resolve(args)

    set_random_seed(args.seed)
    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(task, args, model_str)
    use_chat_template = benchmark.should_use_chat_template(args.model_str)
    logger.info(
        "loading %s of length %d; prompt style=%s",
        benchmark.name,
        benchmark.size,
        "chat template" if use_chat_template else "raw",
    )
    logger.info(
        "alpha=%s proposal_temperature=%s blocks=%s steps=%s "
        "cut_power=%s entropy=%s top_k=%s",
        args.alpha,
        proposal_temperature,
        args.num_blocks,
        args.mcmc_steps,
        args.cut_power,
        args.entropy_mode,
        args.entropy_top_k,
    )

    results, all_stats = run_entropy_cut(
        model_str,
        benchmark,
        args,
        proposal_temperature,
    )

    save_dir = Path(args.save_str)
    save_dir.mkdir(parents=True, exist_ok=True)
    top_k_label = (
        f"topk{args.entropy_top_k}"
        if args.entropy_mode == "topk"
        else "exact"
    )
    output_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}_sglang"
        f"_mcmc{args.mcmc_steps}_blocks{args.num_blocks}"
        f"_maxnew{args.max_new_tokens}_beta{args.cut_power}"
        f"_{top_k_label}"
        f"_alpha{args.alpha}_seed{args.seed}.csv"
    )
    # --run_name is the launcher's stem, the one its .log already carries, so
    # every artifact of a run shares one prefix. Unset keeps the derived name above, which no log matches.
    if getattr(args, "run_name", None):
        output_name = f"{args.run_name}.csv"
    output_path = save_dir / output_name
    pd.DataFrame(results).to_csv(output_path, index=False)
    logger.info("output saved to %s", output_path)

    stats_path = output_path.with_suffix(".stats.jsonl")
    with stats_path.open("w") as handle:
        for stats_row in all_stats:
            json.dump(
                _json_safe(stats_row),
                handle,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    logger.info("stats saved to %s", stats_path)


if __name__ == "__main__":
    main()

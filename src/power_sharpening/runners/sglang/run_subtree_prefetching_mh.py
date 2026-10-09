"""Run SGLang subtree MH; pass --evict_subtree_cache to prune discarded prefixes."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd


from power_sharpening.tasks.constants import (
    MAX_NEW_TOKENS,
    MODEL_MAP,
)

from power_sharpening.backends.sglang.samplers import (
    SGL_LLM_Wrapper,
    sglang_load_model_and_tokenizer,
)
from power_sharpening.backends.sglang.kv_cache_stats import SglangKvCacheStats
from power_sharpening.backends.sglang.samplers.subtree_prefetching_sampler import subtree_prefetching_sampling
from power_sharpening.common.prefetch_rank import RANK_FNS
from power_sharpening.tasks.registry import (
    add_benchmark_args,
    add_benchmark_selection_args,
    build_benchmark,
    resolve_task,
    set_random_seed,
)

maxi_iters = 10

logger = logging.getLogger("[run_subtree_prefetching_sglang]")


def _log_run_configuration(args, **derived_config):
    """Log the resolved CLI configuration before the run starts."""
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


def run_subtree_prefetching(
        model_str,
        benchmark,
        alpha: float,
        num_blocks: int,
        mcmc_steps: int,
        prefetch_budget: int,
        verbose: bool,
        rank_fn: str = "accept_first",
        evict_subtree_cache: bool = False,
        print_tree: bool = False,
        mem_fraction_static: float | None = None,
        engine_kwargs: dict | None = None,
):
    engine_kwargs = dict(engine_kwargs or {})
    # Unset keeps SGLang's own default; lower values leave more headroom for the batched scoring pass's logits.
    if mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = mem_fraction_static
    engine, tokenizer = sglang_load_model_and_tokenizer(
        model_str, enable_subtree_cache_eviction=evict_subtree_cache, **engine_kwargs,
    )
    kv_stats = SglangKvCacheStats(engine)
    kv_stats.log_capacity()
    sampler_wrapper = SGL_LLM_Wrapper(
        engine, tokenizer,
        temperature=1.0 / alpha,
        alpha=alpha,
    )

    logger.info(
        "Benchmark on %s (subtree prefetching MH, SGLang), temperature=%.6f",
        benchmark.name,
        sampler_wrapper.temperature,
    )
    results = []
    all_stats = []
    begin = time.time()

    seen = 0
    for batch in benchmark.dataset_loader:
        if seen >= maxi_iters:
            break
        benchmark.seen = seen
        prompts, problems = _get_prompts_and_problems(
            benchmark,
            batch,
            tokenizer,
        )
        input_text, problem = prompts[0], problems[0]

        st = time.time()
        question = problem["prompt"]
        # Math-style benchmarks carry a scalar "answer"; code benchmarks (LCB, HumanEval, MBPP) grade against
        # the full problem dict (e.g. "test") instead and have no such key.
        answer = problem.get("answer", problem.get("correct_letter", problem.get("task_id")))
        print("IDX", seen, "\t QUESTION:", question.split("\n")[0], flush=True)
        prefix = tokenizer.encode(input_text)

        sampler_started = time.time()
        sampled_tokens, stats = subtree_prefetching_sampling(
            sampler_wrapper,
            prompt=prefix,
            mcmc_steps=mcmc_steps,
            max_batch_size=prefetch_budget,
            rank_fn=rank_fn,
            num_of_blocks=num_blocks,
            max_new_tokens=MAX_NEW_TOKENS,
            verbose=verbose,
            evict_subtree_cache=evict_subtree_cache,
            print_tree=print_tree,
        )

        # Same wording as the vLLM runner; the SGLang sampler returns prompt + response, so subtract the prompt.
        logger.info(
            "subtree_prefetching_sampling took %.3f seconds "
            "for %d-th prompts, response tokens: %d",
            time.time() - sampler_started,
            seen,
            len(sampled_tokens) - len(prefix),
        )
        # Grade and save only the response, as the vLLM runner does.
        completion = tokenizer.decode(sampled_tokens[len(prefix):], skip_special_tokens=True)
        print(">>> completion:\n", completion, flush=True)
        print("=" * 30)
        result = benchmark.evaluate_completions([problem], [completion])[0]

        logger.info(
            "response length: %d, response tokens: %d",
            len(completion),
            len(sampled_tokens) - len(prefix),
        )
        if result["prediction"]:
            logger.info(
                "\tpred length: %d, pred: %s, is_correct: %s",
                len(result["prediction"]),
                result["prediction"],
                result["is_correct"],
            )
        else:
            logger.info("\tpred is empty")
        logger.info("correct answer: %s", answer)
        stats_json = stats.to_json(mcmc_steps=mcmc_steps, num_blocks=num_blocks)
        stats_json["idx"] = seen
        stats_json["evict_subtree_cache"] = evict_subtree_cache
        resources = kv_stats.mark(seen)
        stats_json.update(resources)
        all_stats.append(stats_json)
        results.append(
            {
                "method": "subtree_prefetching_MH",
                "base_llm": model_str,
                "question": question,
                "correct_answer": answer,
                "pred_answer": result["prediction"],
                **result,
                "alpha": alpha,
                "mcmc_steps": mcmc_steps,
                "prefetch_budget": prefetch_budget,
                "rank_fn": rank_fn,
                "evict_subtree_cache": evict_subtree_cache,
                "acceptance_ratio": stats_json.get("acceptance_rate", 0.0),
                "total_workload": stats.total_workload,
                "total_nfe": stats.total_nfe,
                "total_acceptances": stats.total_acceptances,
                **resources,
            }
        )

        seen += 1
        now = time.time()
        logger.info("used time: each %.3f sec, total %.3f sec", now - st, now - begin)
        print("-" * 80)

    logger.info("resources: %s", kv_stats.summary_line())
    return results, all_stats, kv_stats.summary()


if __name__ == "__main__":
    # Discover --task before registering its task-specific benchmark flags.
    task = resolve_task()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", type=str, default="results/")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Basename for this run's outputs. Pass the stem the launcher "
            "gives the run's .log so both share one prefix; defaults to a "
            "name derived from the run configuration."
        ),
    )
    parser.add_argument("--model_str", type=str, default="qwen")

    # Benchmark selection (--task / --batch_size). Task-specific options (aime/lcb) are registered below via
    # add_benchmark_args once the task is known.
    add_benchmark_selection_args(parser)

    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--prefetch_budget", type=int, default=10)
    parser.add_argument(
        "--rank_fn", type=str, default="accept_first", choices=sorted(RANK_FNS),
        help="Proposal-subtree frontier ranking mode from power_sharpening.common.prefetch_rank.RANK_FNS.",
    )
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument(
        "--evict_subtree_cache", action=argparse.BooleanOptionalAction, default=False,
        help="Evict discarded radix-cache prefixes after MH selection (SGLang 0.5.2); default off.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_samples", type=int, default=maxi_iters,
        help="Number of benchmark problems to run.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
        help="Generation budget (tokens) per completion.",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument(
        "--engine_log_level", type=str, default="info",
        help=(
            "SGLang engine log level. 'info' makes its scheduler log KV-cache token usage and cached tokens on every "
            "Prefill batch and on every --decode_log_interval-th Decode batch; SGLang's own Engine default is 'error'."
        ),
    )
    parser.add_argument(
        "--decode_log_interval", type=int, default=200,
        help="Decode iterations between the SGLang scheduler's Decode batch lines (SGLang default 40).",
    )
    parser.add_argument(
        "--mem_fraction_static", type=float, default=None,
        help="Fraction of GPU memory SGLang reserves for weights + KV cache; default: SGLang's choice.",
    )
    parser.add_argument(
        "--print_tree", action=argparse.BooleanOptionalAction, default=False,
        help="Print each prefetched proposal tree and its per-request features in the run log.",
    )

    # Register only the selected task's benchmark-specific options.
    add_benchmark_args(parser, task)

    args = parser.parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    set_random_seed(args.seed)
    # The samplers read these module globals.
    maxi_iters = args.max_samples
    MAX_NEW_TOKENS = args.max_new_tokens

    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(args.task, args, model_str)
    # Sets benchmark.use_chat_template, which format_prompt reads. The scorer reads this line back to rebuild prompts.
    use_chat_template = benchmark.should_use_chat_template(args.model_str)
    logger.info(
        "prompt style for %s: %s",
        args.model_str,
        "chat template" if use_chat_template else "raw",
    )
    logger.info("loading %s of length %d", benchmark.name, benchmark.size)
    _log_run_configuration(
        args,
        benchmark_name=benchmark.name,
        benchmark_size=benchmark.size,
        model_str=model_str,
        task=args.task,
    )

    results, all_stats, resource_summary = run_subtree_prefetching(
        model_str=model_str,
        benchmark=benchmark,
        alpha=args.alpha,
        mcmc_steps=args.mcmc_steps,
        prefetch_budget=args.prefetch_budget,
        rank_fn=args.rank_fn,
        num_blocks=args.num_blocks,
        verbose=args.verbose,
        evict_subtree_cache=args.evict_subtree_cache,
        print_tree=args.print_tree,
        mem_fraction_static=args.mem_fraction_static,
        engine_kwargs={
            "log_level": args.engine_log_level,
            "decode_log_interval": args.decode_log_interval,
        },
    )

    df = pd.DataFrame(results)
    os.makedirs(args.save_str, exist_ok=True)

    out_name = (
        f"{Path(model_str).name}_{benchmark.name.lower()}"
        f"_mcmc{args.mcmc_steps}_batch{args.prefetch_budget}"
        f"_alpha{args.alpha}_seed{args.seed}"
        f"{'_cache-evict' if args.evict_subtree_cache else ''}.csv"
    )
    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name above, which no log matches.
    if getattr(args, "run_name", None):
        out_name = f"{args.run_name}.csv"
    output_filename = os.path.join(args.save_str, out_name)

    df.to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

    stats_filename = output_filename.replace(".csv", ".stats.json")
    with open(stats_filename, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("stats saved to %s", stats_filename)

    resource_filename = output_filename.replace(".csv", ".resources.json")
    with open(resource_filename, "w") as handle:
        json.dump(
            {"run": Path(output_filename).stem, "backend": "sglang", **resource_summary},
            handle,
            indent=2,
        )
    logger.info("resource metrics saved to %s", resource_filename)

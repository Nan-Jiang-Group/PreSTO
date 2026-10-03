"""Run SGLang subtree MH; pass --evict_subtree_cache to prune discarded prefixes."""

import argparse
import json
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
from power_sharpening.backends.sglang.samplers.subtree_prefetching_sampler import subtree_prefetching_sampling
from power_sharpening.tasks.registry import (
    add_benchmark_args,
    add_benchmark_selection_args,
    build_benchmark,
    resolve_task,
    set_random_seed,
)

maxi_iters = 10


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
        evict_subtree_cache: bool = False,
):
    engine, tokenizer = sglang_load_model_and_tokenizer(
        model_str, enable_subtree_cache_eviction=evict_subtree_cache,
    )
    sampler_wrapper = SGL_LLM_Wrapper(
        engine, tokenizer,
        temperature=1.0 / alpha,
        alpha=alpha,
    )

    print(f"Benchmark on {benchmark.name} (subtree prefetching MH)...")
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
        answer = problem["answer"]
        print("IDX", seen, "\t QUESTION:", question.split("\n")[0], flush=True)
        prefix = tokenizer.encode(input_text)

        sampled_tokens, stats = subtree_prefetching_sampling(
            sampler_wrapper,
            prompt=prefix,
            mcmc_steps=mcmc_steps,
            max_batch_size=prefetch_budget,
            num_of_blocks=num_blocks,
            max_new_tokens=MAX_NEW_TOKENS,
            verbose=verbose,
            evict_subtree_cache=evict_subtree_cache,
        )

        completion = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
        print(">>> completion:\n", completion[len(input_text):], flush=True)
        print("=" * 30)
        result = benchmark.evaluate_completions([problem], [completion])[0]

        print(f"response length: {len(completion)}", flush=True)
        if result["prediction"]:
            print(
                f"\tpred length: {len(result['prediction'])}, "
                f"pred: {result['prediction']}, is_correct: {result['is_correct']}",
                flush=True,
            )
        else:
            print("\tpred is empty", flush=True)
        print("correct answer:", answer)
        stats_json = stats.to_json(mcmc_steps=mcmc_steps, num_blocks=num_blocks)
        stats_json["idx"] = seen
        stats_json["evict_subtree_cache"] = evict_subtree_cache
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
                "evict_subtree_cache": evict_subtree_cache,
                "acceptance_ratio": stats_json.get("acceptance_rate", 0.0),
                "total_workload": stats.total_workload,
                "total_nfe": stats.total_nfe,
                "total_acceptances": stats.total_acceptances,
            }
        )

        seen += 1
        now = time.time()
        print(f"used time: each {now - st:.3f} sec, total {now - begin:.3f} sec", flush=True)
        print("-" * 80)

    return results, all_stats


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
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument(
        "--evict_subtree_cache", action=argparse.BooleanOptionalAction, default=False,
        help="Evict discarded radix-cache prefixes after MH selection (SGLang 0.5.2); default off.",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

    # Register only the selected task's benchmark-specific options.
    add_benchmark_args(parser, task)

    args = parser.parse_args()
    print(args)

    set_random_seed(args.seed)

    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(args.task, args, model_str)
    # Sets benchmark.use_chat_template, which format_prompt reads.
    benchmark.should_use_chat_template(args.model_str)
    print(f"loading {benchmark.name} of length {benchmark.size}")

    results, all_stats = run_subtree_prefetching(
        model_str=model_str,
        benchmark=benchmark,
        alpha=args.alpha,
        mcmc_steps=args.mcmc_steps,
        prefetch_budget=args.prefetch_budget,
        num_blocks=args.num_blocks,
        verbose=args.verbose,
        evict_subtree_cache=args.evict_subtree_cache,
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
    print("output saved to", output_filename)

    stats_filename = output_filename.replace(".csv", ".stats.json")
    with open(stats_filename, "w") as f:
        json.dump(all_stats, f, indent=2)
    print("stats saved to", stats_filename)

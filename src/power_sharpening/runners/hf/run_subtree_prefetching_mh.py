"""Run subtree-prefetching MH with the HuggingFace backend.

Run on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.hf.run_subtree_prefetching_mh \
      --dataset math500 --algorithm subtree_prefetching_mh
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from power_sharpening.common.prefetch_rank import RANK_FNS
from power_sharpening.config import add_config_selection_args, resolve_config
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.tasks.registry import (
    build_benchmark,
    set_random_seed,
)

from power_sharpening.backends.hf.wrapper import (
    HF_LLM_Wrapper,
    hf_load_model_and_tokenizer,
)
from power_sharpening.backends.hf.samplers.subtree_prefetching_sampler import (
    subtree_prefetching_sampling,
)

device_name = "cuda:0"
logger = logging.getLogger("[run_subtree_prefetching_hf]")


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
    return prompts, problems[: len(prompts)]


def run_subtree_prefetching(
    model_str,
    benchmark,
    args,
):
    alpha = args.alpha
    device, tokenizer, hf_model = hf_load_model_and_tokenizer(model_str, device_name)
    sampler_wrapper = HF_LLM_Wrapper(
        hf_model,
        tokenizer,
        device,
        temperature=1.0 / alpha,
        alpha=alpha,
    )

    logger.info("Benchmark on %s (subtree prefetching MH)...", benchmark.name)
    results = []
    all_stats = []
    begin = time.time()

    seen = 0
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

            st = time.time()
            question = problem["prompt"]
            answer = problem["answer"]
            print("IDX", seen, "\t QUESTION:", question.split("\n")[0], flush=True)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            prefix = [int(t.item()) for t in input_ids[0]]

            sampled_tokens, stats = subtree_prefetching_sampling(
                sampler_wrapper,
                prompt=prefix,
                mcmc_steps=args.mcmc_steps,
                max_batch_size=args.prefetch_budget,
                num_of_blocks=args.num_blocks,
                max_new_tokens=args.max_new_tokens,
                rank_fn=args.rank_fn,
                cut_dist_type=args.cut_dist_type,
                cut_dist_param=getattr(args, "cut_dist_param", None),
                print_tree=args.print_tree,
                verbose=args.verbose,
            )
            response_ids = sampled_tokens[len(prefix):]
            completion = tokenizer.decode(response_ids, skip_special_tokens=True)
            print(">>> completion:\n" + completion, flush=True)
            print("=" * 30)
            result = benchmark.evaluate_completions([problem], [completion])[0]

            diagnostics = stats.base_diagnostics
            print(
                f"response length: {len(completion)}, "
                f"response tokens: {diagnostics.num_tokens}, "
                f"log-likelihood: {diagnostics.log_likelihood:.6f}, "
                f"confidence: {diagnostics.confidence:.6f}",
                flush=True,
            )

            if result["prediction"]:
                print(
                    f"\tpred length: {len(result['prediction'])}, "
                    f"pred: {result['prediction']}, "
                    f"is_correct: {result['is_correct']}",
                    flush=True,
                )
            else:
                print("\tpred is empty", flush=True)
            print("correct answer:", answer)
            stats_json = stats.to_json(
                mcmc_steps=args.mcmc_steps,
                num_blocks=args.num_blocks,
            )
            stats_json["idx"] = seen
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
                    "mcmc_steps": args.mcmc_steps,
                    "prefetch_budget": args.prefetch_budget,
                    "rank_fn": args.rank_fn,
                    "cut_dist_type": args.cut_dist_type,
                    "cut_dist_param": getattr(args, "cut_dist_param", None),
                    "num_blocks": args.num_blocks,
                    "max_new_tokens": args.max_new_tokens,
                    "num_response_tokens": diagnostics.num_tokens,
                    "log_likelihood": diagnostics.log_likelihood,
                    "confidence": diagnostics.confidence,
                    "acceptance_ratio": stats_json.get("acceptance_rate", 0.0),
                    "total_workload": stats.total_workload,
                    "total_nfe": stats.total_nfe,
                    "total_acceptances": stats.total_acceptances,
                }
            )

            seen += 1
            now = time.time()
            print(
                f"used time: each {now - st:.3f} sec, total {now - begin:.3f} sec",
                flush=True,
            )
            print("-" * 80)

    return results, all_stats


REQUIRED_SUBTREE_PREFETCHING_KEYS = (
    "task",
    "batch_size",
    "seed",
    "alpha",
    "mcmc_steps",
    "prefetch_budget",
    "num_blocks",
    "max_new_tokens",
    "rank_fn",
    "cut_dist_type",
    "print_tree",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven HF subtree-prefetching MH runner. Selects a dataset "
            "+ algorithm from dataset.yaml / algorithms.yaml; use --override "
            "to change merged values from the CLI."
        ),
    )
    add_config_selection_args(
        parser,
        algorithm_default="subtree_prefetching_mh",
        algorithm_choices=["subtree_prefetching_mh"],
        save_str_default="subtree_prefetching_mh_results/",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    task = resolve_config(
        args,
        args.algorithm,
        REQUIRED_SUBTREE_PREFETCHING_KEYS,
    )
    if args.rank_fn not in RANK_FNS:
        raise SystemExit(f"rank_fn={args.rank_fn!r} not in {sorted(RANK_FNS)}.")
    if args.cut_dist_type not in ("uniform", "entropy"):
        raise SystemExit(
            f"cut_dist_type={args.cut_dist_type!r} not in ['entropy', 'uniform']."
        )
    if not hasattr(args, "max_samples"):
        args.max_samples = None
    set_random_seed(args.seed)

    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(task, args, model_str)
    logger.info("loading %s of length %d", benchmark.name, benchmark.size)
    # Same prompt policy as the PowerMH runner, so both methods see identical prompts. Sets benchmark.use_chat_template,
    # which format_prompt reads.
    use_chat_template = benchmark.should_use_chat_template(args.model_str)
    logger.info(
        "prompt style for %s: %s",
        args.model_str,
        "chat template" if use_chat_template else "raw",
    )
    _log_run_configuration(
        args,
        benchmark_name=benchmark.name,
        benchmark_size=benchmark.size,
        model_str=model_str,
        task=task,
    )

    results, all_stats = run_subtree_prefetching(
        model_str=model_str,
        benchmark=benchmark,
        args=args,
    )

    df = pd.DataFrame(results)
    os.makedirs(args.save_str, exist_ok=True)

    # The cut law (and its shape parameter, when pinned) belongs in the name so runs that differ only by cut policy do
    # not overwrite each other.
    cut_dist_param = getattr(args, "cut_dist_param", None)
    cut_tag = f"_cut-{args.cut_dist_type}"
    if cut_dist_param is not None:
        cut_tag += f"{cut_dist_param}"

    # --run_name is the launcher's stem, the one its .log and the case-study
    # PDFs already carry, so every artifact of a run shares one prefix. Without it the name is derived here, in a layout
    # no parser reads and no log matches -- fine for an ad-hoc run, useless for pairing.
    out_stem = getattr(args, "run_name", None) or (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_hf"
        f"_mcmc{args.mcmc_steps}_batch{args.prefetch_budget}"
        f"_blocks{args.num_blocks}_maxnew{args.max_new_tokens}"
        f"_rank-{args.rank_fn}"
        f"{cut_tag}"
        f"_alpha{args.alpha}_seed{args.seed}"
    )
    output_filename = os.path.join(args.save_str, f"{out_stem}.csv")

    df.to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

    stats_filename = output_filename.replace(".csv", ".stats.json")
    with open(stats_filename, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("stats saved to %s", stats_filename)

import os
import time
from pathlib import Path
import argparse
import pandas as pd


from power_sharpening.tasks.constants import (
    MAX_NEW_TOKENS,
    MODEL_MAP,
)

from power_sharpening.backends.sglang.samplers import (
    SGL_LLM_Wrapper,
    sglang_load_model_and_tokenizer,
)
from power_sharpening.backends.sglang.samplers.power_samp_utils import mcmc_power_sampler
from power_sharpening.backends.sglang.samplers.base_lm_sampler import (
    base_LLM_sampling,
    low_temperature_sampling,
)
from power_sharpening.tasks.registry import (
    add_benchmark_args,
    add_benchmark_selection_args,
    build_benchmark,
    resolve_task,
    set_random_seed,
)

maxi_iters = 200


def _get_prompts_and_problems(
    benchmark,
    batch,
    tokenizer,
):
    """Call get_question_and_answer while preserving raw problems for grading."""
   
    prompts, _ = benchmark.get_question_and_answer(batch, tokenizer)
    problems = benchmark.batch_to_problems(batch)
    return prompts, problems[:len(prompts)]


def low_temperature_sampler(
    model_str,
    benchmark,
    temp: float,
    temperature_schedule_type: str = "const",
):
    engine, tokenizer = sglang_load_model_and_tokenizer(model_str)
    base_lm_wrapper = SGL_LLM_Wrapper(engine, tokenizer, temperature=temp)
    results = []
    print(f"Benchmark on {benchmark.name} (low-temp)")
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

        input_ids = tokenizer.encode(input_text)

        completion = low_temperature_sampling(base_lm_wrapper, input_ids, MAX_NEW_TOKENS)
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
        results.append(
            {
                "method": "low_temperature",
                "temperature": temp,
                "base_llm": model_str,
                "question": question,
                "correct_answer": answer,
                "pred_answer": result["prediction"],
                **result,
            }
        )

        seen += 1
        now = time.time()
        print(
            f"used time: each {now - st:.3f} sec, total {now - begin:.3f} sec",
            flush=True,
        )
        print("-" * 80)

    return results


def standard_baseLLM_sampler(
    model_str,
    benchmark,
    temperature_schedule_type: str = "const",
):
    engine, tokenizer = sglang_load_model_and_tokenizer(model_str)
    base_lm_wrapper = SGL_LLM_Wrapper(engine, tokenizer)
    results = []
    begin = time.time()
    print(f"Benchmark on {benchmark.name} (standard)")
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

        input_ids = tokenizer.encode(input_text)

        completion = base_LLM_sampling(base_lm_wrapper, input_ids, MAX_NEW_TOKENS)
        result = benchmark.evaluate_completions([problem], [completion])[0]
        if result["prediction"]:
            print(
                f"\tpred length: {len(result['prediction'])}, "
                f"pred: {result['prediction']}, is_correct: {result['is_correct']}",
                flush=True,
            )
        else:
            print("\tpred is empty", flush=True)
        print("correct answer:", answer)
        results.append(
            {
                "method": "standard",
                "base_llm": model_str,
                "question": question,
                "correct_answer": answer,
                "pred_answer": result["prediction"],
                **result,
            }
        )
        seen += 1
        now = time.time()
        print(
            f"used time: each {now - st:.3f} sec, total {now - begin:.3f} sec",
            flush=True,
        )
        print("-" * 80, flush=True)

    return results


def power_sampling(
    model_str,
    benchmark,
    alpha: float,
    mcmc_steps: int,
    num_blocks: int,
    temperature_schedule_type: str = "const",
    init_temperature: float = 0.5,
    cut_dist_type: str = "uniform",
    verbose=False,
):
    engine, tokenizer = sglang_load_model_and_tokenizer(model_str)
    base_sampler = SGL_LLM_Wrapper(
        engine, tokenizer, temperature=init_temperature, alpha=alpha
    )
    print("loaded base_sampler", base_sampler)

    print(f"Benchmark on {benchmark.name} (power-MCMC)...")
    results = []
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
        mcmc_tokens, _, _, acceptance_ratio = mcmc_power_sampler(
            base_sampler,
            prefix,
            mcmc_steps,
            num_of_blocks=num_blocks,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature_schedule_type=temperature_schedule_type,
            cut_dist_type=cut_dist_type,
            verbose=True,
        )

        # mcmc_tokens is the full prefix+completion sequence of token ids.
        completion = tokenizer.decode(mcmc_tokens, skip_special_tokens=True)
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
        results.append(
            {
                "method": "power_sampling_mcmc",
                "base_llm": model_str,
                "question": question,
                "correct_answer": answer,
                "pred_answer": result["prediction"],
                **result,
                "alpha": alpha,
                "mcmc_steps": mcmc_steps,
                "cut_dist_type": cut_dist_type,
                "acceptance_ratio": acceptance_ratio,
            }
        )
        seen += 1
        now = time.time()
        print(
            f"used time: each {now - st:.3f} sec, total {now - begin:.3f} sec",
            flush=True,
        )
        print("-" * 80, flush=True)

    return results


if __name__ == "__main__":
    # Discover --task before registering its task-specific benchmark flags.
    task = resolve_task()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_str", type=str, default="wilcoxon_test_mcmc_vs_standard/"
    )
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

    parser.add_argument(
        "--algorithm",
        type=str,
        default="standard",
        help="Which algorithm to run and generate results",
    )

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument(
        "--cut_dist_type",
        type=str,
        default="uniform",
        choices=("uniform",),
        help="State-independent cut-index distribution used by PowerMH.",
    )
    parser.add_argument(
        "--temperature_schedule_type",
        type=str,
        default="const",
        choices=[
            "const",
            "cosine",
            "step",
            "exp_decay",
            "cyclic",
            "linear",
            "cosine_warm_restarts",
        ],
        help="Temperature schedule type",
    )

    parser.add_argument("--seed", type=int, default=0)

    # Register only the selected task's benchmark-specific options.
    add_benchmark_args(parser, task)

    args = parser.parse_args()
    set_random_seed(args.seed)

    model_str = MODEL_MAP[args.model_str]
    benchmark = build_benchmark(args.task, args, model_str)
    # Sets benchmark.use_chat_template, which format_prompt reads.
    benchmark.should_use_chat_template(args.model_str)
    print(f"loading {benchmark.name} of length {benchmark.size}")
    results = None
    if args.algorithm == "low_temp":
        results = low_temperature_sampler(
            model_str,
            benchmark,
            temp=args.temperature,
            temperature_schedule_type=args.temperature_schedule_type,
        )
    if args.algorithm == "standard":
        results = standard_baseLLM_sampler(
            model_str,
            benchmark,
            temperature_schedule_type=args.temperature_schedule_type,
        )
    if args.algorithm == "power_mcmc":
        results = power_sampling(
            model_str,
            benchmark,
            alpha=args.alpha,
            mcmc_steps=args.mcmc_steps,
            num_blocks=args.num_blocks,
            temperature_schedule_type=args.temperature_schedule_type,
            init_temperature=args.temperature,
            cut_dist_type=args.cut_dist_type,
        )

    df = pd.DataFrame(results)

    save_dir = args.save_str
    os.makedirs(save_dir, exist_ok=True)

    cut_tag = ""
    if args.algorithm == "power_mcmc" and args.cut_dist_type != "uniform":
        cut_tag = f"_cut-{args.cut_dist_type}"
    out_name = (
        f"{Path(model_str).name}_{benchmark.name}_{args.algorithm}"
        f"_mcmc{args.mcmc_steps}_temp{args.temperature}_alpha{args.alpha}"
        f"{cut_tag}_seed{args.seed}.csv"
    )
    # --run_name is the launcher's stem, the one its .log already carries,
    # so every artifact of a run shares one prefix. Unset keeps the derived name above, which no log matches.
    if getattr(args, "run_name", None):
        out_name = f"{args.run_name}.csv"
    output_filename = os.path.join(save_dir, out_name)

    df.to_csv(output_filename, index=False)
    print("output saved to", output_filename)

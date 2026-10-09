"""SGLang runner for low-temperature, standard, and PowerMH sampling.

Run with ``python -m power_sharpening.runners.sglang.run_power_sample_mh --task math500 --algorithm power_mcmc``. The
PowerMH log lines match the vLLM runner (``runners/vllm/run_power_mh.py``) so the case-study parsers read both backends.
"""
import json
import logging
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
from power_sharpening.backends.sglang.kv_cache_stats import SglangKvCacheStats
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

logger = logging.getLogger("[run_powersampling_mh_sglang]")


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
        # Math-style benchmarks carry a scalar "answer"; code benchmarks (LCB, HumanEval, MBPP) grade against
        # the full problem dict (e.g. "test") instead and have no such key.
        answer = problem.get("answer", problem.get("correct_letter", problem.get("task_id")))
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
        # Math-style benchmarks carry a scalar "answer"; code benchmarks (LCB, HumanEval, MBPP) grade against
        # the full problem dict (e.g. "test") instead and have no such key.
        answer = problem.get("answer", problem.get("correct_letter", problem.get("task_id")))
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
    engine_kwargs: dict | None = None,
):
    engine, tokenizer = sglang_load_model_and_tokenizer(model_str, **(engine_kwargs or {}))
    kv_stats = SglangKvCacheStats(engine)
    kv_stats.log_capacity()
    base_sampler = SGL_LLM_Wrapper(
        engine, tokenizer, temperature=init_temperature, alpha=alpha
    )
    logger.info("loaded base_sampler %s", base_sampler)

    logger.info(
        "Benchmark on %s (power-MCMC, %s cuts)...",
        benchmark.name,
        cut_dist_type,
    )
    logger.info(
        "temperature schedule=%s mcmc_steps=%d",
        temperature_schedule_type,
        mcmc_steps,
    )
    results = []
    begin = time.time()
    seen = 0
    total_accuracy = 0
    run_size = min(benchmark.size, maxi_iters)
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

        # mcmc_tokens is the full prefix+completion sequence of token ids; grade and save only the response, as the vLLM
        # runner does.
        completion = tokenizer.decode(mcmc_tokens[len(prefix):], skip_special_tokens=True)
        # Wording held identical to the vLLM runner: the case-study parser closes a sample on this line.
        logger.info(
            "mcmc_power_sampler took %.3f seconds for %d-th prompts, "
            "response lengths: %s",
            time.time() - sampler_started,
            seen,
            [len(completion)],
        )
        result = benchmark.evaluate_completions([problem], [completion])[0]
        logger.info(
            "sample %d response_length=%d prediction=%s "
            "is_correct=%s justification=%s",
            seen,
            len(result["completion"]),
            result["prediction"],
            result["is_correct"],
            result.get("justification"),
        )
        logger.info("correct answer: %s", answer)
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
                **kv_stats.mark(seen),
            }
        )
        total_accuracy += int(result["is_correct"])
        seen += 1
        total_elapsed = time.time() - begin
        avg_per_sample = total_elapsed / seen
        logger.info(
            "[%d/%d] correct=%d/%d acc=%.4f | "
            "batch=%.1fs avg/sample=%.1fs total=%.4fhours eta=%.1fsec",
            seen,
            run_size,
            total_accuracy,
            seen,
            total_accuracy / seen,
            time.time() - st,
            avg_per_sample,
            total_elapsed / 3600,
            avg_per_sample * max(run_size - seen, 0),
        )

    logger.info("resources: %s", kv_stats.summary_line())
    return results, kv_stats.summary()


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
    parser.add_argument(
        "--max_samples", type=int, default=maxi_iters,
        help="Number of benchmark problems to run.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
        help="Generation budget (tokens) per completion.",
    )

    # Register only the selected task's benchmark-specific options.
    add_benchmark_args(parser, task)

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
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
    results = None
    resource_summary = None
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
        results, resource_summary = power_sampling(
            model_str,
            benchmark,
            alpha=args.alpha,
            mcmc_steps=args.mcmc_steps,
            num_blocks=args.num_blocks,
            temperature_schedule_type=args.temperature_schedule_type,
            init_temperature=args.temperature,
            cut_dist_type=args.cut_dist_type,
            engine_kwargs={
                "log_level": args.engine_log_level,
                "decode_log_interval": args.decode_log_interval,
            },
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
    logger.info("output saved to %s", output_filename)

    if resource_summary is not None:
        resource_filename = output_filename.replace(".csv", ".resources.json")
        with open(resource_filename, "w") as handle:
            json.dump(
                {"run": Path(output_filename).stem, "backend": "sglang", **resource_summary},
                handle,
                indent=2,
            )
        logger.info("resource metrics saved to %s", resource_filename)

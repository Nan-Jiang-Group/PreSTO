"""Config-driven vLLM runner for subtree-prefetching Metropolis-Hastings.

Run on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.vllm.run_subtree_prefetching_mh \
      --dataset math500 --algorithm subtree_prefetching_mh

This is the vLLM counterpart of ``runners/hf/run_subtree_prefetching_mh.py``:
same ``dataset.yaml`` / ``algorithms.yaml`` config path, same ``subtree_prefetching_mh`` algorithm block, same
per-sample CSV plus ``.stats.json`` output. The sampler runs one prompt at a time (a subtree is single-prompt), but each
MH step issues a single batched ``vllm.LLM.generate`` over the whole proposal frontier.

Backend-specific behavior:

  * the custom vLLM engine is built once per run and reused for every prompt (``vLLM_Wrapper(engine_type="custom")``);
  * ``cut_dist_type=entropy`` selects the EntropyCut sampler, which reads full-vocabulary entropy from the engine and
    applies the cut-probability correction;
  * those entropy estimates also populate base likelihood/confidence diagnostics.
"""

import os

# Disable tqdm progress bars (vLLM emits them per request, which is noisy in batch logs). Must be set before vLLM is
# imported.
os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import logging
import time
from contextlib import closing
from pathlib import Path

# Load the patched vLLM engine before benchmark/native dependencies. On TACC's aarch64 CUDA nodes, loading it later can
# exhaust glibc's static TLS bookkeeping and abort in _dl_allocate_tls_init before Python can report an exception.
from power_sharpening.backends.vllm.engine_patch import SamplingParams

import numpy as np
import pandas as pd

from power_sharpening.common.prefetch_rank import RANK_FNS
from power_sharpening.common.resource_probe import ResourceProbe
from power_sharpening.common.temp_scheduler import set_schedule
from power_sharpening.config import add_config_selection_args, resolve_config
from power_sharpening.config.model_context import (
    batch_max_new_tokens,
    resolve_max_model_len,
)
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.tasks.registry import (
    build_benchmark,
    set_random_seed,
)

from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper
from power_sharpening.backends.vllm.samplers.subtree_prefetching_MH_sampler import (
    subtree_prefetching_sampling,
    subtree_prefetching_sampling_with_entropy_cut,
)

logger = logging.getLogger("[run_subtree_prefetching_vllm]")


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
    """Format prompts and retain each benchmark's grading-ready problems."""
    prompts, problems = benchmark.get_question_and_answer(batch, tokenizer)
    return prompts, problems[: len(prompts)]


def _question_text(problem: dict, formatted_prompt: str) -> str:
    """Return the dataset's unformatted question text for logs and CSV output."""
    return (
        problem.get("prompt")
        or problem.get("question")
        or problem.get("text")
        or formatted_prompt
    )


def _block_aligned_budget(max_new_tokens: int, num_blocks: int) -> int:
    """Round a generation budget down to a whole number of blocks.

    The sampler asserts ``max_new_tokens % num_of_blocks == 0``, so a budget trimmed to fit the context window has to be
    re-aligned before use.
    """
    aligned = (max_new_tokens // num_blocks) * num_blocks
    if aligned <= 0:
        raise ValueError(
            f"generation budget {max_new_tokens} is smaller than one block "
            f"of a {num_blocks}-block run; lower num_blocks or shorten the prompt"
        )
    return aligned


def run_subtree_prefetching(
    model_str,
    benchmark,
    args,
    max_model_len: int | None,
    init_temperature: float,
):
    """Run subtree-prefetching MH over a shared benchmark implementation."""
    evict_subtree_cache = getattr(args, "evict_subtree_cache", False)
    if evict_subtree_cache and not args.prefix_cache:
        raise ValueError("--evict_subtree_cache requires --override prefix_cache=true")
    if args.cut_dist_type == "entropy":
        sampling_fn = subtree_prefetching_sampling_with_entropy_cut
        cut_options = {"cut_power": getattr(args, "cut_dist_param", None)}
    elif args.cut_dist_type == "uniform":
        sampling_fn = subtree_prefetching_sampling
        cut_options = {}
    else:
        raise ValueError(f"unknown cut_dist_type {args.cut_dist_type!r}; expected 'uniform' or 'entropy'")

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
        vllm_kwargs.setdefault("disable_log_stats", False)

    # One generator drives every cut draw, seeded so a run is reproducible (numpy's default_rng ignores the global seed
    # set by set_random_seed).
    rng = np.random.default_rng(args.seed)

    results = []
    all_stats = []
    begin = time.time()
    seen = 0

    # Start sampling before the engine allocates anything, so the baseline separates our footprint from whatever else
    # shares the device.
    probe = ResourceProbe(enabled=getattr(args, "resource_probe", True))
    probe.start()

    # Release the engine and probe hooks even if construction or sampling fails.
    with closing(probe), vLLM_Wrapper(
        model=model_str,
        engine_type="custom",
        seed=args.seed,
        enable_prefix_caching=args.prefix_cache,
        enable_subtree_cache_eviction=evict_subtree_cache,
        dtype=getattr(args, "dtype", "auto"),
        verbose=args.verbose,
        **vllm_kwargs,
    ) as mh_llm:
        probe.attach(mh_llm.llm)
        tokenizer = mh_llm.tokenizer
        sampling_params = SamplingParams(
            temperature=init_temperature,
            alpha=args.alpha,
            seed=args.seed,
        )

        # The subtree sampler steps the schedule once per block, so the schedule spans num_blocks steps. Unset (the
        # algorithms.yaml default) means the proposal temperature stays at init_temperature, matching the HF runner.
        schedule_type = getattr(args, "temperature_schedule_type", None)
        scheduler = (
            set_schedule(init_temperature, schedule_type, total_steps=args.num_blocks)
            if schedule_type
            else None
        )
        logger.info(
            "Benchmark on %s (subtree prefetching MH, vLLM), temperature=%.6f, "
            "schedule=%s",
            benchmark.name,
            init_temperature,
            schedule_type or "none",
        )

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

                question = _question_text(problem, input_text)

                # Math-style benchmarks carry a scalar "answer"; code benchmarks (LCB, HumanEval, MBPP) grade against
                # the full problem dict (e.g. "test") instead and have no such key.
                answer = problem.get(
                    "answer",
                    problem.get("correct_letter", problem.get("task_id")),
                )
                print("IDX", seen, "\t QUESTION:", question.split("\n")[0], flush=True)
                prompt_ids = tokenizer.encode(input_text)

                # Cap the budget so prompt + output fits the context window, then re-align it to a whole number of
                # blocks.
                max_new_tokens = _block_aligned_budget(
                    batch_max_new_tokens(
                        tokenizer,
                        [input_text],
                        max_model_len,
                        args.max_new_tokens,
                    ),
                    args.num_blocks,
                )

                # A fresh schedule per prompt keeps sample i independent of the temperature the previous sample happened
                # to end on.
                if scheduler is not None:
                    scheduler.step_count = 0

                sampler_started = time.time()
                response_ids, stats = sampling_fn(
                    mh_llm,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    mcmc_steps=args.mcmc_steps,
                    max_batch_size=args.prefetch_budget,
                    num_of_blocks=args.num_blocks,
                    max_new_tokens=max_new_tokens,
                    rank_fn=args.rank_fn,
                    stop_on_eos=getattr(args, "stop_on_eos", True),
                    print_tree=args.print_tree,
                    temperature_scheduler=scheduler,
                    rng=rng,
                    verbose=args.verbose,
                    evict_subtree_cache=evict_subtree_cache,
                    **cut_options,
                )
                logger.info(
                    "subtree_prefetching_sampling took %.3f seconds "
                    "for %d-th prompts, response tokens: %d",
                    time.time() - sampler_started,
                    seen,
                    len(response_ids),
                )
                # Close the resource window on this sample before decoding and grading, which touch no GPU.
                resources = probe.mark()
                # The vLLM sampler already excludes the prompt.
                completion = tokenizer.decode(response_ids, skip_special_tokens=True)
                print(">>> completion:\n" + completion, flush=True)
                print("=" * 30)
                result = benchmark.evaluate_completions([problem], [completion])[0]

                # EntropyCut supplies base-model likelihood and confidence diagnostics.
                diagnostics = stats.base_diagnostics
                print(
                    f"response length: {len(completion)}, "
                    f"response tokens: {len(response_ids)}",
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
                stats_json["evict_subtree_cache"] = evict_subtree_cache
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
                        "alpha": args.alpha,
                        "mcmc_steps": args.mcmc_steps,
                        "prefetch_budget": args.prefetch_budget,
                        "evict_subtree_cache": evict_subtree_cache,
                        "rank_fn": args.rank_fn,
                        "cut_dist_type": args.cut_dist_type,
                        "cut_dist_param": getattr(args, "cut_dist_param", None),
                        "num_blocks": args.num_blocks,
                        "max_new_tokens": max_new_tokens,
                        "init_temperature": init_temperature,
                        "temperature_schedule_type": schedule_type,
                        "max_model_len": max_model_len,
                        "dtype": getattr(args, "dtype", "auto"),
                        "num_response_tokens": (
                            None if diagnostics is None else diagnostics.num_tokens
                        ),
                        "log_likelihood": (
                            None if diagnostics is None else diagnostics.log_likelihood
                        ),
                        "confidence": (
                            None if diagnostics is None else diagnostics.confidence
                        ),
                        "acceptance_ratio": stats_json.get("acceptance_rate", 0.0),
                        "total_workload": stats.total_workload,
                        "total_nfe": stats.total_nfe,
                        "total_acceptances": stats.total_acceptances,
                        **resources,
                    }
                )

                seen += 1
                now = time.time()
                print(
                    f"used time: each {now - st:.3f} sec, total {now - begin:.3f} sec",
                    flush=True,
                )
                print("-" * 80)

    # Run-level peaks remain available after engine teardown and probe cleanup.
    resource_summary = probe.summary()
    logger.info("resources: %s", probe.summary_line())

    return results, all_stats, resource_summary


REQUIRED_SUBTREE_PREFETCHING_KEYS = (
    "task",
    "batch_size",
    "seed",
    "alpha",
    "temperature",
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
            "Config-driven vLLM subtree-prefetching MH runner. Selects a dataset "
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


    probe_group = parser.add_argument_group("resource measurement")
    probe_group.add_argument(
        "--evict_subtree_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each MH subtree, evict idle cached prefixes exclusive to discarded "
            "sequences and reuse those blocks first. Requires prefix_cache=true and "
            "vLLM 0.27.1. Default off; --no-evict_subtree_cache disables cleanup. "
            "--override evict_subtree_cache=true/false takes precedence."
        ),
    )
    probe_group.add_argument(
        "--resource_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure the GPU-memory peak inside llm.generate (including its "
            "increment above the loaded-model baseline), peak KV-cache "
            "occupancy, physical-block eviction rate, and prefix-cache hit "
            "rate. Results are written to <run>.resources.json and per-sample "
            "columns. Turning this on also enables vLLM engine statistics. Set "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 for exact occupancy and eviction "
            "metrics; otherwise occupancy is polled and eviction is unavailable. "
            "Pass --no-resource_probe to run without any of it. "
            "(--override resource_probe=false wins over this flag.)"
        ),
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

    # Decide the context window: some benchmarks (e.g. long-context code) need an explicit cap derived from the
    # checkpoint; others leave it to vLLM.
    if not benchmark.uses_max_model_len:
        max_model_len = None
    else:
        max_model_len, checkpoint_limit = resolve_max_model_len(
            model_str,
            requested=getattr(args, "max_model_len", None),
        )
        logger.info(
            "checkpoint context limit=%s; using max_model_len=%s",
            checkpoint_limit,
            max_model_len,
        )

    # Convention: --override temperature=-1 means "use 1/alpha" (the natural
    # proposal temperature for sampling from p(x)^alpha).
    init_temperature = (
        1.0 / args.alpha if args.temperature == -1.0 else args.temperature
    )

    _log_run_configuration(
        args,
        benchmark_name=benchmark.name,
        benchmark_size=benchmark.size,
        model_str=model_str,
        task=task,
        max_model_len=max_model_len,
        init_temperature=init_temperature,
    )

    results, all_stats, resource_summary = run_subtree_prefetching(
        model_str=model_str,
        benchmark=benchmark,
        args=args,
        max_model_len=max_model_len,
        init_temperature=init_temperature,
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
        f"_vllm"
        f"_mcmc{args.mcmc_steps}_batch{args.prefetch_budget}"
        f"_blocks{args.num_blocks}_maxnew{args.max_new_tokens}"
        f"_rank-{args.rank_fn}"
        f"{cut_tag}"
        f"_temp{args.temperature}_alpha{args.alpha}_seed{args.seed}"
        f"{'_cache-evict' if args.evict_subtree_cache else ''}"
    )
    output_filename = os.path.join(args.save_str, f"{out_stem}.csv")

    df.to_csv(output_filename, index=False)
    logger.info("output saved to %s", output_filename)

    stats_filename = output_filename.replace(".csv", ".stats.json")
    with open(stats_filename, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("stats saved to %s", stats_filename)

    # Run-level resource metrics: one JSON per run, so a sweep can be collected by globbing the dump directory the same
    # way the other case studies do.
    resource_filename = output_filename.replace(".csv", ".resources.json")
    with open(resource_filename, "w") as f:
        json.dump(
            {
                "run": out_stem,
                "rank_fn": args.rank_fn,
                "prefetch_budget": args.prefetch_budget,
                "num_blocks": args.num_blocks,
                "mcmc_steps": args.mcmc_steps,
                "prefix_cache": args.prefix_cache,
                "evict_subtree_cache": args.evict_subtree_cache,
                "dtype": getattr(args, "dtype", "auto"),
                **resource_summary,
            },
            f,
            indent=2,
        )
    logger.info("resource metrics saved to %s", resource_filename)

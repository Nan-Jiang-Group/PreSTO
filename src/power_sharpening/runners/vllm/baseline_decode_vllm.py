"""Standard vLLM decode baseline no power sampling.

Reuses the same TaskAdapter (prompt formatting + parsing + grading) as ``repps_exact_math_vllm.py`` so the resulting
accuracy is directly comparable to the REPPS runs. One vLLM ``generate`` call per problem with ``n=1`` (greedy by
default).

"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
import gc
import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

from constants import MODEL_MAP
from repps_vllm.tasks import get_task
from repps_vllm.tokenizer_fix import safe_tokenizer_path
from result_paths import prepare_results_dir


logger = logging.getLogger("[baseline_decode_vllm]")
logging.basicConfig(
    format="%(name)s %(levelname)s: %(message)s", level=logging.INFO,
)

# Repo root is two parents up from this module; datasets live under
# <repo_root>/data — used as the default --dataset_folder.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plain vLLM decode baseline (n=1) over a task adapter."
    )
    p.add_argument("--save_str", type=str, default="results/")
    p.add_argument("--model_str", type=str, required=True,
                   choices=list(MODEL_MAP.keys()))
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset_folder", type=Path,
                   default=_DEFAULT_DATA_DIR,
                   help="Directory holding dataset files "
                        "(default: <repo_root>/data).")
    p.add_argument("--cot", type=bool, default=True)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy. >0 enables stochastic sampling.")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=3072)
    p.add_argument("--seed", type=int, default=10086)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--enable_prefix_caching",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_chat_template", action="store_true")
    args = p.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)

    task = get_task(args.dataset)
    dataset = task.load(args.dataset_folder, max_examples=args.max_examples)

    model_str = MODEL_MAP[args.model_str]
    # Some checkpoints (e.g. deepseek-math-*) ship a ByteLevel tokenizer mislabelled as a Llama tokenizer, which makes
    # AutoTokenizer strip whitespace and zeroes out code tasks. Resolve a safe tokenizer source (a no-op for healthy
    # tokenizers) and hand it to vLLM so both prompt encoding and detokenization preserve whitespace.
    tokenizer_str = safe_tokenizer_path(model_str, args.trust_remote_code)
    llm = LLM(
        model=model_str,
        tokenizer=tokenizer_str,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )

    tokenizer = llm.get_tokenizer()
    prompts, metas = [], []
    for data in dataset:
        text, meta = task.format_prompt(
            item=data, model=args.model_str, tokenizer=tokenizer,
            cot=args.cot, use_chat_template=args.use_chat_template, rng=random.Random(args.seed),
        )
        prompts.append(text)
        metas.append((data, meta))
    

    sp = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_new_tokens, seed=args.seed,
        stop=task.stop_tokens(),
    )
    t0 = time.monotonic()
    outs = llm.generate(prompts, sp)
    dt = time.monotonic() - t0
    logger.info(f"llm generate reponse done, time taken: {dt}")
    # Parse each completion, then run the task's optional two-stage extraction for any answer it considers unreliable.
    # This mirrors the REPPS runner: a base model that ignores the boxed-letter instruction (e.g. GPQA) gets its final
    # answer pinned by a tiny greedy continuation instead of being scored as a NaN/zero. No-op for tasks that don't
    # override ``needs_extraction`` (the default returns False).
    try:
        max_model_len = llm.llm_engine.model_config.max_model_len
    except Exception:
        max_model_len = 4096
    # leave room for the greedy continuation + a few boundary/special tokens
    ext_budget = max_model_len - task.extract_max_tokens() - 16

    comps = [o.outputs[0].text for o in outs]
    parsed_list = [task.parse_completion(comps[i], metas[i][1]) for i in range(len(outs))]
    pending = []  # (row_idx, extraction_prompt)
    for i, parsed in enumerate(parsed_list):
        if task.needs_extraction(parsed):
            cue = task.extract_answer_text(comps[i], metas[i][1])
            if not cue:
                continue
            ext_prompt = prompts[i] + (comps[i] or "") + cue
            # (prompt + completion + cue) can exceed the model context for long completions. Keep the question head and
            # the conclusion tail (which holds the cue) and drop the middle of the reasoning, which extraction does not
            # need.
            ids = tokenizer.encode(ext_prompt)
            if len(ids) > ext_budget:
                head = ext_budget // 2
                ids = ids[:head] + ids[-(ext_budget - head):]
                ext_prompt = tokenizer.decode(ids, skip_special_tokens=True)
            pending.append((i, ext_prompt))
    if pending:
        ext_sp = SamplingParams(
            n=1, temperature=0.0,
            max_tokens=task.extract_max_tokens(),
            stop=task.extract_stop_tokens(), seed=args.seed,
        )
        # Best-effort: a failure in the extraction pass must never abort the run.
        try:
            ext_outs = llm.generate([p for _, p in pending], ext_sp)
            n_rec = 0
            for (i, _), eo in zip(pending, ext_outs):
                ext_parsed = task.parse_extraction(eo.outputs[0].text, metas[i][1])
                if ext_parsed:
                    parsed_list[i] = ext_parsed
                    n_rec += 1
            logger.info(
                "two-stage extraction ran on %d/%d rows, recovered %d answers",
                len(pending), len(outs), n_rec,
            )
        except Exception as e:
            logger.warning("two-stage extraction skipped (error): %s", e)

    rows = []
    for i, out in enumerate(outs):
        data, meta = metas[i]
        comp = comps[i]
        n_tok = len(out.outputs[0].token_ids)
        parsed = parsed_list[i]
        is_correct = task.grade(parsed, data, meta)
        rows.append({
            "problem_idx": i,
            "problem_id": data.get("id", i),
            "question": task.question_for_csv(data, meta),
            "correct_answer": task.gold_for_csv(data, meta),
            "baseline_completion": comp,
            "baseline_answer": parsed,
            "baseline_is_correct": is_correct,
            "baseline_n_tokens": n_tok,
        })
    logger.info(f"reponses are parsed: {len(rows)}, example of rows: {rows[0]}")

    ## Save as csv to output folder
    run_dir, save_str = prepare_results_dir(
        args.save_str, args.model_str, f"{task.name}_baseline_vllm",
        [args.temperature], resume=False,
    )
    fname = (
        f"{args.model_str}_{task.name}_baseline_T{args.temperature}_{args.seed}.csv"
    )
    out_path = os.path.join(save_str, fname)
    logger.info("Saving results to %s", out_path)

    df = pd.DataFrame(rows)
    # The run dir is created at startup, but on shared storage it can be removed out from under a long job before the
    # write lands; re-create it here so the write can't fail on a missing directory.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    n_correct = int(df["baseline_is_correct"].sum())
    n_total = len(df)
    logger.info("=" * 60)
    logger.info("BASELINE (vLLM) SUMMARY")
    logger.info("Model:       %s", model_str)
    logger.info("Dataset:     %s   (n=%d)", task.name, n_total)
    logger.info("Decoding:    n=1, T=%.2f, top_p=%.2f, max_new=%d",
                args.temperature, args.top_p, args.max_new_tokens)
    logger.info("Accuracy:    %d/%d = %.1f%%",
                n_correct, n_total, 100.0 * n_correct / max(n_total, 1))
    logger.info("Avg tokens:  %.1f", df["baseline_n_tokens"].mean())
    logger.info("Wall:        %.1fs total (%.2fs / problem)",
                dt, dt / max(n_total, 1))
    logger.info("Saved to:    %s", out_path)
    logger.info("=" * 60)

    # after logger.info("=" * 60), before script exits
    try:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)
    except Exception as e:
        logger.warning("vLLM cleanup warning: %s", e)
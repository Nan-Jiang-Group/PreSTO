import os
import json
import random
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

from vllm import LLM

from grader_utils.parse_utils import parse_answer
from grader_utils.math_grader import grade_answer
from constants import *
from power_samp_utils import format_prompt
from topk_constrained_beam import topk_constrained_beam_search


def safe_grade_math(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action="store", type=str, default="results/", dest="save_str")
    parser.add_argument("--model", action="store", default="qwen", type=str,
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--topk", action="store", type=int, default=40, help="Per‑step top‑k next-token logprobs to consider per beam")
    parser.add_argument("--k", action="store", type=int, default=10, help="Number of sequences (beam width)")
    parser.add_argument("--max_problems", action="store", type=int, default=None, help="Limit number of problems for debugging")
    parser.add_argument("--debug", action="store_true", help="Enable verbose progress for k‑CBS inner loop")
    parser.add_argument("--quiet_vllm", action="store_true", help="Reduce vLLM logging/progress output")
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--length_penalty", action="store", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", action="store", type=int, default=3072)
    # No explicit cap; pass topk to vLLM and allow it to error if unsupported
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = args.model
    dataset_name = args.dataset
    cot = args.cot
    topk = args.topk
    k = args.k
    length_penalty = args.length_penalty
    max_new_tokens = args.max_new_tokens

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    # Model mapping
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = 'microsoft/Phi-3.5-mini-instruct'
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Load dataset
    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        dataset = json.load(open(json_file, "r"))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("Dataset loaded")
    if args.max_problems is not None and args.max_problems > 0:
        dataset = dataset[:args.max_problems]
        print(f"Debug: limiting to first {len(dataset)} problems")

    # Quiet vLLM logs/progress if requested
    if args.quiet_vllm:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        os.environ.setdefault("VLLM_DISABLE_LOG_STATS", "1")
        try:
            import logging
            logging.getLogger("vllm").setLevel(logging.WARNING)
        except Exception:
            pass

    # Initialize vLLM (set conservative GPU utilization)
    llm = LLM(model=model_str, trust_remote_code=True, gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()

    print("vLLM model loaded")

    results = []

    print(f"Running Top‑k‑Constrained Beam Search (k‑CBS)")
    print(f"Requested beam_width (k sequences): {k}; topk (per-step logprobs): {topk}; length_penalty: {length_penalty}; max_new_tokens: {max_new_tokens}")
    if topk < k:
        print(f"Warning: topk ({topk}) < beam_width k ({k}). Early steps may have <k beams available.")

    for i, data in enumerate(tqdm(dataset, desc="k‑CBS problems")):
        question = data["prompt"]
        correct_answer = data["answer"]

        base_prompt = format_prompt(question, model, tokenizer, cot)

        search_out = topk_constrained_beam_search(
            llm=llm,
            tokenizer=tokenizer,
            base_prompt=base_prompt,
            beam_width=k,
            topk=topk,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_token_id,
            length_penalty=length_penalty,
            return_k2_preprune=False,
            debug=args.debug,
            suppress_lib_stdout=args.quiet_vllm,
        )

        beams = search_out["beams"]
        topk_eff = search_out.get("topk_eff", topk)

        # Decode continuations (generated tokens only)
        # Use accumulated text (more robust across vLLM logprobs formats)
        completions = [b.text for b in beams]
        answers = [parse_answer(c) for c in completions]
        correct_flags = [safe_grade_math(a, correct_answer) for a in answers]
        pass_at_k = int(max(correct_flags)) if len(correct_flags) > 0 else 0

        problem_result = {
            "question": question,
            "correct_answer": correct_answer,
            "completions": completions,
            "answers": answers,
            "correct_flags": correct_flags,
            "pass_at_k": pass_at_k,
            "k": k,  # report requested beam width for consistent pass@k curves
            "k_requested": k,
            "topk_eff": topk_eff,
            "topk_requested": topk,
        }

        results.append(problem_result)

    total_pass_at_k = sum(r["pass_at_k"] for r in results)
    pass_at_k_accuracy = total_pass_at_k / len(results)

    print(f"\nOverall Pass@k (k‑CBS): {pass_at_k_accuracy:.4f} ({total_pass_at_k}/{len(results)})")

    # Save results
    df = pd.DataFrame(results)
    # File naming mirrors other scripts; include effective k
    output_file = os.path.join(save_str, f"{model}_vllm_kcbs_bw_{k}_topk_{topk}_seed_{args.seed}.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    summary = {
        "model": model,
        "method": "kcbs",
        "beam_width": k,
        "k_requested": k,
        "topk_eff": topk,
        "topk_requested": topk,
        "length_penalty": length_penalty,
        "max_new_tokens": max_new_tokens,
        "dataset": dataset_name,
        "total_problems": len(results),
        "pass_at_k_accuracy": pass_at_k_accuracy,
        "pass_at_k_count": total_pass_at_k,
    }
    summary_file = os.path.join(save_str, f"{model}_vllm_kcbs_bw_{k}_topk_{topk}_summary_seed_{args.seed}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

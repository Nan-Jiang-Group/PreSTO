import argparse
from tqdm import tqdm
import pandas as pd


from vllm import LLM, SamplingParams

from grader_utils.parse_utils import parse_answer
from grader_utils.math_grader import grade_answer
import numpy as np
import os
import json
import random
from power_samp_utils import format_prompt


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
    parser.add_argument("--temperature", action="store", default=0.8, type=float, dest="temperature")
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--k", action="store", type=int, default=16, help="Number of samples per problem")
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--max_problems", action="store", type=int, default=None, help="Limit number of problems for debugging")
    parser.add_argument("--quiet_vllm", action="store_true", help="Reduce vLLM logging/progress output")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = args.model
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    k = args.k

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    print(f"Model: {model}")
    print(f"Temperature: {temp}")
    print(f"K samples: {k}")

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

    # Load dataset
    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        dataset = json.load(open(json_file, "r"))

    print("Dataset loaded")
    if args.max_problems is not None and args.max_problems > 0:
        dataset = dataset[:args.max_problems]
        print(f"Debug: limiting to first {len(dataset)} problems")

    if args.quiet_vllm:
        import os as _os
        _os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        _os.environ.setdefault("VLLM_DISABLE_LOG_STATS", "1")
        try:
            import logging
            logging.getLogger("vllm").setLevel(logging.WARNING)
        except Exception:
            pass

    # Initialize vLLM
    llm = LLM(model=model_str, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temp,
        max_tokens=3072,
        n=k,  # Generate k samples per prompt
    )

    print("vLLM model loaded")

    # Prepare all prompts at once - vLLM handles batching efficiently
    all_prompts = []
    all_metadata = []
    
    for i, data in enumerate(dataset):
        question = data["prompt"]
        answer = data["answer"]
        
        input_text = format_prompt(question, model, tokenizer, cot)
        all_prompts.append(input_text)
        all_metadata.append({
            "problem_idx": i,
            "question": question,
            "correct_answer": answer
        })
    
    print(f"Generating {k} samples for {len(all_prompts)} problems...")
    
    # Generate responses for all prompts at once
    if args.quiet_vllm:
        import io as _io
        from contextlib import redirect_stdout as _rs, redirect_stderr as _re
        _buf_out, _buf_err = _io.StringIO(), _io.StringIO()
        with _rs(_buf_out), _re(_buf_err):
            outputs = llm.generate(all_prompts, sampling_params)
    else:
        outputs = llm.generate(all_prompts, sampling_params)
    
    # Process outputs
    results = []
    for output, metadata in tqdm(zip(outputs, all_metadata), total=len(outputs), desc="Processing results"):
        question = metadata["question"]
        correct_answer = metadata["correct_answer"]
        
        # Extract all k completions for this problem
        completions = [o.text for o in output.outputs]
        answers = [parse_answer(completion) for completion in completions]
        
        # Compute pass@k results
        correct_flags = [safe_grade_math(ans, correct_answer) for ans in answers]
        pass_at_k_result = max(correct_flags)  # 1 if any answer is correct, 0 otherwise
        
        # Store individual results
        problem_result = {
            "question": question,
            "correct_answer": correct_answer,
            "completions": completions,
            "answers": answers,
            "correct_flags": correct_flags,
            "pass_at_k": pass_at_k_result,
            "k": k
        }
        
        results.append(problem_result)

    # Compute overall pass@k accuracy
    total_pass_at_k = sum(result["pass_at_k"] for result in results)
    pass_at_k_accuracy = total_pass_at_k / len(results)
    
    print(f"\nOverall Pass@{k} Accuracy: {pass_at_k_accuracy:.4f} ({total_pass_at_k}/{len(results)})")

    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(save_str, f"{model}_vllm_passk_{k}_temp_{temp}_seed_{args.seed}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")

    # Also save summary
    summary = {
        "model": model,
        "temperature": temp,
        "k": k,
        "total_problems": len(results),
        "pass_at_k_accuracy": pass_at_k_accuracy,
        "pass_at_k_count": total_pass_at_k
    }
    
    summary_file = os.path.join(save_str, f"{model}_vllm_passk_{k}_summary_temp_{temp}_seed_{args.seed}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

import os
import json
import random
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

from vllm import LLM
from vllm.sampling_params import BeamSearchParams

from grader_utils.parse_utils import parse_answer
from grader_utils.math_grader import grade_answer

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
    parser.add_argument("--temperature", action="store", default=1.0, type=float, dest="temperature")
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--k", action="store", type=int, default=16, help="Number of beams")
    parser.add_argument("--seed", action="store", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = args.model
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    k = args.k

    # Limit beam width to avoid logprobs issues
    actual_beam_width = min(k, 10)

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    print(f"Model: {model}")
    print(f"Beam width: {actual_beam_width} (requested: {k})")
    print(f"Using beam search with temperature 0.0")

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

    # Initialize vLLM with memory constraints for beam search
    llm = LLM(model=model_str, trust_remote_code=True, gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()
    
    # Beam search parameters
    beam_params = BeamSearchParams(
        beam_width=actual_beam_width,
        max_tokens=3072,
        temperature=0.0,
        length_penalty=1.0,
    )

    print("vLLM model loaded")

    # Prepare all prompts - beam search expects list of dicts with 'prompt' key
    all_prompts = []
    all_metadata = []
    
    for i, data in enumerate(dataset):
        question = data["prompt"]
        answer = data["answer"]
        
        input_text = format_prompt(question, model, tokenizer, cot)
        all_prompts.append({"prompt": input_text})
        all_metadata.append({
            "problem_idx": i,
            "question": question,
            "correct_answer": answer
        })
    
    # Process in smaller batches to avoid memory issues
    batch_size = 1  # Process 1 problem at a time to avoid CUDA memory errors
    all_outputs = []
    
    print(f"Generating {actual_beam_width} beams for {len(all_prompts)} problems using beam search...")
    print(f"Processing in batches of {batch_size} for progress tracking...")
    
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Beam search batches"):
        batch_prompts = all_prompts[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_prompts) + batch_size - 1)//batch_size} ({len(batch_prompts)} problems)")
        
        # Generate responses for this batch
        batch_outputs = llm.beam_search(batch_prompts, beam_params)
        all_outputs.extend(batch_outputs)
        
        print(f"Completed batch {i//batch_size + 1}, total processed: {len(all_outputs)}")
    
    outputs = all_outputs
    
    # Process outputs
    results = []
    for output, metadata in tqdm(zip(outputs, all_metadata), total=len(outputs), desc="Processing results"):
        question = metadata["question"]
        correct_answer = metadata["correct_answer"]
        
        # Extract all k beam sequences for this problem
        completions = [seq.text for seq in output.sequences]
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
            "k": actual_beam_width
        }
        
        results.append(problem_result)

    # Compute overall pass@k accuracy
    total_pass_at_k = sum(result["pass_at_k"] for result in results)
    pass_at_k_accuracy = total_pass_at_k / len(results)
    
    print(f"\nOverall Pass@{actual_beam_width} Accuracy (Beam Search): {pass_at_k_accuracy:.4f} ({total_pass_at_k}/{len(results)})")

    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(save_str, f"{model}_vllm_beam_{actual_beam_width}_temp_{temp}_seed_{args.seed}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")

    # Also save summary
    summary = {
        "model": model,
        "method": "beam_search",
        "beam_width": actual_beam_width,
        "temperature": 0.0,
        "length_penalty": 1.0,
        "total_problems": len(results),
        "pass_at_k_accuracy": pass_at_k_accuracy,
        "pass_at_k_count": total_pass_at_k
    }
    
    summary_file = os.path.join(save_str, f"{model}_vllm_beam_{actual_beam_width}_temp_{temp}_summary_seed_{args.seed}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
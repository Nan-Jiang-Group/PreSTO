#!/usr/bin/env python3
"""
Beam search with power sampling on MATH benchmark.
Compares: standard sampling, naive_temp, beam_search_power_samp
"""

import os
import argparse
import json
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

from power_samp_utils import (
    load_model_and_tokenizer,
    format_prompt,
    add_model_argument,
    naive_temp
)
from beam_search_utils import beam_search_power_samp, beam_search_greedy
from grader_utils.parse_utils import parse_answer
from constants import *


def main():
    parser = argparse.ArgumentParser(description="Beam search power sampling on MATH")
    parser.add_argument("--save_str", type=str, default="results/")
    add_model_argument(parser, default="qwen")
    parser.add_argument("--temperature", type=float, default=0.25, 
                       help="Temperature for p^α (α=1/temp)")
    parser.add_argument("--beam_width", type=int, default=5,
                       help="Number of parallel beams")
    parser.add_argument("--tokens_per_step", type=int, default=16,
                       help="Generate this many tokens per expansion")
    parser.add_argument("--length_penalty", type=float, default=0.6,
                       help="Length normalization (0.6-1.0)")
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--cot", type=bool, default=True)
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0,
                       help="Batch index for parallel processing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare_mcmc", action="store_true",
                       help="Also compare with MCMC power sampling")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed beam search info")
    
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup
    model = args.model
    device = args.device
    temp = args.temperature
    beam_width = args.beam_width
    tokens_per_step = args.tokens_per_step
    
    save_str = os.path.join(args.save_str, model, "beam_search")
    os.makedirs(save_str, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("BEAM SEARCH POWER SAMPLING ON MATH BENCHMARK")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Temperature: {temp} (α = {1/temp:.2f})")
    print(f"Beam width: {beam_width}")
    print(f"Tokens per step: {tokens_per_step}")
    print(f"Length penalty: {args.length_penalty}")
    print(f"Batch index: {args.batch_idx}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Load dataset
    if args.dataset == "MATH":
        json_file = 'llm_experiments/data/MATH500.json'
        with open(json_file, "r") as f:
            dataset = json.load(f)
    
    print(f"Loading model: {model}")
    hf_model, tokenizer, autoreg_sampler = load_model_and_tokenizer(model, device)
    print("Model loaded successfully\n")
    
    results = []
    
    # Batch processing
    start = 100 * args.batch_idx
    end = 100 * (args.batch_idx + 1)
    
    dataset_slice = dataset[start:min(end, len(dataset))]
    
    for problem_idx, data in enumerate(tqdm(dataset_slice, desc="Processing MATH problems")):
        absolute_idx = problem_idx + start
        question = data["prompt"]
        correct_answer = data["answer"]
        
        print(f"\n{'='*80}")
        print(f"Problem {absolute_idx}")
        print(f"{'='*80}")
        print(f"Question: {question[:150]}...")
        print(f"Correct answer: {correct_answer}")
        
        # Format prompt
        input_text = format_prompt(question, model, tokenizer, args.cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefix = [idx.item() for idx in input_ids[0]]
        
        # ===== 1. Standard Sampling (baseline) =====
        print(f"\n[1/3] Standard sampling...")
        try:
            std_output = hf_model.generate(
                input_ids, 
                max_new_tokens=3072,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            std_generated_ids = std_output.sequences[0][len(input_ids[0]):].to("cpu")
            std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
            std_answer = parse_answer(std_completion)
            print(f"  Answer: {std_answer}")
        except Exception as e:
            print(f"  Error: {e}")
            std_completion = ""
            std_answer = None
        
        # ===== 2. Naive Temperature Sampling =====
        print(f"\n[2/3] Naive temp sampling (temp={temp})...")
        try:
            naive_output, _, _ = naive_temp(
                autoreg_sampler, 
                prefix, 
                temp=temp, 
                seq_len=len(prefix) + 3072
            )
            naive_generated_ids = naive_output[len(prefix):]
            naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
            naive_answer = parse_answer(naive_completion)
            print(f"  Answer: {naive_answer}")
        except Exception as e:
            print(f"  Error: {e}")
            naive_completion = ""
            naive_answer = None
        
        # ===== 3. Beam Search Power Sampling =====
        print(f"\n[3/3] Beam search power sampling (width={beam_width}, tokens_per_step={tokens_per_step})...")
        try:
            beam_output, lp_norm, lp_unnorm, metadata = beam_search_power_samp(
                autoreg_sampler,
                prefix,
                temp=temp,
                beam_width=beam_width,
                tokens_per_step=tokens_per_step,
                max_new_tokens=3072,
                length_penalty=args.length_penalty,
                verbose=args.verbose
            )
            
            beam_generated_ids = beam_output[len(prefix):]
            beam_completion = tokenizer.decode(beam_generated_ids, skip_special_tokens=True)
            beam_answer = parse_answer(beam_completion)
            print(f"  Answer: {beam_answer}")
            print(f"  Expansions: {metadata['num_expansions']}")
            print(f"  Final score: {metadata.get('final_score', 0.0):.4f}")
            print(f"  Completed beams: {metadata['num_completed_beams']}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            beam_completion = ""
            beam_answer = None
            metadata = {'num_expansions': 0, 'final_score': 0.0, 'num_completed_beams': 0}
        
        # ===== Optional: MCMC Comparison =====
        mcmc_completion = None
        mcmc_answer = None
        mcmc_acceptance = None
        
        if args.compare_mcmc:
            print(f"\n[BONUS] MCMC power sampling for comparison...")
            try:
                from power_samp_utils import mcmc_power_samp
                mcmc_output, _, _, acceptance_ratio = mcmc_power_samp(
                    autoreg_sampler,
                    prefix,
                    temp=temp,
                    mcmc_steps=10,
                    max_new_tokens=3072
                )
                mcmc_generated_ids = mcmc_output[len(prefix):]
                mcmc_completion = tokenizer.decode(mcmc_generated_ids, skip_special_tokens=True)
                mcmc_answer = parse_answer(mcmc_completion)
                mcmc_acceptance = acceptance_ratio
                print(f"  Answer: {mcmc_answer}")
                print(f"  Acceptance ratio: {acceptance_ratio:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
        
        # ===== Store Results =====
        result = {
            "problem_idx": absolute_idx,
            "question": question,
            "correct_answer": correct_answer,
            "std_completion": std_completion,
            "std_answer": std_answer,
            "std_correct": (std_answer == correct_answer) if std_answer is not None else False,
            "naive_completion": naive_completion,
            "naive_answer": naive_answer,
            "naive_correct": (naive_answer == correct_answer) if naive_answer is not None else False,
            "beam_completion": beam_completion,
            "beam_answer": beam_answer,
            "beam_correct": (beam_answer == correct_answer) if beam_answer is not None else False,
            "beam_num_expansions": metadata["num_expansions"],
            "beam_final_score": metadata.get("final_score", 0.0),
            "beam_num_completed": metadata["num_completed_beams"],
        }
        
        if args.compare_mcmc:
            result.update({
                "mcmc_completion": mcmc_completion,
                "mcmc_answer": mcmc_answer,
                "mcmc_correct": (mcmc_answer == correct_answer) if mcmc_answer is not None else False,
                "mcmc_acceptance_ratio": mcmc_acceptance,
            })
        
        results.append(result)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary for problem {absolute_idx}:")
        print(f"  Correct: {correct_answer}")
        print(f"  Standard: {std_answer} {'✓' if result['std_correct'] else '✗'}")
        print(f"  Naive: {naive_answer} {'✓' if result['naive_correct'] else '✗'}")
        print(f"  Beam: {beam_answer} {'✓' if result['beam_correct'] else '✗'}")
        if args.compare_mcmc:
            print(f"  MCMC: {mcmc_answer} {'✓' if result['mcmc_correct'] else '✗'}")
        print(f"{'='*80}")
    
    # ===== Save Results =====
    df = pd.DataFrame(results)
    
    filename = (f"{model}_math_beam_search_"
               f"w{beam_width}_tps{tokens_per_step}_"
               f"t{temp}_b{args.batch_idx}_s{args.seed}.csv")
    
    output_path = os.path.join(save_str, filename)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    
    # ===== Print Aggregate Stats =====
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    print(f"Total problems: {len(results)}")
    print(f"Standard accuracy: {df['std_correct'].mean()*100:.1f}%")
    print(f"Naive temp accuracy: {df['naive_correct'].mean()*100:.1f}%")
    print(f"Beam search accuracy: {df['beam_correct'].mean()*100:.1f}%")
    if args.compare_mcmc:
        print(f"MCMC accuracy: {df['mcmc_correct'].mean()*100:.1f}%")
        if 'mcmc_acceptance_ratio' in df.columns:
            mcmc_acc_mean = df['mcmc_acceptance_ratio'].dropna().mean()
            print(f"Avg MCMC acceptance ratio: {mcmc_acc_mean:.3f}")
    print(f"Avg beam expansions: {df['beam_num_expansions'].mean():.1f}")
    print(f"Avg beam final score: {df['beam_final_score'].mean():.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

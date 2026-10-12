import os
import gc
import json
import random
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import traceback
from datetime import datetime
import itertools

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


def print_memory_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved, 
            "max_allocated_gb": max_allocated
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}


def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def log_config_start(config, run_id):
    """Log the start of a configuration run"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}")
    print(f"[{timestamp}] STARTING CONFIG RUN {run_id}")
    print(f"{'='*80}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}")


def log_config_result(config, run_id, success, error_msg=None, results=None, memory_stats=None):
    """Log the result of a configuration run"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if success else "FAILED"
    print(f"\n[{timestamp}] CONFIG RUN {run_id} - {status}")
    print(f"Configuration: {config}")
    
    if success and results:
        print(f"Results: Pass@{config['beam_width']} = {results['pass_at_k_accuracy']:.4f} ({results['pass_at_k_count']}/{results['total_problems']})")
    
    if error_msg:
        print(f"Error: {error_msg}")
    
    if memory_stats:
        print(f"Memory - Allocated: {memory_stats['allocated_gb']:.2f}GB, Reserved: {memory_stats['reserved_gb']:.2f}GB")
    
    print(f"{'='*80}")


def test_config(config, dataset, model_str, tokenizer, num_problems=5):
    """Test a single configuration"""
    beam_width = config['beam_width']
    max_tokens = config['max_tokens']
    gpu_memory_util = config['gpu_memory_utilization']
    max_model_len = config.get('max_model_len', 2048)
    max_num_batched_tokens = config.get('max_num_batched_tokens', 1024)
    
    try:
        # Clear memory before starting
        clear_memory()
        
        # Initialize vLLM with current config
        llm = LLM(
            model=model_str, 
            trust_remote_code=True, 
            gpu_memory_utilization=gpu_memory_util,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=True,
        )
        
        memory_after_load = print_memory_stats()
        print(f"Memory after model loading: {memory_after_load}")
        
        # Beam search parameters
        beam_params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=max_tokens,
            temperature=0.0,
            length_penalty=1.0,
        )
        
        # Test on subset of problems
        test_dataset = dataset[:num_problems]
        results = []
        successful_problems = 0
        
        for i, data in enumerate(test_dataset):
            try:
                question = data["prompt"]
                answer = data["answer"]
                
                input_text = format_prompt(question, config['model'], tokenizer, config['cot'])
                prompt = {"prompt": input_text}
                
                # Process single problem
                output = llm.beam_search([prompt], beam_params)[0]
                
                # Extract results
                completions = [seq.text for seq in output.sequences]
                answers = [parse_answer(completion) for completion in completions]
                correct_flags = [safe_grade_math(ans, answer) for ans in answers]
                pass_at_k_result = max(correct_flags)
                
                results.append({
                    "problem_idx": i,
                    "pass_at_k": pass_at_k_result,
                })
                
                successful_problems += 1
                print(f"  Problem {i+1}/{num_problems}: {'✓' if pass_at_k_result else '✗'}")
                
            except Exception as e:
                print(f"  Problem {i+1}/{num_problems}: ERROR - {str(e)}")
                continue
        
        # Compute results
        if results:
            total_pass_at_k = sum(r["pass_at_k"] for r in results)
            pass_at_k_accuracy = total_pass_at_k / len(results)
            
            final_results = {
                "total_problems": len(results),
                "successful_problems": successful_problems,
                "pass_at_k_accuracy": pass_at_k_accuracy,
                "pass_at_k_count": total_pass_at_k
            }
            
            # Clean up
            del llm
            clear_memory()
            
            return True, None, final_results, print_memory_stats()
        else:
            del llm
            clear_memory()
            return False, "No problems processed successfully", None, print_memory_stats()
            
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Configuration failed: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Try to clean up
        try:
            if 'llm' in locals():
                del llm
        except:
            pass
        clear_memory()
        
        return False, error_msg, None, print_memory_stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action="store", type=str, default="sweep_results/", dest="save_str")
    parser.add_argument("--model", action="store", default="qwen_math", type=str, 
                       choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--dataset", action="store", default="MATH", type=str)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--num_problems", action="store", type=int, default=5, help="Number of problems to test per config")
    
    # Sweep parameters
    parser.add_argument("--beam_widths", nargs="+", type=int, default=[3, 5, 8, 10, 16], help="Beam widths to test")
    parser.add_argument("--max_tokens_list", nargs="+", type=int, default=[1024, 2048, 3072], help="Max tokens to test")
    parser.add_argument("--gpu_memory_utils", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.85, 0.9], help="GPU memory utilizations to test")
    parser.add_argument("--max_model_lens", nargs="+", type=int, default=[2048, 4096], help="Max model lengths to test")
    parser.add_argument("--max_batched_tokens", nargs="+", type=int, default=[1024, 2048], help="Max batched tokens to test")
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = args.model
    dataset_name = args.dataset
    cot = args.cot
    num_problems = args.num_problems

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    print(f"=== BEAM SEARCH CONFIGURATION SWEEP ===")
    print(f"Model: {model}")
    print(f"Problems per config: {num_problems}")
    print(f"Total configs to test: {len(args.beam_widths) * len(args.max_tokens_list) * len(args.gpu_memory_utils) * len(args.max_model_lens) * len(args.max_batched_tokens)}")
    print(f"Output directory: {save_str}")
    print("="*50)

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

    print(f"Dataset loaded: {len(dataset)} problems (testing on {num_problems} per config)")

    # Load tokenizer once (we'll reload model for each config)
    temp_llm = LLM(model=model_str, trust_remote_code=True, gpu_memory_utilization=0.3)
    tokenizer = temp_llm.get_tokenizer()
    del temp_llm
    clear_memory()

    # Generate all configuration combinations
    config_combinations = list(itertools.product(
        args.beam_widths,
        args.max_tokens_list, 
        args.gpu_memory_utils,
        args.max_model_lens,
        args.max_batched_tokens
    ))
    
    print(f"Testing {len(config_combinations)} configurations...")
    
    # Track results
    sweep_results = []
    run_id = 0
    
    for beam_width, max_tokens, gpu_memory_util, max_model_len, max_batched_tokens in config_combinations:
        run_id += 1
        
        config = {
            "model": model,
            "beam_width": beam_width,
            "max_tokens": max_tokens,
            "gpu_memory_utilization": gpu_memory_util,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_batched_tokens,
            "cot": cot,
            "num_problems": num_problems
        }
        
        log_config_start(config, run_id)
        
        success, error_msg, results, memory_stats = test_config(config, dataset, model_str, tokenizer, num_problems)
        
        log_config_result(config, run_id, success, error_msg, results, memory_stats)
        
        # Record result
        sweep_result = config.copy()
        sweep_result.update({
            "run_id": run_id,
            "success": success,
            "error_msg": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        if success and results:
            sweep_result.update(results)
        
        if memory_stats:
            sweep_result.update({f"final_{k}": v for k, v in memory_stats.items()})
        
        sweep_results.append(sweep_result)
        
        # Save intermediate results
        df = pd.DataFrame(sweep_results)
        interim_file = os.path.join(save_str, f"{model}_beam_sweep_interim_seed_{args.seed}.csv")
        df.to_csv(interim_file, index=False)

    # Final analysis
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE - FINAL ANALYSIS")
    print(f"{'='*80}")
    
    successful_configs = [r for r in sweep_results if r['success']]
    failed_configs = [r for r in sweep_results if not r['success']]
    
    print(f"Total configurations tested: {len(sweep_results)}")
    print(f"Successful configurations: {len(successful_configs)}")
    print(f"Failed configurations: {len(failed_configs)}")
    
    if successful_configs:
        print(f"\nBest performing successful config:")
        best_config = max(successful_configs, key=lambda x: x.get('pass_at_k_accuracy', 0))
        print(f"  Beam width: {best_config['beam_width']}")
        print(f"  Max tokens: {best_config['max_tokens']}")
        print(f"  GPU memory util: {best_config['gpu_memory_utilization']}")
        print(f"  Max model len: {best_config['max_model_len']}")
        print(f"  Max batched tokens: {best_config['max_num_batched_tokens']}")
        print(f"  Pass@k accuracy: {best_config.get('pass_at_k_accuracy', 0):.4f}")
    
    if failed_configs:
        print(f"\nMost common failure patterns:")
        error_types = {}
        for config in failed_configs:
            error = config.get('error_msg', 'Unknown')
            error_type = error.split(':')[0] if ':' in error else error
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(config)
        
        for error_type, configs in error_types.items():
            print(f"  {error_type}: {len(configs)} configs")
    
    # Save final results
    df = pd.DataFrame(sweep_results)
    final_file = os.path.join(save_str, f"{model}_beam_sweep_final_seed_{args.seed}.csv")
    df.to_csv(final_file, index=False)
    
    # Save summary
    summary = {
        "model": model,
        "total_configs": len(sweep_results),
        "successful_configs": len(successful_configs), 
        "failed_configs": len(failed_configs),
        "sweep_parameters": {
            "beam_widths": args.beam_widths,
            "max_tokens_list": args.max_tokens_list,
            "gpu_memory_utils": args.gpu_memory_utils,
            "max_model_lens": args.max_model_lens,
            "max_batched_tokens": args.max_batched_tokens
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if successful_configs:
        summary["best_config"] = best_config
    
    summary_file = os.path.join(save_str, f"{model}_beam_sweep_summary_seed_{args.seed}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {final_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
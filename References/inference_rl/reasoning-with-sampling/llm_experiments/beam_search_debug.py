import gc
import argparse
import sys
import traceback
from datetime import datetime
import pandas as pd
import torch
import os
import json

from vllm import LLM
from vllm.sampling_params import BeamSearchParams

from grader_utils.parse_utils import parse_answer
from grader_utils.math_grader import grade_answer

from power_samp_utils import format_prompt


# Logging functions
def log_info(message):
    """Log an info message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] INFO: {message}")
    sys.stdout.flush()

def log_error(message):
    """Log an error message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ERROR: {message}")
    sys.stdout.flush()

def log_success(message):
    """Log a success message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] SUCCESS: {message}")
    sys.stdout.flush()

def log_warning(message):
    """Log a warning message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] WARNING: {message}")
    sys.stdout.flush()

def log_separator():
    """Print a separator line"""
    print("=" * 80)
    sys.stdout.flush()


def safe_grade_math(ans, correct_ans):
    """Safely grade a math answer, returning 0 on any error"""
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception as e:
        log_warning(f"Failed to grade answer '{ans}' against '{correct_ans}': {str(e)}")
        return 0


def print_memory_stats():
    """Print GPU memory statistics with structured logging"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        log_info(f"GPU Memory - Allocated: {allocated:.2f} GB")
        log_info(f"GPU Memory - Reserved: {reserved:.2f} GB") 
        log_info(f"GPU Memory - Max Allocated: {max_allocated:.2f} GB")
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated
        }
    else:
        log_warning("CUDA not available - cannot get GPU memory stats")
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}


def clear_memory():
    """Clear GPU memory and run garbage collection"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_info("GPU memory cache cleared")
        gc.collect()
        log_info("Python garbage collection completed")
    except Exception as e:
        log_error(f"Failed to clear memory: {str(e)}")


def main():
    """Main function for beam search debugging"""
    
    # Initialize error tracking
    global_errors = []
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_str", action="store", type=str, default="debug_results/", dest="save_str")
        parser.add_argument("--model", action="store", default="qwen_math", type=str, 
                           choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
        parser.add_argument("--temperature", action="store", default=1.0, type=float, dest="temperature")
        parser.add_argument("--dataset", action="store", default="MATH", type=str)
        parser.add_argument("--cot", action="store", type=bool, default=True)
        parser.add_argument("--k", action="store", type=int, default=5, help="Number of beams (reduced for debugging)")
        parser.add_argument("--seed", action="store", type=int, default=0)
        parser.add_argument("--num_problems", action="store", type=int, default=10, help="Number of problems to test (debugging)")
        parser.add_argument("--gpu_memory", action="store", type=float, default=0.7, help="GPU memory utilization fraction")
        parser.add_argument("--max_model_len", action="store", type=int, default=2048, help="Max model sequence length")
        parser.add_argument("--max_num_batched_tokens", action="store", type=int, default=1024, help="Max number of batched tokens")
        parser.add_argument("--max_tokens", action="store", type=int, default=1024, help="Max tokens for beam search")
        args = parser.parse_args()

        log_separator()
        log_info("BEAM SEARCH DEBUG SESSION STARTED")
        log_separator()

        random.seed(args.seed)
        np.random.seed(args.seed)
        log_info(f"Random seeds set to {args.seed}")

        model = args.model
        dataset_name = args.dataset
        cot = args.cot
        temp = args.temperature
        k = args.k
        num_problems = args.num_problems

        # Even more conservative beam width for debugging
        actual_beam_width = min(k, 5)

        save_str = os.path.join(args.save_str, model)
        os.makedirs(save_str, exist_ok=True)
        log_info(f"Results will be saved to: {save_str}")

        log_separator()
        log_info("CONFIGURATION")
        log_separator()
        log_info(f"Model: {model}")
        log_info(f"Beam width: {actual_beam_width} (requested: {k})")
        log_info(f"Number of problems: {num_problems}")
        log_info(f"GPU memory utilization: {args.gpu_memory}")
        log_info(f"Max model length: {args.max_model_len}")
        log_info(f"Max batched tokens: {args.max_num_batched_tokens}")
        log_info(f"Max tokens: {args.max_tokens}")
        log_info(f"Using beam search with temperature 0.0")
        log_info(f"Chain-of-thought: {cot}")
        log_info(f"Dataset: {dataset_name}")
        log_separator()
        
        log_info("ENVIRONMENT VARIABLES")
        log_separator()
        log_info(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set')}")
        log_info(f"TORCH_USE_CUDA_DSA: {os.environ.get('TORCH_USE_CUDA_DSA', 'not set')}")
        log_info(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")
        log_info(f"VLLM_LOGGING_LEVEL: {os.environ.get('VLLM_LOGGING_LEVEL', 'not set')}")
        log_separator()

        # Model mapping
        log_info("Mapping model name to model path...")
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
            error_msg = f"Unknown model: {model}"
            log_error(error_msg)
            global_errors.append(error_msg)
            sys.exit(1)
            
        log_success(f"Model path: {model_str}")

        # Load dataset
        log_info("Loading dataset...")
        try:
            if dataset_name == "MATH":
                json_file = 'data/MATH500.json'
                if not os.path.exists(json_file):
                    raise FileNotFoundError(f"Dataset file not found: {json_file}")
                    
                with open(json_file, "r") as f:
                    dataset = json.load(f)
                    
                # Take only first N problems for debugging
                dataset = dataset[:num_problems]
                log_success(f"Dataset loaded: {len(dataset)} problems (from {json_file})")
            else:
                error_msg = f"Unknown dataset: {dataset_name}"
                log_error(error_msg)
                global_errors.append(error_msg)
                sys.exit(1)
                
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            log_error(error_msg)
            log_error(f"Full traceback: {traceback.format_exc()}")
            global_errors.append(error_msg)
            sys.exit(1)

        # Print initial memory stats
        log_separator()
        log_info("INITIAL GPU MEMORY STATE")
        log_separator()
        initial_memory = print_memory_stats()
        log_separator()

        # Initialize vLLM with very conservative memory settings
        log_info("LOADING MODEL")
        log_separator()
        try:
            log_info(f"Initializing vLLM with model: {model_str}")
            log_info("vLLM settings:")
            log_info(f"  - trust_remote_code: True")
            log_info(f"  - gpu_memory_utilization: {args.gpu_memory}")
            log_info(f"  - max_model_len: {args.max_model_len}")
            log_info(f"  - max_num_batched_tokens: {args.max_num_batched_tokens}")
            log_info(f"  - enforce_eager: True (CUDA graphs disabled for debugging)")
            
            llm = LLM(
                model=model_str, 
                trust_remote_code=True, 
                gpu_memory_utilization=args.gpu_memory,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                enforce_eager=True,  # Disable CUDA graphs for debugging
            )
            tokenizer = llm.get_tokenizer()
            log_success("vLLM model loaded successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR loading model: {str(e)}"
            log_error(error_msg)
            log_error(f"Full traceback: {traceback.format_exc()}")
            log_separator()
            log_info("MEMORY STATE AT ERROR")
            log_separator()
            print_memory_stats()
            global_errors.append(error_msg)
            sys.exit(1)

        # Print memory after model loading
        log_separator()
        log_info("GPU MEMORY AFTER MODEL LOADING")
        log_separator()
        post_load_memory = print_memory_stats()
        memory_increase = post_load_memory["allocated_gb"] - initial_memory["allocated_gb"]
        log_info(f"Model loading memory increase: {memory_increase:.2f} GB")
        log_separator()
    
        # Beam search parameters
        log_info("Setting up beam search parameters...")
        beam_params = BeamSearchParams(
            beam_width=actual_beam_width,
            max_tokens=args.max_tokens,
            temperature=0.0,
            length_penalty=1.0,
        )
        log_info(f"Beam search configuration:")
        log_info(f"  - beam_width: {actual_beam_width}")
        log_info(f"  - max_tokens: {args.max_tokens}")
        log_info(f"  - temperature: 0.0")
        log_info(f"  - length_penalty: 1.0")

        # Prepare all prompts
        log_info("Preparing prompts for all problems...")
        all_prompts = []
        all_metadata = []
        
        try:
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
            
            log_success(f"Prepared {len(all_prompts)} prompts successfully")
            
        except Exception as e:
            error_msg = f"Failed to prepare prompts: {str(e)}"
            log_error(error_msg)
            log_error(f"Full traceback: {traceback.format_exc()}")
            global_errors.append(error_msg)
            sys.exit(1)
        
        # Process one problem at a time with detailed monitoring
        all_outputs = []
        successful_problems = 0
        failed_problems = 0
        
        log_separator()
        log_info("STARTING BEAM SEARCH PROCESSING")
        log_separator()
        log_info(f"Processing {len(all_prompts)} problems with beam width {actual_beam_width}...")
        log_info("Processing strategy: One problem at a time for maximum observability")
        log_separator()
    
        for i, (prompt, metadata) in enumerate(zip(all_prompts, all_metadata)):
            problem_start_time = datetime.now()
            
            log_separator()
            log_info(f"PROCESSING PROBLEM {i+1}/{len(all_prompts)}")
            log_separator()
            
            # Show problem details
            question = metadata['question']
            question_preview = question[:100] + "..." if len(question) > 100 else question
            log_info(f"Problem preview: {question_preview}")
            log_info(f"Correct answer: {metadata['correct_answer']}")
            
            # Print memory before processing
            log_info("GPU memory before processing:")
            memory_before = print_memory_stats()
            
            try:
                log_info(f"Starting beam search for problem {i+1}...")
                
                # Process single problem
                batch_outputs = llm.beam_search([prompt], beam_params)
                all_outputs.extend(batch_outputs)
                
                successful_problems += 1
                processing_time = (datetime.now() - problem_start_time).total_seconds()
                log_success(f"Problem {i+1} completed successfully in {processing_time:.2f}s")
                
                # Print memory after processing
                log_info("GPU memory after processing:")
                memory_after = print_memory_stats()
                memory_used = memory_after["allocated_gb"] - memory_before["allocated_gb"]
                log_info(f"Memory used for this problem: {memory_used:.2f} GB")
                
                # Clear memory after each problem
                log_info("Clearing memory...")
                clear_memory()
                log_info("GPU memory after cleanup:")
                print_memory_stats()
                
            except Exception as e:
                failed_problems += 1
                processing_time = (datetime.now() - problem_start_time).total_seconds()
                error_msg = f"ERROR on problem {i+1} after {processing_time:.2f}s: {str(e)}"
                log_error(error_msg)
                log_error(f"Full traceback: {traceback.format_exc()}")
                global_errors.append(f"Problem {i+1}: {error_msg}")
                
                log_info("GPU memory at error:")
                print_memory_stats()
                
                # Try to clear memory and continue
                log_info("Attempting memory cleanup after error...")
                clear_memory()
                log_info("GPU memory after error cleanup:")
                print_memory_stats()
                
                # Add empty output to maintain alignment
                all_outputs.append(None)
                continue
                
            # Progress update
            log_info(f"Progress: {successful_problems} successful, {failed_problems} failed, {len(all_prompts) - i - 1} remaining")
    
        # Process outputs
        log_separator()
        log_info("PROCESSING RESULTS")
        log_separator()
        results = []
        successful_outputs = 0
        grading_errors = 0
        
        for i, (output, metadata) in enumerate(zip(all_outputs, all_metadata)):
            if output is None:
                log_warning(f"Skipping problem {i+1} (failed during beam search)")
                continue
                
            question = metadata["question"]
            correct_answer = metadata["correct_answer"]
            
            try:
                log_info(f"Processing results for problem {i+1}...")
                
                # Extract all k beam sequences for this problem
                completions = [seq.text for seq in output.sequences]
                answers = [parse_answer(completion) for completion in completions]
                
                log_info(f"Problem {i+1}: Extracted {len(completions)} completions")
                log_info(f"Problem {i+1}: Parsed {len(answers)} answers")
                
                # Compute pass@k results
                correct_flags = [safe_grade_math(ans, correct_answer) for ans in answers]
                pass_at_k_result = max(correct_flags)  # 1 if any answer is correct, 0 otherwise
                correct_count = sum(correct_flags)
                
                # Store individual results
                problem_result = {
                    "problem_idx": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "completions": completions,
                    "answers": answers,
                    "correct_flags": correct_flags,
                    "pass_at_k": pass_at_k_result,
                    "k": actual_beam_width
                }
                
                results.append(problem_result)
                successful_outputs += 1
                
                # Log detailed results
                if pass_at_k_result:
                    log_success(f"Problem {i+1}: Pass@{actual_beam_width} = {pass_at_k_result} ({correct_count}/{len(answers)} correct)")
                else:
                    log_warning(f"Problem {i+1}: Pass@{actual_beam_width} = {pass_at_k_result} ({correct_count}/{len(answers)} correct)")
                
                # Show first few answers for debugging
                log_info(f"Problem {i+1} sample answers:")
                for j, (answer, is_correct) in enumerate(zip(answers[:3], correct_flags[:3])):
                    status = "✓" if is_correct else "✗"
                    log_info(f"  Beam {j+1}: {status} '{answer}'")
                
            except Exception as e:
                grading_errors += 1
                error_msg = f"ERROR processing results for problem {i+1}: {str(e)}"
                log_error(error_msg)
                log_error(f"Full traceback: {traceback.format_exc()}")
                global_errors.append(f"Results processing problem {i+1}: {error_msg}")
                continue

        # Compute overall results
        if results:
            total_pass_at_k = sum(result["pass_at_k"] for result in results)
            pass_at_k_accuracy = total_pass_at_k / len(results)
            
            log_separator()
            log_info("FINAL RESULTS SUMMARY")
            log_separator()
            log_info(f"Total problems attempted: {len(all_prompts)}")
            log_info(f"Beam search successful: {successful_problems}")
            log_info(f"Beam search failed: {failed_problems}")
            log_info(f"Results processing successful: {successful_outputs}")
            log_info(f"Results processing errors: {grading_errors}")
            log_success(f"Overall Pass@{actual_beam_width} Accuracy: {pass_at_k_accuracy:.4f} ({total_pass_at_k}/{len(results)})")
            log_separator()

            # Save results
            try:
                df = pd.DataFrame(results)
                output_file = os.path.join(save_str, f"{model}_debug_beam_{actual_beam_width}_temp_{temp}_seed_{args.seed}.csv")
                df.to_csv(output_file, index=False)
                log_success(f"Results saved to: {output_file}")
                
                # Save summary
                summary = {
                    "model": model,
                    "method": "beam_search_debug",
                    "beam_width": actual_beam_width,
                    "temperature": 0.0,
                    "length_penalty": 1.0,
                    "total_problems_attempted": len(all_prompts),
                    "successful_problems": successful_problems,
                    "failed_problems": failed_problems,
                    "successful_outputs": successful_outputs,
                    "grading_errors": grading_errors,
                    "pass_at_k_accuracy": pass_at_k_accuracy,
                    "pass_at_k_count": total_pass_at_k,
                    "gpu_memory_utilization": args.gpu_memory,
                    "errors": global_errors,
                    "session_timestamp": datetime.now().isoformat()
                }
                
                summary_file = os.path.join(save_str, f"{model}_debug_beam_{actual_beam_width}_summary_seed_{args.seed}.json")
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                log_success(f"Summary saved to: {summary_file}")
                
            except Exception as e:
                error_msg = f"Failed to save results: {str(e)}"
                log_error(error_msg)
                log_error(f"Full traceback: {traceback.format_exc()}")
                global_errors.append(error_msg)
                
        else:
            log_separator()
            log_error("CRITICAL: No problems were successfully processed!")
            log_error("This indicates a major issue with the beam search setup")
            log_separator()
            # Still try to save error information
            try:
                error_summary = {
                    "model": model,
                    "method": "beam_search_debug",
                    "status": "TOTAL_FAILURE",
                    "total_problems_attempted": len(all_prompts),
                    "successful_problems": 0,
                    "errors": global_errors,
                    "session_timestamp": datetime.now().isoformat()
                }
                
                error_file = os.path.join(save_str, f"{model}_debug_FAILED_seed_{args.seed}.json")
                with open(error_file, 'w') as f:
                    json.dump(error_summary, f, indent=2)
                    
                log_info(f"Error summary saved to: {error_file}")
            except Exception as e:
                log_error(f"Could not even save error summary: {str(e)}")

        # Final memory stats
        log_separator()
        log_info("FINAL GPU MEMORY STATE")
        log_separator()
        final_memory = print_memory_stats()
        log_separator()
        
        # Session summary
        if global_errors:
            log_error(f"Session completed with {len(global_errors)} errors:")
            for i, error in enumerate(global_errors, 1):
                log_error(f"  {i}. {error}")
        
        # Exit with appropriate code
        if failed_problems == 0 and len(global_errors) == 0:
            log_success("BEAM SEARCH DEBUG SESSION COMPLETED SUCCESSFULLY")
            sys.exit(0)
        elif successful_problems > 0:
            log_warning(f"BEAM SEARCH DEBUG SESSION COMPLETED WITH ISSUES ({successful_problems} successes, {failed_problems} failures)")
            sys.exit(1)
        else:
            log_error("BEAM SEARCH DEBUG SESSION FAILED COMPLETELY")
            sys.exit(2)
            
    except Exception as e:
        # Catch-all error handler
        log_separator()
        log_error("CRITICAL UNHANDLED ERROR IN MAIN")
        log_separator()
        log_error(f"Error: {str(e)}")
        log_error(f"Full traceback: {traceback.format_exc()}")
        
        # Try to save critical error info
        try:
            critical_error = {
                "status": "CRITICAL_ERROR",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open("debug_results/CRITICAL_ERROR.json", "w") as f:
                json.dump(critical_error, f, indent=2)
                
            log_info("Critical error saved to debug_results/CRITICAL_ERROR.json")
        except:
            pass
            
        sys.exit(3)


if __name__ == "__main__":
    main()
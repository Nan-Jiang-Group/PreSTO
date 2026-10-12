import time
from typing import Dict, Any, Optional
from collections import Counter

import torch

from .models import ModelWrapper
from .answer_extraction import extract_answer, normalize_answer


def run_grpo_single(
    prompt: str,
    model_wrapper: ModelWrapper,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if seed is not None:
        torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    prompt_ids = model_wrapper.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)
    output_ids = model_wrapper.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    wall_time = end_time - start_time
    if output_ids.dim() > 1:
        output_ids = output_ids[0]
    generated_ids = output_ids[prompt_len:]
    output_text = model_wrapper.decode(generated_ids)
    raw_answer = extract_answer(output_text)
    final_answer = normalize_answer(raw_answer)
    
    return {
        "output_text": output_text,
        "output_ids": generated_ids.cpu().tolist() if isinstance(generated_ids, torch.Tensor) else generated_ids,
        "final_answer": final_answer,
        "raw_answer": raw_answer,
        "wall_time_s": wall_time,
        "output_tokens": len(generated_ids),
        "method": "grpo_single",
    }


def run_grpo_majority_vote(
    prompt: str,
    model_wrapper: ModelWrapper,
    n_samples: int = 10,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    tie_break: str = "logprob",  # "logprob" or "shortest"
    verbose: bool = False,
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    prompt_ids = model_wrapper.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)
    samples = []
    all_answers = []
    total_output_tokens = 0
    
    for i in range(n_samples):
        if seed is not None:
            torch.manual_seed(seed + i)
            
        if verbose:
            print(f"Generating sample {i+1}/{n_samples}...")
            
        # Generate
        output_ids = model_wrapper.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )
        
        if output_ids.dim() > 1:
            output_ids = output_ids[0]
        generated_ids = output_ids[prompt_len:]
        output_text = model_wrapper.decode(generated_ids)
        raw_answer = extract_answer(output_text)
        normalized_answer = normalize_answer(raw_answer)
        logprob = None
        if tie_break == "logprob" and len(generated_ids) > 0:
            logprob = model_wrapper.teacher_forced_logp(output_ids)
            
        samples.append({
            "output_text": output_text,
            "output_ids": generated_ids.cpu().tolist() if isinstance(generated_ids, torch.Tensor) else generated_ids,
            "raw_answer": raw_answer,
            "normalized_answer": normalized_answer,
            "n_tokens": len(generated_ids),
            "logprob": logprob,
        })
        
        all_answers.append(normalized_answer)
        total_output_tokens += len(generated_ids)
        
    vote_counts = Counter(all_answers)
    
    if verbose:
        print(f"Vote counts: {vote_counts}")
        
    max_votes = max(vote_counts.values())
    winners = [ans for ans, count in vote_counts.items() if count == max_votes]
    
    if len(winners) == 1:
        final_answer = winners[0]
    else:
        if tie_break == "logprob":
            answer_logprobs = {}
            for ans in winners:
                relevant_samples = [s for s in samples if s["normalized_answer"] == ans]
                logprobs = [s["logprob"] for s in relevant_samples if s["logprob"] is not None]
                if logprobs:
                    answer_logprobs[ans] = sum(logprobs) / len(logprobs)
                else:
                    answer_logprobs[ans] = float("-inf")
                    
            final_answer = max(answer_logprobs.keys(), key=lambda x: answer_logprobs[x])
            
            if verbose:
                print(f"Tie-break by logprob: {answer_logprobs}")
                
        elif tie_break == "shortest":
            final_answer = min(winners, key=len)
            
            if verbose:
                print(f"Tie-break by length: chose '{final_answer}'")
        else:
            final_answer = winners[0]
            
    winning_sample = next(s for s in samples if s["normalized_answer"] == final_answer)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    wall_time = end_time - start_time
    
    return {
        "final_answer": final_answer,
        "output_text": winning_sample["output_text"],
        "output_ids": winning_sample["output_ids"],
        "raw_answer": winning_sample["raw_answer"],
        "wall_time_s": wall_time,
        "output_tokens": winning_sample["n_tokens"],
        "total_output_tokens": total_output_tokens,
        "n_samples": n_samples,
        "vote_counts": dict(vote_counts),
        "all_answers": all_answers,
        "method": "grpo_majority_vote",
        # For compatibility with MCMC wilcoxon_test_mcmc_vs_standard
        "internal_generated_tokens": total_output_tokens,
        "accept_rate": None,
        "num_steps": 0,
    }


def run_baseline(
    prompt: str,
    model_wrapper: ModelWrapper,
    method: str = "single",
    **kwargs
) -> Dict[str, Any]:
    if method == "single":
        return run_grpo_single(prompt, model_wrapper, **kwargs)
    elif method == "majority_vote" or method == "vote":
        return run_grpo_majority_vote(prompt, model_wrapper, **kwargs)
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def compare_with_mcmc(
    baseline_result: Dict[str, Any],
    mcmc_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "baseline_answer": baseline_result.get("final_answer"),
        "mcmc_answer": mcmc_result.get("final_answer"),
        "answers_match": baseline_result.get("final_answer") == mcmc_result.get("final_answer"),
        "baseline_time_s": baseline_result.get("wall_time_s"),
        "mcmc_time_s": mcmc_result.get("wall_time_s"),
        "time_ratio": (mcmc_result.get("wall_time_s", 0) / baseline_result.get("wall_time_s", 1))
                      if baseline_result.get("wall_time_s", 0) > 0 else float("inf"),
        "baseline_tokens": baseline_result.get("total_output_tokens", baseline_result.get("output_tokens")),
        "mcmc_tokens": mcmc_result.get("output_tokens"),
        "mcmc_internal_tokens": mcmc_result.get("internal_generated_tokens"),
        "mcmc_accept_rate": mcmc_result.get("accept_rate"),
    }

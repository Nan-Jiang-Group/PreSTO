import time
import random
import math
import torch
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field

from .models import ModelWrapper
from .sampling import (
    sample_suffix,
    logprob_suffix_given_prefix,
    compute_base_logp,
    extend_sequence,
)
from .answer_extraction import extract_answer, normalize_answer


@dataclass
class MCMCStats:
    total_proposals: int = 0
    accepted_proposals: int = 0
    log_acceptance_ratios: List[float] = field(default_factory=list)
    block_sizes: List[int] = field(default_factory=list)
    
    @property
    def accept_rate(self) -> float:
        if self.total_proposals == 0:
            return 0.0
        return self.accepted_proposals / self.total_proposals
    
    @property
    def avg_logA(self) -> float:
        if not self.log_acceptance_ratios:
            return 0.0
        return sum(self.log_acceptance_ratios) / len(self.log_acceptance_ratios)


def compute_mh_acceptance(
    model_wrapper: ModelWrapper,
    prefix_ids: torch.Tensor,
    old_suffix_ids: torch.Tensor,
    new_suffix_ids: torch.Tensor,
    alpha: float,
    proposal_temperature: float,
) -> Tuple[float, Dict[str, float]]:
    device = model_wrapper.device
    
    if prefix_ids.dim() > 1:
        prefix_ids = prefix_ids.squeeze(0)
    if old_suffix_ids.dim() > 1:
        old_suffix_ids = old_suffix_ids.squeeze(0)
    if new_suffix_ids.dim() > 1:
        new_suffix_ids = new_suffix_ids.squeeze(0)
    
    prefix_ids = prefix_ids.to(device)
    old_suffix_ids = old_suffix_ids.to(device)
    new_suffix_ids = new_suffix_ids.to(device)
    
    old_full = torch.cat([prefix_ids, old_suffix_ids], dim=0)
    new_full = torch.cat([prefix_ids, new_suffix_ids], dim=0)
    
    prefix_len = len(prefix_ids)
    
    log_p_old_suffix = compute_base_logp(
        model_wrapper, 
        old_full, 
        start_pos=prefix_len - 1 if prefix_len > 0 else 0
    )
    
    log_p_new_suffix = compute_base_logp(
        model_wrapper,
        new_full,
        start_pos=prefix_len - 1 if prefix_len > 0 else 0
    )
    
    log_pi_ratio = alpha * (log_p_new_suffix - log_p_old_suffix)
    
    log_q_forward = logprob_suffix_given_prefix(
        model_wrapper,
        prefix_ids,
        new_suffix_ids,
        temperature=proposal_temperature,
    )
    
    log_q_reverse = logprob_suffix_given_prefix(
        model_wrapper,
        prefix_ids,
        old_suffix_ids,
        temperature=proposal_temperature,
    )
    
    log_q_ratio = log_q_reverse - log_q_forward
    
    log_A = log_pi_ratio + log_q_ratio
    
    debug_info = {
        "log_p_old_suffix": log_p_old_suffix,
        "log_p_new_suffix": log_p_new_suffix,
        "log_pi_ratio": log_pi_ratio,
        "log_q_forward": log_q_forward,
        "log_q_reverse": log_q_reverse,
        "log_q_ratio": log_q_ratio,
        "log_A": log_A,
    }
    
    return log_A, debug_info


def mh_step(
    model_wrapper: ModelWrapper,
    current_ids: torch.Tensor,
    prompt_len: int,
    alpha: float,
    proposal_temperature: float,
    max_suffix_len: int,
    eos_token_id: int,
    top_p: float = 1.0,
    stats: Optional[MCMCStats] = None,
) -> Tuple[torch.Tensor, bool, float, int]:
    current_len = len(current_ids)
    
    min_cut = prompt_len + 1  
    max_cut = current_len     
    
    if min_cut > max_cut:
        return current_ids, False, 0.0, 0
    
    cut_pos = random.randint(min_cut, max_cut)
    
    prefix_ids = current_ids[:cut_pos]
    old_suffix_ids = current_ids[cut_pos:]
    
    target_suffix_len = len(old_suffix_ids)
    if target_suffix_len == 0:
        target_suffix_len = min(max_suffix_len, 10)  # Generate something
    
    new_suffix_ids, log_q_forward_direct, n_gen_tokens = sample_suffix(
        model_wrapper,
        prefix_ids,
        max_new_tokens=target_suffix_len,
        temperature=proposal_temperature,
        eos_token_id=eos_token_id,
        top_p=top_p,
    )
    
    log_A, debug_info = compute_mh_acceptance(
        model_wrapper,
        prefix_ids,
        old_suffix_ids,
        new_suffix_ids,
        alpha,
        proposal_temperature,
    )

    log_u = math.log(random.random()) if random.random() > 0 else float('-inf')
    accepted = log_u <= min(0.0, log_A)
    
    if accepted:
        device = model_wrapper.device
        prefix_ids = prefix_ids.to(device)
        new_suffix_ids = new_suffix_ids.to(device)
        new_ids = torch.cat([prefix_ids, new_suffix_ids], dim=0)
    else:
        new_ids = current_ids
        
    if stats is not None:
        stats.total_proposals += 1
        if accepted:
            stats.accepted_proposals += 1
        if not math.isinf(log_A) and not math.isnan(log_A):
            stats.log_acceptance_ratios.append(log_A)
            
    return new_ids, accepted, log_A, n_gen_tokens


def power_sampling_mcmc(
    prompt: str,
    model_wrapper: ModelWrapper,
    alpha: float = 4.0,
    B: int = 192,
    N_mcmc: int = 10,
    T_max: int = 1024,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    proposal_temperature = 1.0 / alpha
    eos_token_id = model_wrapper.tokenizer.eos_token_id
    
    prompt_ids = model_wrapper.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)
    
    if verbose:
        print(f"Power Sampling MCMC: α={alpha}, B={B}, N_mcmc={N_mcmc}, T_max={T_max}")
        print(f"Proposal temperature: {proposal_temperature}")
        print(f"Prompt length: {prompt_len} tokens")
    
    current_ids = prompt_ids.to(model_wrapper.device)
    
    stats = MCMCStats()
    total_internal_tokens = 0
    hit_eos = False
    
    block_idx = 0
    while len(current_ids) < T_max and not hit_eos:
        target_len = min(prompt_len + (block_idx + 1) * B, T_max)
        
        if verbose:
            print(f"\nBlock {block_idx}: target length {target_len}")
            
        if len(current_ids) < target_len:
            current_ids, logq_ext, n_ext, hit_eos = extend_sequence(
                model_wrapper,
                current_ids,
                target_length=target_len,
                temperature=proposal_temperature,
                eos_token_id=eos_token_id,
                top_p=top_p,
            )
            total_internal_tokens += n_ext
            
            if verbose:
                print(f"  Extended by {n_ext} tokens, new length: {len(current_ids)}")
                
            if hit_eos:
                if verbose:
                    print("  Hit EOS during extension")
                break
                
        if N_mcmc > 0:
            current_block_len = len(current_ids)
            max_suffix_for_mh = current_block_len - prompt_len
            
            for mh_iter in range(N_mcmc):
                current_ids, accepted, log_A, n_gen = mh_step(
                    model_wrapper,
                    current_ids,
                    prompt_len=prompt_len,
                    alpha=alpha,
                    proposal_temperature=proposal_temperature,
                    max_suffix_len=max_suffix_for_mh,
                    eos_token_id=eos_token_id,
                    top_p=top_p,
                    stats=stats,
                )
                total_internal_tokens += n_gen
                
                if len(current_ids) > 0 and current_ids[-1].item() == eos_token_id:
                    hit_eos = True
                    
            if verbose:
                print(f"  MH: {stats.accepted_proposals}/{stats.total_proposals} accepted "
                      f"({stats.accept_rate:.2%}), avg logA: {stats.avg_logA:.2f}")
                
        stats.block_sizes.append(len(current_ids))
        block_idx += 1
        
        if block_idx > 100:
            if verbose:
                print("Warning: Too many blocks, stopping")
            break
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    wall_time = end_time - start_time
    
    # Extract output (everything after prompt)
    output_ids = current_ids[prompt_len:]
    output_text = model_wrapper.decode(output_ids)
    full_text = model_wrapper.decode(current_ids)
    
    # Extract answer
    raw_answer = extract_answer(output_text)
    final_answer = normalize_answer(raw_answer)
    
    result = {
        "output_text": output_text,
        "output_ids": output_ids.cpu().tolist() if isinstance(output_ids, torch.Tensor) else output_ids,
        "full_text": full_text,
        "final_answer": final_answer,
        "raw_answer": raw_answer,
        "wall_time_s": wall_time,
        "output_tokens": len(output_ids),
        "internal_generated_tokens": total_internal_tokens,
        "accept_rate": stats.accept_rate,
        "num_steps": stats.total_proposals,
        "avg_logA": stats.avg_logA,
        "num_blocks": block_idx,
        "alpha": alpha,
        "B": B,
        "N_mcmc": N_mcmc,
        "T_max": T_max,
    }
    
    if verbose:
        print(f"\nCompleted in {wall_time:.2f}s")
        print(f"Output: {len(output_ids)} tokens, internal: {total_internal_tokens} tokens")
        print(f"Final answer: {final_answer}")
        
    return result


def power_sampling_mcmc_batch(
    prompts: List[str],
    model_wrapper: ModelWrapper,
    alpha: float = 4.0,
    B: int = 192,
    N_mcmc: int = 10,
    T_max: int = 1024,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    
    for i, prompt in enumerate(prompts):
        if seed is not None:
            current_seed = seed + i
        else:
            current_seed = None
            
        if verbose:
            print(f"\n=== Processing prompt {i+1}/{len(prompts)} ===")
            
        result = power_sampling_mcmc(
            prompt=prompt,
            model_wrapper=model_wrapper,
            alpha=alpha,
            B=B,
            N_mcmc=N_mcmc,
            T_max=T_max,
            top_p=top_p,
            seed=current_seed,
            verbose=verbose,
        )
        results.append(result)
        
    return results

"""
Beam search with power sampling for language models.
Maintains multiple hypotheses and scores using p^α distribution.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from power_samp_utils import AutoregressiveSampler, naive_temp


@dataclass
class Beam:
    """Represents a single beam hypothesis in beam search."""
    sequence: list[int]              # Token IDs
    log_probs_norm: list[float]      # q(token_i) - proposal distribution log probs
    log_probs_unnorm: list[float]    # p^α(token_i) - target distribution log probs
    score: float                      # Length-normalized cumulative score
    is_completed: bool                # Has EOS token
    
    def __repr__(self):
        return (f"Beam(len={len(self.sequence)}, score={self.score:.4f}, "
                f"completed={self.is_completed})")


def calculate_beam_score(
    log_probs_unnorm: list[float],
    sequence_length: int,
    length_penalty: float = 0.6
) -> float:
    """
    Calculate length-normalized score for beam.
    
    Formula: sum(log_probs) / (length ^ length_penalty)
    
    Args:
        log_probs_unnorm: Target distribution log probs (p^α)
        sequence_length: Number of generated tokens
        length_penalty: Normalization exponent (0.6-1.0 typical)
    
    Returns:
        Normalized score (higher is better)
    """
    if sequence_length == 0:
        return 0.0
    
    cumulative_score = sum(log_probs_unnorm)
    normalized_score = cumulative_score / (sequence_length ** length_penalty)
    return normalized_score


def print_beam_info(beams: list[Beam], tokenizer, top_k: int = 3):
    """Print top-k beams for debugging."""
    print(f"\n{'='*80}")
    print(f"Top {top_k} Beams:")
    print(f"{'='*80}")
    
    for i, beam in enumerate(beams[:top_k]):
        decoded = tokenizer.decode(beam.sequence, skip_special_tokens=False)
        print(f"\nBeam {i+1} (score: {beam.score:.4f}, len: {len(beam.sequence)}):")
        print(f"  Text: {decoded[:100]}...")
        print(f"  Completed: {beam.is_completed}")
        if beam.log_probs_unnorm:
            print(f"  Avg log_prob_unnorm: {np.mean(beam.log_probs_unnorm):.4f}")


def beam_search_power_samp(
    p: AutoregressiveSampler,
    context: list[int],
    temp: float,
    beam_width: int = 5,
    tokens_per_step: int = 16,
    max_new_tokens: int = 3072,
    length_penalty: float = 0.6,
    verbose: bool = False
) -> tuple[list[int], list[float], list[float], dict]:
    """
    Beam search with power sampling from p^α distribution.
    
    Maintains beam_width parallel hypotheses, generates tokens_per_step tokens
    at a time, and scores using the target power-scaled distribution p^α where
    α = 1/temp.
    
    Args:
        p: AutoregressiveSampler instance
        context: Initial token IDs (prefix)
        temp: Temperature for p^α (α = 1/temp)
        beam_width: Number of parallel beams to maintain
        tokens_per_step: Generate this many tokens per expansion
        max_new_tokens: Maximum total tokens to generate
        length_penalty: Length normalization exponent
        verbose: Print debugging info
    
    Returns:
        (best_sequence, log_probs_norm, log_probs_unnorm, metadata)
        metadata = {
            'num_expansions': int,
            'final_beam_scores': list[float],
            'num_completed_beams': int,
            'final_score': float
        }
    """
    c = len(context)
    
    if verbose:
        print(f"\nBeam Search Power Sampling:")
        print(f"  α (power) = 1/{temp} = {1/temp:.2f}")
        print(f"  beam_width = {beam_width}")
        print(f"  tokens_per_step = {tokens_per_step}")
        print(f"  max_new_tokens = {max_new_tokens}")
        print(f"  length_penalty = {length_penalty}")
    
    # Initialize with context
    initial_beam = Beam(
        sequence=context.copy(),
        log_probs_norm=[],
        log_probs_unnorm=[],
        score=0.0,
        is_completed=False
    )
    
    active_beams = [initial_beam]
    completed_beams = []
    num_expansions = 0
    
    # Main beam search loop
    progress_bar = tqdm(total=max_new_tokens, desc="Beam search") if verbose else None
    
    while active_beams and (len(active_beams[0].sequence) - c) < max_new_tokens:
        candidate_beams = []
        
        # EXPANSION PHASE: Generate next chunk for each active beam
        for beam in active_beams:
            current_len = len(beam.sequence)
            target_len = min(current_len + tokens_per_step, c + max_new_tokens)
            
            if target_len <= current_len:
                # Already at max length
                beam.is_completed = True
                candidate_beams.append(beam)
                continue
            
            try:
                # Generate next chunk using naive_temp
                new_seq, lp_norm_chunk, lp_unnorm_chunk = naive_temp(
                    p,
                    beam.sequence,
                    temp=temp,
                    seq_len=target_len
                )
                
                # Create new beam with extended sequence
                new_beam = Beam(
                    sequence=new_seq.copy(),
                    log_probs_norm=beam.log_probs_norm + lp_norm_chunk,
                    log_probs_unnorm=beam.log_probs_unnorm + lp_unnorm_chunk,
                    score=0.0,  # Will be calculated in scoring phase
                    is_completed=False
                )
                
                # Check for EOS token
                if p.tokenizer.eos_token_id in new_seq[current_len:]:
                    # Find first EOS after current position
                    eos_idx = new_seq.index(p.tokenizer.eos_token_id, current_len)
                    # Truncate at EOS
                    tokens_generated = eos_idx - c
                    new_beam.sequence = new_seq[:eos_idx + 1]
                    new_beam.log_probs_norm = new_beam.log_probs_norm[:tokens_generated]
                    new_beam.log_probs_unnorm = new_beam.log_probs_unnorm[:tokens_generated]
                    new_beam.is_completed = True
                
                candidate_beams.append(new_beam)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Expansion failed for beam: {e}")
                # Keep original beam if expansion fails
                beam.is_completed = True
                candidate_beams.append(beam)
        
        num_expansions += 1
        
        # SCORING PHASE: Calculate scores for all candidates
        for beam in candidate_beams:
            num_generated = len(beam.sequence) - c
            if num_generated > 0:
                beam.score = calculate_beam_score(
                    beam.log_probs_unnorm,
                    num_generated,
                    length_penalty
                )
            else:
                beam.score = 0.0
        
        # PRUNING PHASE: Separate completed and active, keep top-k
        completed = [b for b in candidate_beams if b.is_completed]
        active = [b for b in candidate_beams if not b.is_completed]
        
        # Sort by score (descending)
        completed.sort(key=lambda b: b.score, reverse=True)
        active.sort(key=lambda b: b.score, reverse=True)
        
        # Add completed beams
        completed_beams.extend(completed[:beam_width])
        
        # Keep top beam_width active beams
        active_beams = active[:beam_width]
        
        # Update progress
        if progress_bar and active_beams:
            current_max_len = max(len(b.sequence) for b in active_beams) - c
            progress_bar.n = min(current_max_len, max_new_tokens)
            progress_bar.refresh()
        
        # TERMINATION CHECK
        if len(completed_beams) >= beam_width:
            if verbose:
                print(f"\n  Stopping: {len(completed_beams)} beams completed")
            break
        
        if not active_beams:
            if verbose:
                print(f"\n  Stopping: No active beams remaining")
            break
        
        if verbose and num_expansions % 5 == 0:
            print(f"\n  Expansion {num_expansions}: "
                  f"{len(active_beams)} active, {len(completed_beams)} completed")
            if active_beams:
                best_active = active_beams[0]
                print(f"    Best active score: {best_active.score:.4f} "
                      f"(len={len(best_active.sequence)-c})")
    
    if progress_bar:
        progress_bar.close()
    
    # RETURN BEST BEAM
    all_final_beams = completed_beams + active_beams
    
    if not all_final_beams:
        # Fallback: return context if no beams generated
        return (
            context,
            [],
            [],
            {
                'num_expansions': num_expansions,
                'final_beam_scores': [],
                'num_completed_beams': 0,
                'final_score': 0.0
            }
        )
    
    # Sort all beams by score
    all_final_beams.sort(key=lambda b: b.score, reverse=True)
    best_beam = all_final_beams[0]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Beam Search Complete:")
        print(f"  Total expansions: {num_expansions}")
        print(f"  Completed beams: {len(completed_beams)}")
        print(f"  Best beam score: {best_beam.score:.4f}")
        print(f"  Best beam length: {len(best_beam.sequence) - c} tokens")
        print(f"{'='*80}")
    
    metadata = {
        'num_expansions': num_expansions,
        'final_beam_scores': [b.score for b in all_final_beams[:beam_width]],
        'num_completed_beams': len(completed_beams),
        'final_score': best_beam.score
    }
    
    return (
        best_beam.sequence,
        best_beam.log_probs_norm,
        best_beam.log_probs_unnorm,
        metadata
    )


def beam_search_greedy(
    p: AutoregressiveSampler,
    context: list[int],
    beam_width: int = 5,
    tokens_per_step: int = 16,
    max_new_tokens: int = 3072,
    length_penalty: float = 0.6,
    verbose: bool = False
) -> tuple[list[int], list[float], list[float], dict]:
    """
    Greedy beam search (α → ∞, equivalent to temp → 0).
    
    Similar to beam_search_power_samp but uses very low temperature
    for near-greedy generation.
    
    Args:
        p: AutoregressiveSampler instance
        context: Initial token IDs (prefix)
        beam_width: Number of parallel beams to maintain
        tokens_per_step: Generate this many tokens per expansion
        max_new_tokens: Maximum total tokens to generate
        length_penalty: Length normalization exponent
        verbose: Print debugging info
    
    Returns:
        (best_sequence, log_probs_norm, log_probs_unnorm, metadata)
    """
    # Use very low temperature for greedy behavior
    return beam_search_power_samp(
        p=p,
        context=context,
        temp=0.01,  # Very low temp ≈ greedy (α = 100)
        beam_width=beam_width,
        tokens_per_step=tokens_per_step,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        verbose=verbose
    )

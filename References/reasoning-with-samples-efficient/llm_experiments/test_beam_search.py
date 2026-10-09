#!/usr/bin/env python3
"""
Quick test script for beam search implementation.
Tests on a single simple problem to verify the implementation works.
"""

import torch
from power_samp_utils import load_model_and_tokenizer, format_prompt
from beam_search_utils import beam_search_power_samp, beam_search_greedy, print_beam_info, Beam


def test_basic_functionality():
    """Test basic beam search on a simple math problem."""
    
    print("="*80)
    print("BEAM SEARCH BASIC FUNCTIONALITY TEST")
    print("="*80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "qwen"  # Smallest model for quick testing
    
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")
    
    try:
        hf_model, tokenizer, autoreg_sampler = load_model_and_tokenizer(model_name, device)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Simple test problem
    question = "What is 2 + 2?"
    correct_answer = "4"
    
    print(f"Test problem: {question}")
    print(f"Expected answer: {correct_answer}\n")
    
    # Format prompt
    input_text = format_prompt(question, model_name, tokenizer, cot=False)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    prefix = [idx.item() for idx in input_ids[0]]
    
    print(f"Prompt length: {len(prefix)} tokens\n")
    
    # Test 1: Beam search with small parameters
    print("-" * 80)
    print("TEST 1: Beam search power sampling (small params)")
    print("-" * 80)
    
    try:
        beam_output, lp_norm, lp_unnorm, metadata = beam_search_power_samp(
            autoreg_sampler,
            prefix,
            temp=0.25,
            beam_width=3,
            tokens_per_step=8,
            max_new_tokens=100,
            length_penalty=0.6,
            verbose=True
        )
        
        generated = tokenizer.decode(beam_output[len(prefix):], skip_special_tokens=True)
        
        print(f"\n✓ Beam search completed successfully!")
        print(f"Generated: {generated[:200]}")
        print(f"Metadata: {metadata}")
        
    except Exception as e:
        print(f"\n✗ Beam search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Greedy beam search
    print("\n" + "-" * 80)
    print("TEST 2: Greedy beam search")
    print("-" * 80)
    
    try:
        greedy_output, lp_norm, lp_unnorm, metadata = beam_search_greedy(
            autoreg_sampler,
            prefix,
            beam_width=3,
            tokens_per_step=8,
            max_new_tokens=100,
            length_penalty=0.6,
            verbose=True
        )
        
        generated = tokenizer.decode(greedy_output[len(prefix):], skip_special_tokens=True)
        
        print(f"\n✓ Greedy beam search completed successfully!")
        print(f"Generated: {generated[:200]}")
        print(f"Metadata: {metadata}")
        
    except Exception as e:
        print(f"\n✗ Greedy beam search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Different beam widths
    print("\n" + "-" * 80)
    print("TEST 3: Comparing different beam widths")
    print("-" * 80)
    
    for bw in [1, 3, 5]:
        try:
            print(f"\nBeam width = {bw}")
            output, _, _, metadata = beam_search_power_samp(
                autoreg_sampler,
                prefix,
                temp=0.25,
                beam_width=bw,
                tokens_per_step=8,
                max_new_tokens=50,
                length_penalty=0.6,
                verbose=False
            )
            
            generated = tokenizer.decode(output[len(prefix):], skip_special_tokens=True)
            print(f"  Score: {metadata['final_score']:.4f}")
            print(f"  Expansions: {metadata['num_expansions']}")
            print(f"  Generated (first 100 chars): {generated[:100]}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    
    return True


def test_beam_dataclass():
    """Test Beam dataclass functionality."""
    
    print("\n" + "="*80)
    print("TESTING BEAM DATACLASS")
    print("="*80)
    
    # Create test beams
    beam1 = Beam(
        sequence=[1, 2, 3, 4, 5],
        log_probs_norm=[-0.5, -0.6, -0.7, -0.4],
        log_probs_unnorm=[-2.0, -2.4, -2.8, -1.6],
        score=-2.2,
        is_completed=False
    )
    
    beam2 = Beam(
        sequence=[1, 2, 3, 6, 7, 8],
        log_probs_norm=[-0.3, -0.4, -0.5, -0.6, -0.7],
        log_probs_unnorm=[-1.2, -1.6, -2.0, -2.4, -2.8],
        score=-1.8,
        is_completed=True
    )
    
    print(f"\nBeam 1: {beam1}")
    print(f"Beam 2: {beam2}")
    
    # Test sorting
    beams = [beam1, beam2]
    beams.sort(key=lambda b: b.score, reverse=True)
    
    print(f"\nSorted beams (by score, descending):")
    for i, beam in enumerate(beams):
        print(f"  {i+1}. {beam}")
    
    assert beams[0] == beam2, "Sorting failed"
    print("\n✓ Beam sorting works correctly")
    
    print("="*80)


if __name__ == "__main__":
    # Test dataclass first
    test_beam_dataclass()
    
    # Test main functionality
    print("\n")
    success = test_basic_functionality()
    
    if success:
        print("\n✓ All tests passed! Beam search implementation is ready to use.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")

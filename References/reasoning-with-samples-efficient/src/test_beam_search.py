#!/usr/bin/env -S uv run --python 3.12 python
"""
Quick test for BeamSearchSampling with Grok API.
Requires: Python 3.12, uv package manager
Run with: uv run --python 3.12 python test_beam_search.py
"""
import os
from dotenv import load_dotenv
from benchmark_runner import BeamSearchSampling
from openai import OpenAI

load_dotenv()

def test_beam_search():
    """Test basic beam search functionality."""
    
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: Please set XAI_API_KEY environment variable")
        return False
    
    # Initialize client
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    client.default_model = "grok-2-1212"
    
    # Create beam search strategy
    strategy = BeamSearchSampling(
        alpha=4.0,
        beam_width=3,
        tokens_per_step=50,
        length_penalty=0.6,
        proposal_temperature=1.0,
        debug=True
    )
    
    # Simple test prompt
    prompt = "What is 2 + 2? Please explain step by step."
    
    print("="*80)
    print("BEAM SEARCH TEST")
    print("="*80)
    print(f"Prompt: {prompt}\n")
    
    try:
        # Generate completion
        completion, prompt_tokens, completion_tokens = strategy.generate(
            client, prompt, max_tokens=200
        )
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Completion: {completion}\n")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {prompt_tokens + completion_tokens}")
        print(f"Expansions: {strategy.get_num_expansions()}")
        print(f"Best score: {strategy.get_best_score():.4f}")
        print("="*80)
        
        print("\n✓ Beam search test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Beam search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_beam_search()
    exit(0 if success else 1)

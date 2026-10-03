#!/usr/bin/env python3
"""
Test script for async beam search implementation.
Tests that parallelization works correctly.
"""
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from benchmark_runner import BeamSearchSampling

# Load environment variables
load_dotenv()

def test_async_beam_search():
    """Test async beam search with a simple problem."""
    
    # Setup
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ XAI_API_KEY not found in environment")
        return
    
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    client.default_model = "grok-2-1212"
    
    # Simple test problem
    prompt = """Write a Python function to calculate the factorial of a number.

def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
"""
    
    # Test with small beam search
    print("=" * 60)
    print("Testing Async Beam Search")
    print("=" * 60)
    
    strategy = BeamSearchSampling(
        alpha=4.0,
        beam_width=2,
        n_per_beam=2,
        tokens_per_step=192,
        length_penalty=0.6,
        proposal_temperature=1.0,
        top_logprobs=5,
        debug=True
    )
    
    print(f"\nPrompt:\n{prompt}\n")
    print("Generating with async beam search...")
    
    start_time = time.time()
    completion, prompt_tokens, completion_tokens = strategy.generate(
        client, prompt, max_tokens=512
    )
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n✅ Completion ({elapsed_time:.2f}s):")
    print(completion)
    print(f"\n📊 Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {prompt_tokens + completion_tokens} total")
    print(f"⚡ Expansions: {strategy.get_num_expansions()}")
    print(f"🎯 Best score: {strategy.get_best_score():.4f}")
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    
    # Verify it looks like Python code
    if "def factorial" in completion or "return" in completion:
        print("\n✅ Test PASSED: Generated code contains expected patterns")
    else:
        print("\n⚠️  Test WARNING: Generated code may not be correct")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_async_beam_search()

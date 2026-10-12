import os
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

# Model configuration
GROK_MODEL = "grok-2-1212"  # Update to "grok-beta" or other model as needed


def greedy_decode(client, prompt, max_tokens=512):
    """Greedy decoding: temperature=0 for deterministic output."""
    response = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def mcmc_sampling(
    client,
    prompt,
    max_tokens=512,
    alpha=4.0,
    mcmc_steps=10,
    block_size=192,
    proposal_temperature=1.0,
    restrict_to_last_n=None,
    debug=True,
):
    """
    MCMC power sampling with Metropolis-Hastings acceptance and block-wise generation.

    Implements sampling from target π(x) = p(x)^α using proposal q(x) = p(x).
    Based on the paper "Reasoning with Sampling" (https://arxiv.org/abs/2510.14901).

    Algorithm:
    - Target: π = p^α (power distribution)
    - Proposal: q = p (base model)
    - Block-wise generation: generate B tokens per block, refine after each
    - Partial regeneration at block boundaries
    - Accept/reject based on suffix log probabilities

    Args:
        client: OpenAI client
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        alpha: Power for target distribution p^α (default 4.0)
        mcmc_steps: Number of MCMC refinement steps per block (default 10)
        block_size: Block size B for block-wise generation (default 192)
        proposal_temperature: Temperature for proposal distribution (default 1.0)
        restrict_to_last_n: Only resample last N blocks (None = all)
        debug: Print debug info (default True)

    Returns:
        tuple: (text, log_p, log_target, acceptance_ratio)
    """
    attempts = 0
    acceptances = 0

    def extract_logprobs_with_tokens(response):
        """Extract tokens, log p, and log target."""
        if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
            return [], [], []

        tokens = [t.token for t in response.choices[0].logprobs.content]
        log_p = [t.logprob for t in response.choices[0].logprobs.content]
        log_target = [alpha * lp for lp in log_p]

        return tokens, log_p, log_target

    def sample_full(prompt, max_tokens):
        """Generate a full sample from base model."""
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        text = response.choices[0].message.content
        tokens, log_p, log_target = extract_logprobs_with_tokens(response)
        return text, tokens, log_p, log_target

    def sample_continuation(prompt, prefix, max_tokens):
        """Generate a continuation from a prefix."""
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}
            ],
            temperature=proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        text = response.choices[0].message.content
        tokens, log_p, log_target = extract_logprobs_with_tokens(response)
        return text, tokens, log_p, log_target

    # Initialize with empty generation
    tokens_cur = []
    log_p_cur = []
    log_target_cur = []

    # Calculate number of blocks to generate
    num_blocks_to_generate = max(1, max_tokens // block_size)

    if debug:
        print(f"  Block-wise generation: {num_blocks_to_generate} blocks of {block_size} tokens")

    # Generate block by block
    for block_num in range(num_blocks_to_generate):
        prefix = "".join(tokens_cur) if tokens_cur else ""

        if block_num == 0:
            # First block: use sample_full (no prefix)
            _, block_tokens, block_log_p, block_log_target = sample_full(prompt, block_size)
        else:
            # Subsequent blocks: continue from prefix
            _, block_tokens, block_log_p, block_log_target = sample_continuation(
                prompt, prefix, block_size
            )

        # Extend current state with new block
        tokens_cur.extend(block_tokens)
        log_p_cur.extend(block_log_p)
        log_target_cur.extend(block_log_target)

        if debug:
            print(f"  Block {block_num+1}/{num_blocks_to_generate}: generated {len(block_tokens)} tokens, total={len(tokens_cur)}")

        # Run MCMC refinement steps on current state
        for step in range(mcmc_steps):
            # Block-aligned index selection
            num_complete_blocks = len(tokens_cur) // block_size
            if num_complete_blocks < 2:
                if debug:
                    print(f"    Step {step+1}: Skipping, only {num_complete_blocks} complete blocks")
                continue

            attempts += 1

            # Pick random block boundary (keep at least first block)
            if restrict_to_last_n is not None:
                min_block = max(1, num_complete_blocks - restrict_to_last_n)
            else:
                min_block = 1

            if min_block > num_complete_blocks - 1:
                if debug:
                    print(f"    Step {step+1}: Skipping, restrict_to_last_n={restrict_to_last_n} too small")
                continue

            block_idx = random.randint(min_block, num_complete_blocks - 1)
            idx = block_idx * block_size

            # Prefix to keep
            prefix = "".join(tokens_cur[:idx])

            # Target length for proposal
            target_len = len(tokens_cur) - idx

            # Generate new suffix
            new_suffix, tokens_prop, log_p_prop, log_target_prop = sample_continuation(
                prompt, prefix, target_len
            )

            # Current suffix logprobs
            log_p_cur_suffix = log_p_cur[idx:]
            log_target_cur_suffix = log_target_cur[idx:]

            # MH acceptance ratio for suffixes
            log_r = (
                sum(log_target_prop) + sum(log_p_cur_suffix)
                - sum(log_target_cur_suffix) - sum(log_p_prop)
            )

            accepted = np.random.rand() < np.exp(log_r)

            if debug:
                status = "ACCEPT" if accepted else "REJECT"
                print(f"    Step {step+1}: block_idx={block_idx}, idx={idx}, log_r={log_r:.3f}, {status}")

            if accepted:
                acceptances += 1
                tokens_cur = tokens_cur[:idx] + tokens_prop
                log_p_cur = log_p_cur[:idx] + log_p_prop
                log_target_cur = log_target_cur[:idx] + log_target_prop

    # Reconstruct text from final tokens
    current_text = "".join(tokens_cur)

    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    if debug:
        print(f"  Final: {len(tokens_cur)} tokens, acceptance={acceptance_ratio:.1%}")

    return current_text, log_p_cur, log_target_cur, acceptance_ratio


def run_comparison(num_problems=3):
    """Compare greedy vs MCMC sampling on HumanEval benchmark."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: Please set XAI_API_KEY environment variable")
        print("Get your API key from: https://console.x.ai/")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )
    
    # Load HumanEval dataset
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    # Test on first few problems
    for i in range(num_problems):
        problem = dataset[i]
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        
        print(f"\n{'='*80}")
        print(f"Problem {i+1}: {task_id}")
        print('='*80)
        print(f"\nPrompt:\n{prompt[:200]}...")  # Show first 200 chars
        
        # Greedy decoding
        print("\n[GREEDY DECODING]")
        greedy_output = greedy_decode(client, prompt, max_tokens=512)
        print(greedy_output[:300] + "..." if len(greedy_output) > 300 else greedy_output)
        
        # MCMC sampling
        print("\n[MCMC SAMPLING]")
        mcmc_output, _, _, acceptance_ratio = mcmc_sampling(
            client,
            prompt,
            max_tokens=512,
            alpha=1.67,
            mcmc_steps=10,
            block_size=192,
            proposal_temperature=0.59,
            restrict_to_last_n=None,
            debug=True,
        )
        print(mcmc_output[:300] + "..." if len(mcmc_output) > 300 else mcmc_output)
        
        # Show if outputs differ
        print(f"\n[COMPARISON]")
        if greedy_output == mcmc_output:
            print("✓ Same output")
        else:
            print("✗ Different outputs")
            print(f"  Greedy length: {len(greedy_output)} chars")
            print(f"  MCMC length: {len(mcmc_output)} chars")


if __name__ == "__main__":
    run_comparison(num_problems=3)

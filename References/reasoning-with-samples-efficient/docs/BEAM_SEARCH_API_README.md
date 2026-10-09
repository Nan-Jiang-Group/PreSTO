# Beam Search Sampling with Grok API

API-based beam search implementation for power sampling, integrated with the existing benchmark framework.

## Overview

This implementation provides beam search sampling via OpenAI-compatible APIs (like Grok), following the same pattern as the MCMC sampling strategy. It maintains multiple parallel hypotheses (beams) and scores them using p^α logprobs with length normalization.

## Key Features

- **API-Based**: Works with Grok, GPT-4, or any OpenAI-compatible API
- **Power Sampling**: Scores beams using p^α where α is configurable
- **Length Normalization**: Prevents bias toward shorter sequences
- **Parallel Beams**: Maintains multiple hypotheses simultaneously
- **Configurable**: All parameters exposed via Hydra config

## Files Modified

1. **`src/benchmark_runner.py`**: Added `BeamSearchSampling` class
2. **`src/run_benchmark.py`**: Added beam search import and instantiation
3. **`src/conf/config.yaml`**: Added beam search configuration section

## Quick Start

### 1. Test Beam Search

```bash
cd src
python test_beam_search.py
```

This runs a simple test to verify beam search works with the Grok API.

### 2. Run Benchmark with Beam Search

```bash
# Enable beam search, disable others
python run_benchmark.py beam_search.enabled=true mcmc.enabled=false temperature_sampling.enabled=false greedy.enabled=false

# Run beam search with MCMC for comparison
python run_benchmark.py beam_search.enabled=true mcmc.enabled=true

# Customize beam search parameters
python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.alpha=2.0 \
    beam_search.beam_width=10 \
    beam_search.tokens_per_step=100 \
    beam_search.debug=true
```

### 3. Run on More Problems

```bash
python run_benchmark.py \
    beam_search.enabled=true \
    benchmark.num_problems=50 \
    benchmark.max_tokens=1024
```

## Configuration

All parameters are in `src/conf/config.yaml`:

```yaml
beam_search:
  enabled: false                 # Enable/disable beam search
  alpha: 4.0                     # Power factor (π(x) = p(x)^α)
  beam_width: 5                  # Number of parallel beams
  tokens_per_step: 192           # Tokens generated per expansion
  length_penalty: 0.6            # Length normalization (0.6 = Google NMT)
  proposal_temperature: 1.0      # Temperature for generation
  top_logprobs: 5                # Logprobs to retrieve from API
  debug: false                   # Print debug info
```

## Parameter Guide

### `alpha` (Power Factor)
- **Default**: 4.0
- **Range**: 1.0 - 10.0
- **Effect**: Higher α = sharper distribution, prefer high-prob sequences more
- **Examples**:
  - α=1.0: Standard sampling (no power scaling)
  - α=4.0: Balanced (paper default)
  - α=10.0: Very sharp, nearly greedy

### `beam_width` (Number of Beams)
- **Default**: 5
- **Range**: 1 - 20
- **Effect**: More beams = more exploration, slower but potentially better
- **Trade-off**: Each beam = 1 API call per expansion
- **Examples**:
  - 1: Essentially greedy (no beam search)
  - 3-5: Good balance
  - 10+: Thorough but expensive

### `tokens_per_step` (Chunk Size)
- **Default**: 192 (matching MCMC block_size)
- **Range**: 50 - 512
- **Effect**: How many tokens generated per beam per expansion
- **Trade-off**:
  - Smaller: More fine-grained control, more API calls
  - Larger: Faster generation, fewer choices
- **Examples**:
  - 50: Fine-grained (good for short answers)
  - 192: Balanced
  - 512: Coarse (good for long-form)

### `length_penalty` (Normalization)
- **Default**: 0.6 (Google NMT standard)
- **Range**: 0.0 - 1.0
- **Effect**: Prevents bias toward shorter sequences
- **Formula**: score = sum(logprobs) / length^penalty
- **Examples**:
  - 0.0: No normalization (favors very short)
  - 0.6: Mild preference for longer
  - 1.0: Full normalization (strong preference for longer)

### `proposal_temperature`
- **Default**: 1.0
- **Range**: 0.1 - 2.0
- **Effect**: Temperature for token generation
- **Note**: Different from α! This controls generation diversity, α controls scoring
- **Examples**:
  - 0.1: Nearly deterministic generation
  - 1.0: Standard sampling
  - 1.5: More diverse generation

## Algorithm Details

### Beam Search Process

1. **Initialize**: Start with empty beam
2. **Expand**: For each active beam:
   - Generate `tokens_per_step` tokens via API
   - Track tokens, logprobs (both p and p^α)
3. **Score**: Calculate length-normalized score for each beam:
   ```
   score = sum(log(p^α)) / length^length_penalty
         = α * sum(log(p)) / length^length_penalty
   ```
4. **Prune**: Keep top `beam_width` beams by score
5. **Repeat**: Until max_tokens or all beams finish (EOS)
6. **Return**: Best scoring beam

### Comparison: Beam Search vs MCMC

| Aspect | Beam Search | MCMC |
|--------|-------------|------|
| **Exploration** | Deterministic top-k | Stochastic proposals |
| **Diversity** | Lower (pruned) | Higher (random jumps) |
| **API Calls** | beam_width per expansion | 1 + mcmc_steps per block |
| **Reproducibility** | Deterministic (same inputs → same outputs) | Stochastic |
| **Best for** | Clear optimal paths | Multimodal distributions |
| **Cost** | Higher (more parallel calls) | Lower (sequential) |

### Why Both p and p^α?

The implementation tracks two logprob values:
- **`log_p`**: Logprobs from base model p(x)
- **`log_target`**: Logprobs for target distribution p^α = α * log_p

Beam scoring uses `log_target` to prefer high-probability sequences more strongly.

## Usage Examples

### Example 1: Conservative (Cheap, Fast)
```bash
python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=3 \
    beam_search.tokens_per_step=100 \
    beam_search.alpha=2.0
```
- 3 beams = 3 parallel API calls per expansion
- 100 tokens/step = fewer expansions
- α=2.0 = moderate power scaling

### Example 2: Aggressive (Expensive, Better Quality)
```bash
python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=10 \
    beam_search.tokens_per_step=50 \
    beam_search.alpha=4.0 \
    beam_search.length_penalty=0.8
```
- 10 beams = more exploration
- 50 tokens/step = fine-grained control
- α=4.0 = sharp power scaling
- 0.8 penalty = prefer longer answers

### Example 3: Comparison Mode
```bash
# Compare all strategies
python run_benchmark.py \
    greedy.enabled=true \
    temperature_sampling.enabled=true \
    mcmc.enabled=true \
    beam_search.enabled=true \
    beam_search.alpha=4.0 \
    mcmc.alpha=4.0
```

### Example 4: Debug Mode
```bash
python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.debug=true \
    beam_search.beam_width=3 \
    benchmark.num_problems=2
```
Prints detailed info:
- Number of blocks and expansions
- Active vs completed beams per block
- Best scores at each step
- Final token counts and text

## Troubleshooting

### Rate Limiting
If you hit API rate limits:
```bash
# Reduce beam width
beam_search.beam_width=3

# Or increase tokens per step
beam_search.tokens_per_step=256
```

### Poor Quality Results
Try:
```bash
# Increase exploration
beam_search.beam_width=10

# Adjust alpha
beam_search.alpha=6.0  # More aggressive power scaling

# Tune length penalty
beam_search.length_penalty=0.7  # Prefer slightly longer
```

### Timeout/Slow
```bash
# Reduce beams
beam_search.beam_width=3

# Larger chunks
beam_search.tokens_per_step=256

# Or reduce max_tokens
benchmark.max_tokens=256
```

### Empty Results
Check:
1. API key is set: `echo $XAI_API_KEY`
2. Model is available: `model.name=grok-2-1212`
3. Debug mode: `beam_search.debug=true`

## API Cost Estimation

Approximate API calls per problem:
```
num_expansions = ceil(max_tokens / tokens_per_step)
total_calls = beam_width * num_expansions
```

Examples:
- `beam_width=5, tokens_per_step=192, max_tokens=512`
  - Expansions: 512/192 ≈ 3
  - Calls: 5 * 3 = **15 API calls**

- `beam_width=10, tokens_per_step=100, max_tokens=1000`
  - Expansions: 1000/100 = 10
  - Calls: 10 * 10 = **100 API calls**

Compare to MCMC:
- `mcmc_steps=10, block_size=192, max_tokens=512`
  - Blocks: 512/192 ≈ 3
  - Calls per block: 1 + 10 = 11
  - Total: 3 * 11 = **33 API calls**

## Implementation Notes

### Code Structure

The `BeamSearchSampling` class follows the same pattern as `MCMCSampling`:

```python
class BeamSearchSampling(SamplingStrategy):
    def __init__(self, alpha, beam_width, ...):
        # Initialize parameters
        
    def _extract_logprobs_with_tokens(self, response):
        # Extract tokens and compute p^α
        
    def _sample_full(self, client, prompt, max_tokens):
        # Generate from scratch
        
    def _sample_continuation(self, client, prompt, prefix, max_tokens):
        # Continue from prefix
        
    def _calculate_beam_score(self, log_target, length):
        # Length-normalized scoring
        
    def generate(self, client, prompt, max_tokens):
        # Main beam search algorithm
```

### Key Methods

1. **`_extract_logprobs_with_tokens`**: Processes API response
   - Extracts tokens and log_p from response
   - Computes log_target = α * log_p

2. **`_sample_continuation`**: Partial generation
   - Sends prefix as assistant message
   - Model continues from there
   - Returns continuation + logprobs

3. **`_calculate_beam_score`**: Scoring
   - Sums log_target (p^α logprobs)
   - Divides by length^penalty for normalization

4. **`generate`**: Main loop
   - Maintains active_beams and completed_beams
   - Expands, scores, prunes each iteration
   - Returns best beam

## Future Enhancements

Possible improvements:
- [ ] Diversity penalty to encourage beam variety
- [ ] Early stopping when all beams converge
- [ ] Adaptive tokens_per_step based on uncertainty
- [ ] Batch API calls for efficiency
- [ ] Beam pruning based on score threshold
- [ ] Support for custom scoring functions

## References

- Wu et al. (2016): "Google's Neural Machine Translation System" - Length penalty
- Original MCMC implementation: `benchmark_runner.py` - Pattern reference
- Hydra framework: https://hydra.cc/docs/intro/

## Contact

For issues or questions, check the debug output with `beam_search.debug=true`.

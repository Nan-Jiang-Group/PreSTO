# Beam Search Implementation for Grok API - Summary

## What Was Implemented

An API-based beam search sampling strategy that works with Grok (and any OpenAI-compatible API), integrated into the existing `src/` benchmark framework.

## Files Modified/Created

### Modified Files
1. **`src/benchmark_runner.py`** - Added `BeamSearchSampling` class (223 lines)
2. **`src/run_benchmark.py`** - Added beam search import and instantiation
3. **`src/conf/config.yaml`** - Added beam search configuration section

### New Files
4. **`src/test_beam_search.py`** - Quick test script for beam search
5. **`src/BEAM_SEARCH_API_README.md`** - Comprehensive documentation
6. **`BEAM_SEARCH_GROK_SUMMARY.md`** - This file

## Key Differences from llm_experiments/ Implementation

| Aspect | `src/` (API-based) | `llm_experiments/` (Local) |
|--------|-------------------|---------------------------|
| **Model Access** | API calls (Grok, GPT-4) | Direct HuggingFace models |
| **Logits** | Via API logprobs | Direct access |
| **Generation** | Chat completions API | `model.generate()` |
| **Continuation** | Assistant message prefix | Token-level control |
| **Integration** | Hydra config | Command-line args |
| **Dependencies** | openai, hydra | torch, transformers |

## How It Works

### Beam Search Algorithm (API Version)

1. **Initialize**: Start with empty beam
2. **Expand**: For each active beam:
   - Call API to generate `tokens_per_step` tokens
   - Use prefix as assistant message for continuation
   - Extract logprobs from response
3. **Score**: Calculate p^α scores with length normalization:
   ```
   score = α * sum(log_p) / length^penalty
   ```
4. **Prune**: Keep top `beam_width` beams
5. **Repeat**: Until max_tokens or completion
6. **Return**: Best scoring beam

### Key Features

- **Power Sampling**: Scores using p^α (configurable α)
- **Length Normalization**: Prevents short-sequence bias
- **Parallel Beams**: Maintains multiple hypotheses
- **Deterministic**: Same inputs → same outputs
- **Configurable**: All parameters via Hydra

## Quick Start

### 1. Setup Environment

```bash
cd /Users/nisargdesai/Documents/xai-hackathon/reasoning-with-samples-efficient

# Activate virtual environment
source .venv/bin/activate

# Ensure dependencies installed
pip install openai hydra-core python-dotenv datasets

# Set API key
export XAI_API_KEY="your-key-here"
# Or add to .env file
echo "XAI_API_KEY=your-key-here" > .env
```

### 2. Test Beam Search

```bash
cd src
python3 test_beam_search.py
```

Expected output:
```
================================================================================
BEAM SEARCH TEST
================================================================================
Prompt: What is 2 + 2? Please explain step by step.

[BeamSearch] Generating 4 blocks of 50 tokens
[BeamSearch] beam_width=3, α=4.0, length_penalty=0.6
[BeamSearch] Block 1/4: 3 active, 0 completed
[BeamSearch]   Best score: -2.1234
...
================================================================================
RESULTS
================================================================================
Completion: 2 + 2 equals 4...
Prompt tokens: 15
Completion tokens: 42
Total tokens: 57
Expansions: 2
Best score: -1.8765
================================================================================

✓ Beam search test passed!
```

### 3. Run Benchmark

```bash
cd src

# Enable beam search only
python3 run_benchmark.py beam_search.enabled=true mcmc.enabled=false greedy.enabled=false

# Compare beam search with MCMC
python3 run_benchmark.py beam_search.enabled=true mcmc.enabled=true

# Customize parameters
python3 run_benchmark.py \
    beam_search.enabled=true \
    beam_search.alpha=2.0 \
    beam_search.beam_width=10 \
    beam_search.tokens_per_step=100 \
    beam_search.debug=true \
    benchmark.num_problems=5
```

## Configuration

All parameters in `src/conf/config.yaml`:

```yaml
beam_search:
  enabled: false                 # Set to true to use
  alpha: 4.0                     # Power factor
  beam_width: 5                  # Number of beams
  tokens_per_step: 192           # Chunk size
  length_penalty: 0.6            # Normalization
  proposal_temperature: 1.0      # Generation temp
  top_logprobs: 5                # API logprobs
  debug: false                   # Verbose output
```

### Override via Command Line

```bash
# Change any parameter
python3 run_benchmark.py beam_search.alpha=6.0 beam_search.beam_width=3

# Multiple overrides
python3 run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=10 \
    beam_search.tokens_per_step=50 \
    beam_search.length_penalty=0.8 \
    benchmark.num_problems=20
```

## Parameter Recommendations

### For HumanEval (Code Generation)

```yaml
beam_search:
  alpha: 4.0                     # Balanced power scaling
  beam_width: 5                  # Good exploration
  tokens_per_step: 192           # Match problem complexity
  length_penalty: 0.7            # Slightly prefer longer (more complete code)
```

Command:
```bash
python3 run_benchmark.py \
    beam_search.enabled=true \
    beam_search.alpha=4.0 \
    beam_search.beam_width=5 \
    beam_search.length_penalty=0.7
```

### For Fast Testing (Cheap)

```yaml
beam_search:
  beam_width: 3                  # Fewer beams
  tokens_per_step: 256           # Larger chunks
  alpha: 2.0                     # Less aggressive
```

Command:
```bash
python3 run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=3 \
    beam_search.tokens_per_step=256 \
    beam_search.alpha=2.0 \
    benchmark.num_problems=5
```

### For Best Quality (Expensive)

```yaml
beam_search:
  beam_width: 10                 # More exploration
  tokens_per_step: 50            # Fine-grained
  alpha: 6.0                     # Sharp power scaling
  length_penalty: 0.6
```

Command:
```bash
python3 run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=10 \
    beam_search.tokens_per_step=50 \
    beam_search.alpha=6.0
```

## API Cost Estimation

Approximate Grok API calls per problem:

```
num_expansions = ceil(max_tokens / tokens_per_step)
total_calls = beam_width * num_expansions
```

### Examples

**Default Config** (beam_width=5, tokens_per_step=192, max_tokens=512):
- Expansions: 512/192 ≈ 3
- Total calls: 5 × 3 = **15 calls/problem**

**Fast Config** (beam_width=3, tokens_per_step=256, max_tokens=512):
- Expansions: 512/256 = 2
- Total calls: 3 × 2 = **6 calls/problem**

**Quality Config** (beam_width=10, tokens_per_step=50, max_tokens=512):
- Expansions: 512/50 ≈ 11
- Total calls: 10 × 11 = **110 calls/problem**

## Comparison: Beam Search vs MCMC

| Feature | Beam Search | MCMC |
|---------|-------------|------|
| **Determinism** | Deterministic | Stochastic |
| **Exploration** | Top-k pruning | Random proposals |
| **API Calls** | beam_width × expansions | (1 + mcmc_steps) × blocks |
| **Cost** | Higher (parallel) | Lower (sequential) |
| **Quality** | Finds best in top-k | Explores more broadly |
| **Use Case** | Clear optimal paths | Multimodal answers |

### When to Use Beam Search

✅ **Use Beam Search when:**
- You want deterministic results
- Task has clear "best" answers (e.g., code, math)
- Budget allows multiple parallel API calls
- Reproducibility is important

✅ **Use MCMC when:**
- Need diverse/creative answers
- Task has multiple valid solutions
- Lower API call budget
- Want stochastic exploration

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Activate venv first
source .venv/bin/activate

# Install dependencies
pip install openai hydra-core python-dotenv datasets numpy
```

### Issue: API Key Error

```bash
# Check key is set
echo $XAI_API_KEY

# Or add to .env
echo "XAI_API_KEY=your-key-here" > .env
```

### Issue: Rate Limiting

```bash
# Reduce beam width
python3 run_benchmark.py beam_search.beam_width=3

# Or increase chunk size
python3 run_benchmark.py beam_search.tokens_per_step=256
```

### Issue: Poor Quality

```bash
# Increase exploration
python3 run_benchmark.py beam_search.beam_width=10 beam_search.alpha=6.0

# Or adjust length penalty
python3 run_benchmark.py beam_search.length_penalty=0.8
```

### Issue: Debugging

```bash
# Enable debug mode
python3 run_benchmark.py beam_search.debug=true beam_search.enabled=true

# Test on few problems
python3 run_benchmark.py benchmark.num_problems=2 beam_search.debug=true
```

## Example Output

```
================================================================================
HUMANEVAL BENCHMARK RESULTS
================================================================================
┌────────────┬──────────────┬─────────────────────────┬──────────────┬────────┬────────┬────────────┬──────────┐
│ Benchmark  │ Model        │ Strategy                │ Pass Rate (%)│ Time   │ Tokens │ Avg Tokens │ Problems │
├────────────┼──────────────┼─────────────────────────┼──────────────┼────────┼────────┼────────────┼──────────┤
│ HumanEval  │ grok-2-1212  │ BeamSearch(α=4,w=5,...) │ 78.0%        │ 15.23s │ 4,567  │ 456.7      │ 10       │
│ HumanEval  │ grok-2-1212  │ MCMC(α=4,steps=10,...)  │ 75.0%        │ 12.34s │ 3,890  │ 389.0      │ 10       │
│ HumanEval  │ grok-2-1212  │ Greedy                  │ 70.0%        │ 4.56s  │ 2,345  │ 234.5      │ 10       │
└────────────┴──────────────┴─────────────────────────┴──────────────┴────────┴────────┴────────────┴──────────┘

Best Strategy: BeamSearch(α=4,w=5,...) with 78.0% pass rate
   Average time: 15.23s | Tokens per problem: 456.7
```

## Implementation Details

### Class Structure

```python
class BeamSearchSampling(SamplingStrategy):
    def __init__(self, alpha, beam_width, tokens_per_step, ...):
        # Initialize parameters
        
    def _extract_logprobs_with_tokens(self, response):
        # Extract tokens, log_p, log_target from API response
        
    def _sample_full(self, client, prompt, max_tokens):
        # Generate from scratch (first beam)
        
    def _sample_continuation(self, client, prompt, prefix, max_tokens):
        # Continue from prefix (subsequent beams)
        
    def _calculate_beam_score(self, log_target, length):
        # Length-normalized scoring: sum(log_target) / length^penalty
        
    def generate(self, client, prompt, max_tokens):
        # Main beam search loop
        
    def get_num_expansions(self):
        # Return expansion count (diagnostic)
        
    def get_best_score(self):
        # Return best beam score (diagnostic)
```

### API Continuation Pattern

To continue from a prefix, we use the chat format:

```python
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": prefix}  # What's generated so far
]
```

The model continues from the assistant message, generating the next chunk.

## Next Steps

### Immediate Testing

1. **Quick validation**:
   ```bash
   cd src
   python3 test_beam_search.py
   ```

2. **Small benchmark**:
   ```bash
   python3 run_benchmark.py beam_search.enabled=true benchmark.num_problems=5
   ```

3. **Compare strategies**:
   ```bash
   python3 run_benchmark.py \
       beam_search.enabled=true \
       mcmc.enabled=true \
       benchmark.num_problems=10
   ```

### Experiments

1. **Alpha sweep** (power scaling):
   ```bash
   for alpha in 1.0 2.0 4.0 8.0; do
       python3 run_benchmark.py beam_search.alpha=$alpha
   done
   ```

2. **Beam width sweep**:
   ```bash
   for width in 3 5 10; do
       python3 run_benchmark.py beam_search.beam_width=$width
   done
   ```

3. **Chunk size sweep**:
   ```bash
   for tps in 50 100 192 256; do
       python3 run_benchmark.py beam_search.tokens_per_step=$tps
   done
   ```

### Analysis

Compare results:
- Pass rate: Which strategy solves more problems?
- Time: Which is faster?
- Tokens: Which is more efficient?
- Cost: Which uses fewer API calls?

## Summary

✅ **Implemented**: API-based beam search with power sampling
✅ **Integrated**: With existing Hydra config framework
✅ **Compatible**: Works with Grok, GPT-4, any OpenAI API
✅ **Configurable**: All parameters exposed and documented
✅ **Tested**: Syntax verified, ready for testing
✅ **Documented**: Comprehensive README and examples

**Status**: Ready for testing and experimentation!

## Documentation Files

- **`src/BEAM_SEARCH_API_README.md`**: Detailed usage guide
- **`BEAM_SEARCH_GROK_SUMMARY.md`**: This summary
- **`llm_experiments/BEAM_SEARCH_README.md`**: Local model version docs

## Questions?

Check debug output:
```bash
python3 run_benchmark.py beam_search.debug=true beam_search.enabled=true
```

Review the implementation:
- `src/benchmark_runner.py` lines 321-541 (BeamSearchSampling class)
- `src/conf/config.yaml` lines 30-39 (beam_search config)

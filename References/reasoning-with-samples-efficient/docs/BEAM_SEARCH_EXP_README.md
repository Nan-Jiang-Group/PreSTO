# Beam Search with Power Sampling

Implementation of beam search for sampling from power-scaled distributions p^α, where α = 1/temperature.

## Overview

This implementation combines beam search with power sampling to efficiently explore high-probability paths in the distribution p^α. Unlike MCMC which uses random proposals and accept/reject, beam search maintains multiple hypotheses and deterministically selects the best paths.

## Key Features

- **Power Sampling**: Samples from p^α instead of p for better reasoning performance
- **Chunked Generation**: Generates `tokens_per_step` tokens at a time (configurable)
- **Length Normalization**: Uses length penalty to avoid bias toward shorter sequences
- **Multiple Beams**: Maintains `beam_width` parallel hypotheses
- **EOS Handling**: Properly terminates beams at end-of-sequence tokens
- **Metadata Tracking**: Returns expansion count, scores, and completion stats

## Files

- `beam_search_utils.py`: Core beam search implementation
- `beam_search_math.py`: Benchmark script for MATH dataset
- `test_beam_search.py`: Basic functionality tests
- `scripts/beam_search_math.sh`: Experiment runner script

## Quick Start

### 1. Test the Implementation

```bash
cd /Users/nisargdesai/Documents/xai-hackathon/reasoning-with-samples-efficient
python llm_experiments/test_beam_search.py
```

### 2. Run on MATH Benchmark

```bash
# Single run with default parameters
python llm_experiments/beam_search_math.py \
    --model qwen_math \
    --temperature 0.25 \
    --beam_width 5 \
    --tokens_per_step 16 \
    --batch_idx 0

# Compare with MCMC
python llm_experiments/beam_search_math.py \
    --model qwen_math \
    --temperature 0.25 \
    --beam_width 5 \
    --tokens_per_step 16 \
    --batch_idx 0 \
    --compare_mcmc

# Run full experiment suite
bash llm_experiments/scripts/beam_search_math.sh
```

## Parameters

### `beam_search_power_samp()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | AutoregressiveSampler | - | Model sampler instance |
| `context` | list[int] | - | Initial token IDs (prefix) |
| `temp` | float | - | Temperature for p^α (α = 1/temp) |
| `beam_width` | int | 5 | Number of parallel beams |
| `tokens_per_step` | int | 16 | Tokens generated per expansion |
| `max_new_tokens` | int | 3072 | Maximum tokens to generate |
| `length_penalty` | float | 0.6 | Length normalization exponent |
| `verbose` | bool | False | Print debug information |

### Returns

```python
(
    best_sequence,      # list[int]: Token IDs of best beam
    log_probs_norm,     # list[float]: Proposal distribution log probs
    log_probs_unnorm,   # list[float]: Target p^α log probs
    metadata            # dict: Statistics and info
)
```

**Metadata fields:**
- `num_expansions`: Number of beam expansion steps
- `final_beam_scores`: Scores of top beams
- `num_completed_beams`: Number of beams that hit EOS
- `final_score`: Score of the returned best beam

## Algorithm Details

### Beam Search Process

1. **Initialize**: Start with context as single beam
2. **Expand**: Generate `tokens_per_step` tokens for each active beam
3. **Score**: Calculate length-normalized score using p^α log probs
4. **Prune**: Keep top `beam_width` beams by score
5. **Repeat**: Until beams complete or max tokens reached

### Scoring Formula

```
score = sum(log_probs_unnorm) / (length ^ length_penalty)
```

Where:
- `log_probs_unnorm` are from target distribution p^α
- `length` is number of generated tokens
- `length_penalty` ∈ [0, 1] controls length normalization

### Two Log Probability Tracks

1. **`log_probs_norm`**: From proposal distribution q (temperature-scaled)
   - Used by sampling process
   - Tracks what was actually sampled from

2. **`log_probs_unnorm`**: From target distribution p^α
   - Used for beam scoring
   - What we want to optimize for

### Length Penalty

Common values:
- `0.6`: Google NMT (Wu et al. 2016) - mild penalty
- `0.8`: Conservative - moderate penalty
- `1.0`: Full normalization - strong penalty

Lower values favor shorter sequences, higher favor longer.

## Comparison: Beam Search vs MCMC

| Aspect | Beam Search | MCMC Power Sampling |
|--------|-------------|---------------------|
| **Determinism** | Deterministic | Stochastic |
| **Exploration** | Systematic top-k | Random proposals |
| **Diversity** | Lower (pruned) | Higher (random jumps) |
| **Efficiency** | No wasted generations | Accept/reject overhead |
| **Memory** | k sequences in parallel | Single + proposal |
| **Best for** | Clear optimal paths | Multimodal distributions |

## Example Usage

### Basic Usage

```python
from power_samp_utils import load_model_and_tokenizer
from beam_search_utils import beam_search_power_samp

# Load model
model, tokenizer, sampler = load_model_and_tokenizer("qwen_math", "cuda")

# Prepare prompt
text = "What is 2 + 2?"
prefix = tokenizer.encode(text, return_tensors="pt")[0].tolist()

# Run beam search
output, lp_norm, lp_unnorm, metadata = beam_search_power_samp(
    sampler,
    prefix,
    temp=0.25,        # α = 4
    beam_width=5,
    tokens_per_step=16,
    max_new_tokens=512,
    verbose=True
)

# Decode result
generated = tokenizer.decode(output[len(prefix):])
print(f"Generated: {generated}")
print(f"Score: {metadata['final_score']:.4f}")
print(f"Expansions: {metadata['num_expansions']}")
```

### Greedy Variant

```python
from beam_search_utils import beam_search_greedy

# Greedy beam search (α → ∞)
output, _, _, metadata = beam_search_greedy(
    sampler,
    prefix,
    beam_width=5,
    tokens_per_step=16,
    max_new_tokens=512
)
```

## Hyperparameter Tuning

### Beam Width
- **Small (1-3)**: Fast, less exploration
- **Medium (5-10)**: Good balance
- **Large (>10)**: Thorough but expensive

### Tokens Per Step
- **Small (1-8)**: More fine-grained control, slower
- **Medium (16-32)**: Good balance
- **Large (>32)**: Faster but coarser

### Temperature (α = 1/temp)
- **Low temp (0.1)**: α=10, very sharp p^10
- **Medium temp (0.25)**: α=4, balanced
- **High temp (0.5+)**: α≤2, closer to p

### Length Penalty
- **0.5-0.7**: Prefer concise answers
- **0.8-0.9**: Balanced
- **1.0**: Strongly prefer longer answers

## Experiments

Run parameter sweeps:

```bash
# Beam width sweep
for BW in 3 5 10; do
    python llm_experiments/beam_search_math.py --beam_width $BW
done

# Tokens per step sweep
for TPS in 8 16 32; do
    python llm_experiments/beam_search_math.py --tokens_per_step $TPS
done

# Temperature sweep
for TEMP in 0.1 0.25 0.5; do
    python llm_experiments/beam_search_math.py --temperature $TEMP
done
```

## Results Location

Results are saved to:
```
results/{model}/beam_search/{model}_math_beam_search_w{beam_width}_tps{tokens_per_step}_t{temp}_b{batch}_s{seed}.csv
```

## Troubleshooting

### Out of Memory
- Reduce `beam_width`
- Reduce `tokens_per_step`
- Use smaller model

### Slow Generation
- Increase `tokens_per_step` (coarser but faster)
- Reduce `beam_width`
- Reduce `max_new_tokens`

### Poor Quality
- Increase `beam_width` for more exploration
- Adjust `temperature` (try 0.25)
- Tune `length_penalty` for answer length

### Beams Not Completing
- Increase `max_new_tokens`
- Check EOS token configuration
- Adjust `length_penalty` (favor longer)

## References

- Wu et al. (2016): "Google's Neural Machine Translation System" - Length penalty
- Welleck et al. (2024): "Reasoning with Monte Carlo Tree Search" - Power sampling
- Original implementation: `power_samp_utils.py` (MCMC variant)

## Future Enhancements

- [ ] Diversity penalty to encourage beam variety
- [ ] Early stopping with completion threshold
- [ ] Adaptive tokens_per_step based on uncertainty
- [ ] Best-of-N sampling for comparison
- [ ] Parallel batch processing
- [ ] Score visualization and beam tracking

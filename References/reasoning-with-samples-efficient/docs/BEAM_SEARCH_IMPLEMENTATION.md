# Beam Search Implementation - Summary

## What Was Implemented

A complete beam search implementation for power sampling from p^α distributions, designed to work with the existing power sampling framework in `llm_experiments/`.

## Files Created

### Core Implementation
1. **`llm_experiments/beam_search_utils.py`** (11.3 KB)
   - `Beam` dataclass for hypothesis tracking
   - `beam_search_power_samp()` - main beam search function
   - `beam_search_greedy()` - greedy variant
   - `calculate_beam_score()` - scoring helper
   - `print_beam_info()` - debugging utility

### Benchmark Scripts
2. **`llm_experiments/beam_search_math.py`** (11.1 KB)
   - Complete benchmark script for MATH dataset
   - Compares: standard sampling, naive_temp, beam search
   - Optional MCMC comparison
   - Comprehensive result tracking and statistics

### Testing & Utilities
3. **`llm_experiments/test_beam_search.py`** (5.6 KB)
   - Basic functionality tests
   - Dataclass verification
   - Quick sanity checks

4. **`llm_experiments/scripts/beam_search_math.sh`** (2.4 KB)
   - Automated experiment runner
   - Parameter sweeps (beam_width, tokens_per_step, temperature)
   - Comparison experiments

5. **`llm_experiments/scripts/quick_test.sh`** (1.4 KB)
   - Quick validation script
   - Tests basic functionality on small parameters

### Documentation
6. **`llm_experiments/BEAM_SEARCH_README.md`** (7.8 KB)
   - Complete usage guide
   - Parameter explanations
   - Examples and troubleshooting
   - Comparison with MCMC

7. **`BEAM_SEARCH_IMPLEMENTATION.md`** (this file)
   - Implementation summary
   - Quick reference

## Key Features

### Algorithm
- **Chunked Generation**: Generates `tokens_per_step` tokens per expansion
- **Power Sampling**: Scores beams using p^α (target distribution)
- **Length Normalization**: Prevents bias toward shorter sequences
- **Parallel Hypotheses**: Maintains `beam_width` beams simultaneously
- **EOS Handling**: Properly terminates at end-of-sequence tokens

### Configuration
```python
beam_search_power_samp(
    p,                      # AutoregressiveSampler
    context,                # list[int] - initial tokens
    temp=0.25,              # α = 1/temp = 4
    beam_width=5,           # 5 parallel beams
    tokens_per_step=16,     # Generate 16 tokens per step
    max_new_tokens=3072,    # Max total tokens
    length_penalty=0.6,     # Length normalization
    verbose=False           # Debug output
)
```

### Returns
- Best sequence (token IDs)
- Log probabilities (both proposal and target)
- Metadata (expansions, scores, completion stats)

## Quick Start

### 1. Run Basic Tests
```bash
cd /Users/nisargdesai/Documents/xai-hackathon/reasoning-with-samples-efficient

# Test basic functionality
python llm_experiments/test_beam_search.py

# Quick test on 1 problem
bash llm_experiments/scripts/quick_test.sh
```

### 2. Run MATH Benchmark
```bash
# Single run
python llm_experiments/beam_search_math.py \
    --model qwen_math \
    --temperature 0.25 \
    --beam_width 5 \
    --tokens_per_step 16 \
    --batch_idx 0

# With MCMC comparison
python llm_experiments/beam_search_math.py \
    --model qwen_math \
    --beam_width 5 \
    --compare_mcmc

# Full experiment suite
bash llm_experiments/scripts/beam_search_math.sh
```

### 3. Use in Code
```python
from power_samp_utils import load_model_and_tokenizer
from beam_search_utils import beam_search_power_samp

# Load model
model, tokenizer, sampler = load_model_and_tokenizer("qwen_math", "cuda")

# Prepare input
text = "What is 2 + 2?"
prefix = tokenizer.encode(text)[0].tolist()

# Run beam search
output, _, _, metadata = beam_search_power_samp(
    sampler,
    prefix,
    temp=0.25,
    beam_width=5,
    tokens_per_step=16,
    max_new_tokens=512
)

# Decode
result = tokenizer.decode(output[len(prefix):])
print(f"Answer: {result}")
print(f"Score: {metadata['final_score']:.4f}")
```

## Design Decisions

### 1. Chunked Generation (`tokens_per_step`)
- **Why**: Reduces number of expansions, leverages existing `naive_temp()`
- **Trade-off**: Larger chunks = faster but less fine-grained control
- **Typical values**: 8-32 tokens

### 2. Power Sampling Integration
- **Target**: p^α for scoring (α = 1/temp)
- **Proposal**: Temperature-scaled for generation
- **Tracks both**: `log_probs_unnorm` (p^α), `log_probs_norm` (q)

### 3. Length Penalty
- **Formula**: score = sum(log_probs) / length^penalty
- **Purpose**: Prevent bias toward short sequences
- **Values**: 0.6 (Google NMT), 0.8 (conservative), 1.0 (full)

### 4. Beam Pruning
- **Separate**: Completed beams (EOS) vs active beams
- **Keep top-k**: In each category independently
- **Sort by**: Length-normalized score

## Comparison: Beam Search vs MCMC

| Feature | Beam Search | MCMC |
|---------|-------------|------|
| **Approach** | Deterministic top-k | Stochastic proposals |
| **Diversity** | Lower (pruned) | Higher (random) |
| **Efficiency** | All generations useful | Accept/reject overhead |
| **Memory** | k sequences | Single + proposal |
| **Reproducibility** | Same inputs → same outputs | Stochastic |
| **Best for** | Clear optimal paths | Multimodal distributions |

## Testing Strategy

### Unit Tests (`test_beam_search.py`)
- Beam dataclass functionality
- Basic beam search on simple problem
- Greedy variant
- Different beam widths

### Integration Tests (`beam_search_math.py`)
- Full MATH benchmark
- Comparison with standard sampling
- Comparison with naive temperature sampling
- Optional MCMC comparison

### Parameter Sweeps (`beam_search_math.sh`)
- Beam width: 3, 5, 10
- Tokens per step: 8, 16, 32
- Temperature: 0.1, 0.25, 0.5

## Results Location

```
results/{model}/beam_search/
    {model}_math_beam_search_w{beam_width}_tps{tokens_per_step}_t{temp}_b{batch}_s{seed}.csv
```

Each CSV contains:
- Problem details
- Standard/naive/beam/MCMC completions
- Parsed answers
- Correctness flags
- Beam metadata (expansions, scores)

## Hyperparameter Recommendations

### For MATH Benchmark
- **Temperature**: 0.25 (α=4) - balanced power scaling
- **Beam width**: 5 - good exploration/speed trade-off
- **Tokens per step**: 16 - efficient chunking
- **Length penalty**: 0.6 - standard Google NMT value

### For Experimentation
- **Fast testing**: beam_width=3, tokens_per_step=8
- **Best quality**: beam_width=10, tokens_per_step=16
- **Memory constrained**: beam_width=3, tokens_per_step=32

## Next Steps

### Immediate
1. Run `test_beam_search.py` to verify installation
2. Try `quick_test.sh` for end-to-end validation
3. Run small batch: `--batch_idx 0 --beam_width 3`
4. Compare with MCMC: `--compare_mcmc`

### Experiments
1. Beam width sweep to find optimal value
2. Temperature sweep for best α
3. Compare beam search vs MCMC on accuracy/speed
4. Analyze beam scores vs correctness

### Potential Enhancements
- [ ] Diversity penalty for varied beams
- [ ] Early stopping with score threshold
- [ ] Adaptive tokens_per_step based on confidence
- [ ] Parallel batch processing
- [ ] Visualization of beam exploration
- [ ] Best-of-N baseline comparison

## Troubleshooting

### Import Errors
- Ensure you're in the correct directory
- Check all files created successfully
- Verify `power_samp_utils.py` exists

### OOM (Out of Memory)
- Reduce `beam_width` (try 3)
- Reduce `tokens_per_step` (try 8)
- Use smaller model

### Slow Generation
- Increase `tokens_per_step` (try 32)
- Reduce `beam_width`
- Reduce `max_new_tokens`

### Poor Results
- Tune `temperature` (try 0.25)
- Increase `beam_width` (try 10)
- Adjust `length_penalty` (try 0.8)

## Implementation Notes

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Error handling with try/except
- Progress bars for long operations
- Verbose mode for debugging

### Compatibility
- Works with existing `power_samp_utils.py`
- Uses same `AutoregressiveSampler` interface
- Compatible with all models in `MODEL_REGISTRY`
- Follows same structure as `mcmc_power_samp()`

### Testing Coverage
- Basic functionality tests
- Edge cases (EOS, max_tokens)
- Different parameter combinations
- Comparison with baselines

## References

### Papers
- Wu et al. (2016): "Google's Neural Machine Translation System"
- Welleck et al. (2024): "Reasoning with Monte Carlo Tree Search"

### Original Code
- `power_samp_utils.py`: MCMC power sampling
- `naive_temp()`: Temperature-scaled generation

## Contact & Support

For issues or questions:
1. Check `BEAM_SEARCH_README.md` for detailed documentation
2. Run `test_beam_search.py` to verify installation
3. Review error messages in verbose mode
4. Compare with MCMC baseline using `--compare_mcmc`

## Summary

✅ Complete beam search implementation for power sampling
✅ Integrated with existing power sampling framework
✅ Comprehensive testing and documentation
✅ Ready for MATH benchmark experiments
✅ Flexible parameter configuration
✅ Comparison tools for MCMC validation

**Status**: Ready for testing and experimentation!

# Complete Beam Search Implementation Guide

**Version**: 1.0  
**Last Updated**: December 2025

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Beam Search Implementation](#beam-search-implementation)
4. [Async Parallelization](#async-parallelization)
5. [Running Experiments](#running-experiments)
6. [Configuration Reference](#configuration-reference)
7. [Architecture & Implementation](#architecture--implementation)
8. [Benchmarks](#benchmarks)

---

## Quick Start

### Setup

```bash
cd /path/to/project/reasoning-with-samples-efficient

# Set your Grok API key
export XAI_API_KEY="your-key-here"
# Or add to .env file
echo "XAI_API_KEY=your-key-here" > .env
```

### Run Beam Search

```bash
cd src

# Run benchmark with beam search
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    mcmc.enabled=false \
    greedy.enabled=false \
    benchmark.num_problems=10
```

### Compare All Strategies

```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    mcmc.enabled=true \
    benchmark.num_problems=10
```

---

## Project Overview

A flexible framework for comparing different sampling strategies (Greedy, MCMC, Beam Search, Temperature) across various benchmarks (HumanEval, SWE-bench, GSM8K).

### Key Features

- **Multiple Sampling Strategies**: Greedy, MCMC, Beam Search, Temperature
- **Multiple Benchmarks**: HumanEval (code), SWE-bench (bug fixing), GSM8K (math)
- **Beam Search**: Proper branching with `n_per_beam` parameter
  - **Async Parallelization**: 2-10× speedup depending on beam width
- **Extensible**: Easy to add new benchmarks and strategies
- **Comprehensive Metrics**: Pass rate, time, tokens, cost

### File Structure

```
src/
├── benchmark_runner.py          # Core framework (strategies, benchmarks)
├── run_benchmark.py             # Main CLI (Hydra-based)
├── conf/config.yaml             # Configuration defaults
├── gsm8k_benchmark.py           # GSM8K benchmark implementation
├── swebench_benchmark.py        # SWE-bench implementation
├── beam_search_evals.sh         # Parallel experiment script
├── aggregate_results.py         # Result aggregation tool
├── test_beam_search.py          # Basic test script
├── test_async_syntax.py         # Async validation
└── eval_logs/                   # Experiment outputs (auto-created)
```

---

## Beam Search Implementation

Traditional beam search generates one continuation per beam, then prunes. **Our beam search** generates multiple (`n_per_beam`) continuations per beam, exploring more paths before pruning.

### `n` Beam Continuations

The OpenAI-compatible API (including Grok) supports generating multiple samples in one call:

```python
response = client.chat.completions.create(
    ...,
    n=5,  # Generate 5 different samples in ONE API call
)

# Access all samples with their individual logprobs
for choice in response.choices:
    text = choice.message.content
    logprobs = choice.logprobs.content  # Each has its own logprobs
```

**Benefits**:

- Multiple samples from one API call (efficient)
- Each sample has complete logprobs
- All samples share prompt cost (only counted once)
- Enables true beam branching

### Scoring Formula

**Power Sampling**: Target distribution π(x) = p(x)^α where α > 1 concentrates on high-probability sequences.

```python
# Per-token log probabilities
log_target = [alpha * log_p for log_p in log_probs]

# Length-normalized beam score
score = sum(log_target) / (length ** length_penalty)
```

**Length penalty values:**

- `0.0` → No normalization (favors short sequences)
- `0.6` → Moderate (Google NMT default)
- `0.8` → Stronger normalization
- `1.0` → Full normalization (average log probability)

### Usage

```bash
# Enable with defaults (beam_width=2, n_per_beam=2)
uv run --python 3.12 python run_benchmark.py beam_search.enabled=true

# Customize parameters
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=5 \
    beam_search.n_per_beam=5 \
    beam_search.alpha=4.0 \
    beam_search.tokens_per_step=192 \
    beam_search.length_penalty=0.6

# Debug mode
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.debug=true \
    benchmark.num_problems=2
```

---

## Async Parallelization

### Overview

Async parallelization provides **2-10× speedup** by running all beam expansions in parallel instead of sequentially.

### Performance Comparison

| Configuration | Sequential Time | Parallel Time | Speedup |
|--------------|-----------------|---------------|---------|
| `beam_width=2` | 4s | 2s | **2×** |
| `beam_width=5` | 10s | 2s | **5×** |
| `beam_width=10` | 20s | 2s | **10×** |

*Assumes 2s per API call*

### How It Works

**Before (Sequential):**
```python
for beam in active_beams:  # Sequential loop
    continuations = self._sample_continuation_multiple(client, ...)
    # Each API call waits for the previous to complete
# Total time: beam_width × API_latency
```

**After (Parallel):**
```python
# All beams expanded in parallel!
candidate_beams = await self._expand_beams_parallel(client, active_beams, ...)
# Total time: max(API_latency) ≈ 1× API_latency
```

### Implementation

The async implementation uses Python's `asyncio` with `AsyncOpenAI`:

```python
async def _expand_beams_parallel(self, client, active_beams, prompt, block_num):
    """Parallelize beam expansion."""
    tasks = []
    
    # Create async tasks for all beams
    for beam in active_beams:
        task = self._sample_continuation_multiple_async(...)
        tasks.append(task)
    
    # Run ALL API calls in parallel
    results = await asyncio.gather(*tasks)
    
    # Process results into candidate beams
    return candidate_beams, total_pt, total_ct
```

### Resource Management

Uses async context manager to prevent resource leaks:

```python
async def _run_with_client(self, api_key, base_url, model, prompt, max_tokens):
    """Proper client lifecycle management."""
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as async_client:
        async_client.default_model = model
        return await self._generate_async(async_client, prompt, max_tokens)
    # Client automatically closed here - no resource leaks!
```

### Backward Compatibility

**100% backward compatible** - the async implementation is transparent:

```python
# Your existing code works unchanged!
strategy = BeamSearchSampling(beam_width=5, n_per_beam=5)
completion, pt, ct = strategy.generate(client, prompt, max_tokens)
# Automatically uses async parallelization internally
```

---

## Running Experiments

### Parallel Experiment Script

The `beam_search_evals.sh` script runs multiple configurations in parallel and aggregates results.

#### Features

- **Auto-detects OS** and installs GNU parallel if needed
- **Runs multiple experiments** simultaneously (max 4 concurrent)
- **Tests different configurations** automatically
- **Aggregates results** with best configuration identification
- **Saves individual logs** for detailed analysis

#### Quick Start

```bash
cd src
./beam_search_evals.sh
```

#### Default Experiments

| # | beam_width | n_per_beam | alpha | length_penalty | Description |
|---|-----------|------------|-------|----------------|-------------|
| 1 | 2 | 2 | 4.0 | 0.6 | Baseline (Google NMT) |
| 2 | 2 | 2 | 4.0 | 0.8 | Stronger normalization |
| 3 | 2 | 2 | 4.0 | 1.0 | Full normalization |
| 4 | 3 | 2 | 4.0 | 0.6 | Larger beam width |
| 5 | 2 | 3 | 4.0 | 0.6 | More continuations/beam |
| 6 | 5 | 3 | 4.0 | 0.6 | Large configuration |

#### Output Structure

**Experiment Logs:**
```
eval_logs/
├── exp_beam_width=2_n_per_beam=2_alpha=4.0_length_penalty=0.6.log
├── exp_beam_width=2_n_per_beam=2_alpha=4.0_length_penalty=0.8.log
└── ...
```

**Aggregated Results:**
```
==================================================================================================================
AGGREGATED BENCHMARK RESULTS FROM ALL EXPERIMENTS
==================================================================================================================

+----------------------------------------------+--------------------------------+-----------+--------------+
| Experiment                                   | Strategy                       | Pass Rate | Avg Time (s) |
+==============================================+================================+===========+==============+
| beam_width=2_n_per_beam=2_alpha=4.0_lp=0.6  | BeamSearch(α=4.0,w=2,n=2)     | 58.0%     | 1.67         |
+----------------------------------------------+--------------------------------+-----------+--------------+
| beam_width=2_n_per_beam=2_alpha=4.0_lp=0.8  | BeamSearch(α=4.0,w=2,n=2)     | 62.0%     | 1.71         |
+----------------------------------------------+--------------------------------+-----------+--------------+
...

🏆 Best Beam Search Configuration:
   Strategy:   BeamSearch(α=4.0,width=2,n=2,lp=0.8)
   Pass Rate:  62.0%
   Avg Time:   1.71s
   Total Cost: $0.0037

📊 Greedy Baseline: 40.0%
📈 Improvement: +22.0%
```

#### Manual Result Aggregation

```bash
cd src
python aggregate_results.py eval_logs
# or
uv run --python 3.12 python aggregate_results.py eval_logs
```

#### Customization

Edit the script to add/modify experiments:

```bash
# Edit params.txt in beam_search_evals.sh
cat > params.txt << 'EOF'
beam_width=2 n_per_beam=2 alpha=4.0 length_penalty=0.6
beam_width=10 n_per_beam=5 alpha=8.0 length_penalty=1.0  # Add your own!
EOF
```

---

## Configuration Reference

### Configuration File

`src/conf/config.yaml` contains all defaults, overridable via command line.

### Model Configuration

```yaml
model:
  name: grok-4-1-fast-non-reasoning  # or grok-2-1212, grok-beta
  base_url: https://api.x.ai/v1
  provider: xai  # or ollama, vllm
```

### Benchmark Configuration

```yaml
benchmark:
  name: humaneval  # or gsm8k, swebench, swebench-verified
  num_problems: 10
  max_tokens: 128  # Max tokens per completion
```

### Beam Search Configuration

```yaml
beam_search:
  enabled: false
  alpha: 4.0                    # Power factor (π(x) = p(x)^α)
  beam_width: 2                 # Number of beams to keep
  n_per_beam: 2                 # Continuations per beam (TRUE beam search!)
  tokens_per_step: 192          # Tokens per expansion
  length_penalty: 0.6           # Length normalization (0.0-1.0)
  proposal_temperature: 1.0     # Temperature for generation
  top_logprobs: 5               # Number of logprobs to retrieve
  debug: false                  # Verbose output
```

### MCMC Configuration

```yaml
mcmc:
  enabled: true
  alpha: 1.67                   # Power factor
  steps: 10                     # Refinement steps
  block_size: 32                # Block size for generation
  proposal_temperature: 0.59    # Should be alpha^(-1)
  restrict_to_last_n: null      # Only resample last N blocks (null = all)
  top_logprobs: 5
  debug: true
```

### Parameter Guidelines

#### `alpha` (Power Factor)
- **Range**: 1.0 - 10.0
- **Default**: 4.0
- **Effect**: Higher = prefer high-probability sequences more
- **MCMC uses**: 1.67 (from paper)

#### `beam_width` (Number of Beams)
- **Range**: 1 - 20
- **Default**: 2
- **Effect**: More beams = better exploration, higher cost
- **Recommendation**: 2-5 for most cases

#### `n_per_beam` (Continuations per Beam)
- **Range**: 1 - 10
- **Default**: 2
- **Effect**: Controls branching factor
  - `n=1`: Pseudo-beam search
  - `n=2-3`: Good balance
  - `n=5-10`: Maximum exploration
- **Total candidates**: `beam_width × n_per_beam`

#### `tokens_per_step` (Chunk Size)
- **Range**: 32 - 512
- **Default**: 192
- **Effect**: Smaller = more expansions, finer-grained
- **Recommendation**: 128-256 for most cases
- **⚠️ WARNING**: Values too small (e.g., 16) cause fragmentation

#### `length_penalty` (Normalization)
- **Range**: 0.0 - 1.0
- **Default**: 0.6 (Google NMT)
- **Effect**:
  - `0.0`: No normalization (favors short)
  - `0.6`: Moderate normalization
  - `1.0`: Full normalization (average log prob)

### Command Line Overrides

```bash
# Override single parameters
uv run --python 3.12 python run_benchmark.py \
    beam_search.alpha=6.0 \
    beam_search.beam_width=3 \
    benchmark.num_problems=20

# Multiple overrides
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=5 \
    beam_search.n_per_beam=5 \
    beam_search.tokens_per_step=256 \
    beam_search.length_penalty=0.8 \
    benchmark.name=gsm8k \
    benchmark.num_problems=30 \
    benchmark.max_tokens=512
```

### Hydra Multirun (Batch Experiments)

```bash
# Compare multiple benchmarks
uv run --python 3.12 python run_benchmark.py -m benchmark.name=humaneval,gsm8k

# Sweep parameters
uv run --python 3.12 python run_benchmark.py -m beam_search.alpha=2.0,4.0,6.0

# Cartesian product
uv run --python 3.12 python run_benchmark.py -m \
    beam_search.beam_width=2,5 \
    beam_search.n_per_beam=2,5
# Runs 4 experiments: (2,2), (2,5), (5,2), (5,5)
```

---

## Architecture & Implementation

### Core Components

#### 1. Benchmark (Abstract Base Class)

```python
class Benchmark(ABC):
    @abstractmethod
    def load_dataset(self): pass
    
    @abstractmethod
    def format_prompt(self, problem: Dict) -> str: pass
    
    @abstractmethod
    def extract_completion(self, response: str, problem: Dict) -> str: pass
    
    @abstractmethod
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]: pass
```

#### 2. SamplingStrategy (Abstract Base Class)

```python
class SamplingStrategy(ABC):
    @abstractmethod
    def generate(self, client, prompt, max_tokens) -> tuple[str, int, int]:
        """Returns (completion, prompt_tokens, completion_tokens)"""
        pass
```

#### 3. BenchmarkRunner

Orchestrates benchmarking:
- Works with any benchmark implementation
- Tracks comprehensive metrics
- Handles errors gracefully
- Saves prediction files for official evaluation

### BeamSearchSampling Implementation

**Key Methods:**

```python
class BeamSearchSampling(SamplingStrategy):
    def generate(self, client, prompt, max_tokens):
        """Main entry point - uses async internally"""
        return asyncio.run(self._run_with_client(...))
    
    async def _generate_async(self, client, prompt, max_tokens):
        """Main beam search algorithm"""
        for block_num in range(num_blocks):
            # Parallel expansion
            candidate_beams = await self._expand_beams_parallel(...)
            # Score all candidates
            scored_beams = [(self._calculate_beam_score(...), beam) for beam in candidate_beams]
            # Prune to top beam_width
            active_beams = top_k(scored_beams, self.beam_width)
        return best_beam
    
    async def _expand_beams_parallel(self, client, active_beams, prompt, block_num):
        """Parallel beam expansion"""
        tasks = [self._sample_continuation_multiple_async(...) for beam in active_beams]
        results = await asyncio.gather(*tasks)
        return candidate_beams
    
    async def _sample_continuation_multiple_async(self, client, prompt, prefix, max_tokens, n):
        """Generate n continuations from a beam"""
        response = await client.chat.completions.create(..., n=n)
        return [(text, tokens, log_p, log_target, finished) for choice in response.choices]
```

### Data Flow

```
User Request
    ↓
run_benchmark.py (Hydra config)
    ↓
BenchmarkRunner.run_benchmark()
    ↓
For each problem:
    ├─ Format prompt (Benchmark.format_prompt)
    ├─ Generate completion (Strategy.generate)
    │   └─ BeamSearch: async parallel expansion
    ├─ Extract answer (Benchmark.extract_completion)
    └─ Check correctness (Benchmark.check_correctness)
    ↓
Aggregate metrics
    ↓
Display results table
    ↓
Save predictions (for official evaluation)
```

---

## Benchmarks

### HumanEval (Code Completion)

**Description**: 164 Python programming problems requiring function-level code completion.

**Usage**:
```bash
uv run --python 3.12 python run_benchmark.py \
    benchmark.name=humaneval \
    benchmark.num_problems=10 \
    benchmark.max_tokens=512
```

**Evaluation**: Saves to `predictions/humaneval_*.jsonl`, evaluate with:
```bash
evaluate_functional_correctness predictions/humaneval_*.jsonl
```

### GSM8K (Grade School Math)

**Description**: 8.5K high-quality grade school math problems.

**Format**: Problems require step-by-step reasoning with final answer after `####` marker.

**Usage**:
```bash
uv run --python 3.12 python run_benchmark.py \
    benchmark.name=gsm8k \
    benchmark.num_problems=30 \
    benchmark.max_tokens=512
```

**Answer Extraction**: Looks for `#### [number]` or last number in response.

### SWE-bench (Bug Fixing)

**Description**: Real GitHub issues from popular Python repositories.

**Variants**:
- `swebench`: SWE-bench Lite (300 problems)
- `swebench-verified`: Verified subset (500 problems)

**Usage**:
```bash
uv run --python 3.12 python run_benchmark.py \
    benchmark.name=swebench \
    benchmark.num_problems=5 \
    benchmark.max_tokens=2048
```

**Note**: Uses heuristic evaluation (check SWEBENCH_USAGE.md for official evaluation).

### Adding New Benchmarks

1. Create benchmark class:
```python
from benchmark_runner import Benchmark

class MyBenchmark(Benchmark):
    def name(self) -> str:
        return "MyBenchmark"
    
    def load_dataset(self):
        self.dataset = load_dataset(...)
    
    # Implement other abstract methods...
```

2. Register in `run_benchmark.py`:
```python
BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    "gsm8k": GSM8KBenchmark,
    "mybenchmark": MyBenchmark,  # Add here
}
```

3. Use it:
```bash
uv run --python 3.12 python run_benchmark.py benchmark.name=mybenchmark
```

---

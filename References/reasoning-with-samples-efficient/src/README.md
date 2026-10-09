# LLM Sampling Strategy Benchmark

A flexible framework for comparing different sampling strategies across various benchmarks.

## Features

- **Multiple Sampling Strategies**: Greedy, MCMC, Temperature-based sampling
- **Multiple Benchmarks**: HumanEval (code completion), SWE-bench (repository-level bug fixing)
- **Extensible**: Easy to add new benchmarks (MATH template provided)
- **Comprehensive Metrics**: Pass rate, time, token usage, and more
- **Beautiful Output**: Formatted tables and visualizations

## Quick Start

### 1. Setup Environment

```bash
# Make sure your API key is in .env
echo "XAI_API_KEY=your-api-key-here" > .env

# Install dependencies (if not already done)
uv pip install -e ".[dev]"
```

### 2. Run Benchmark

```bash
# Compare all strategies on HumanEval (first 10 problems)
python src/run_benchmark.py

# Test on SWE-bench (repository-level bug fixing)
python src/run_benchmark.py benchmark.name=swebench benchmark.num_problems=5

# With custom options (Hydra overrides)
python src/run_benchmark.py \
  benchmark.name=humaneval \
  model.name=grok-2-1212 \
  benchmark.num_problems=20 \
  mcmc.alpha=4.0 \
  mcmc.steps=10

# Batch comparisons with Hydra multirun
python src/run_benchmark.py -m benchmark.name=humaneval,swebench
python src/run_benchmark.py -m mcmc.alpha=1.0,1.67,4.0
```

### 3. Model Providers

The benchmark supports three providers: X.AI (cloud), Ollama (local), and vLLM (local with GPU batching).

#### X.AI (Default)
```bash
# Set API key
echo "XAI_API_KEY=your-api-key" > .env

# Run benchmark
python src/run_benchmark.py model.provider=xai
```

#### Ollama (Local, Easy Setup)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull smollm2:1.7b

# Run benchmark
python src/run_benchmark.py model.provider=ollama
```

Note: Ollama ignores the `n` parameter, so parallel sampling uses separate API calls.

#### vLLM (Local, True GPU Batching)
```bash
# Install vLLM
pip install vllm

# Start vLLM server
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct --port 8000

# Run benchmark (in another terminal)
python src/run_benchmark.py model.provider=vllm
```

vLLM provides true GPU batching with the `n` parameter, making beam search and parallel MCMC much faster.

**vLLM serve options:**
```bash
# Basic
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct --port 8000

# With options
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096

# Multi-GPU
vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2
```

| Feature | X.AI | Ollama | vLLM |
|---------|------|--------|------|
| Logprobs | ⚠️ Only at temp=0 | ✓ | ✓ |
| `n` parameter | ✓ | ✗ (ignored) | ✓ |
| GPU batching | N/A | ✗ (sequential) | ✓ |
| Setup | API key | Easy | Requires GPU |

### 4. View Results

The script outputs:
- A comparison table with all metrics
- Pass rate comparison bar chart
- Time efficiency rankings
- Token usage analysis

Example output:
```
==================================================
HUMANEVAL BENCHMARK RESULTS
==================================================
+-----------+--------------+-------------+-----------+------+--------+---------+
| Benchmark | Model        | Strategy    | Pass Rate | Time | Tokens | Cost    |
+-----------+--------------+-------------+-----------+------+--------+---------+
| HumanEval | grok-2-1212  | MCMC(...)   | 75.0%     | 12s  | 1,234  | $0.0074 |
| HumanEval | grok-2-1212  | Greedy      | 70.0%     | 5s   | 567    | $0.0034 |
| HumanEval | grok-2-1212  | Temperature | 65.0%     | 6s   | 678    | $0.0041 |
+-----------+--------------+-------------+-----------+------+--------+---------+
```

## Architecture

### Core Components

1. **`Benchmark` (Abstract Base Class)**: Defines interface for benchmarks
   - `load_dataset()`: Load the benchmark data
   - `format_prompt()`: Format problems for LLM
   - `extract_completion()`: Parse LLM responses
   - `check_correctness()`: Verify solutions

2. **`SamplingStrategy` (Abstract Base Class)**: Defines sampling methods
   - `generate()`: Generate completion with specific strategy

3. **`BenchmarkRunner`**: Orchestrates benchmarking
   - Works with any benchmark implementation
   - Tracks metrics (pass rate, time, tokens)
   - Handles errors gracefully

### File Structure

```
src/
├── benchmark_runner.py      # Core framework (Benchmark, Runner, Strategies)
├── benchmark_template.py    # Examples for adding new benchmarks
├── run_benchmark.py         # Main CLI script (Hydra-based)
├── conf/
│   └── config.yaml          # Hydra configuration defaults
├── llm_wrapper.py          # Custom LLM wrapper (optional)
├── test_sampling.py        # Simple test script
└── README.md               # This file
```

## Adding New Benchmarks

### Step 1: Implement Benchmark Class

Create a new class that inherits from `Benchmark`:

```python
from benchmark_runner import Benchmark

class MyBenchmark(Benchmark):
    def name(self) -> str:
        return "MyBenchmark"

    def load_dataset(self):
        # Load your dataset
        pass

    def get_problem(self, index: int) -> Dict:
        # Return problem at index
        pass

    def get_num_problems(self) -> int:
        # Return total problems
        pass

    def format_prompt(self, problem: Dict) -> str:
        # Convert problem to LLM prompt
        pass

    def extract_completion(self, response: str, problem: Dict) -> str:
        # Extract answer from LLM response
        pass

    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        # Verify correctness
        return True, "correct"
```

### Step 2: Register Benchmark

Add to `BENCHMARK_REGISTRY` in `run_benchmark.py`:

```python
BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    "mybenchmark": MyBenchmark,  # Add here
}
```

### Step 3: Use It

```bash
python src/run_benchmark.py benchmark.name=mybenchmark
```

See `benchmark_template.py` for complete examples (SWEBench, MATH).

## Available Strategies

### 1. Greedy Sampling
- Temperature = 0 (deterministic)
- Always picks most likely token
- Fast, consistent, but may miss creative solutions

### 2. MCMC Sampling
- Uses Metropolis-Hastings accept/reject with partial regeneration
- Block-wise generation: generates B tokens per block, refines after each
- Explores alternative solutions by regenerating suffixes from block boundaries
- Configurable via Hydra: `mcmc.steps`, `mcmc.alpha`, `mcmc.block_size`, `mcmc.restrict_to_last_n`

## MCMC Power Sampling: Implementation Notes

The MCMC sampling in `src/` implements power sampling from the paper ["Reasoning with Sampling"](https://arxiv.org/abs/2510.14901).

### Algorithm

**Target distribution:** π(x) = p(x)^α where α > 1 concentrates on high-probability sequences.

**Proposal distribution:** q(x) = p(x) (base model at temperature=1)

**Metropolis-Hastings acceptance:**
```
log A = log(π(x')/π(x)) + log(q(x)/q(x'))
      = α·log p(x') - α·log p(x) + log p(x) - log p(x')
      = (α - 1) · [log p(x') - log p(x)]
```

For **α=1.67** (default): proposals with higher log probability are more likely to be accepted.

### What's Implemented

| Feature | Paper | `src/` | Notes |
|---------|-------|--------|-------|
| **Partial regeneration** | ✓ | ✓ | Pick random block boundary, regenerate suffix |
| **MH acceptance** | ✓ | ✓ | Accept/reject based on suffix log probs |
| **Block-wise generation** | B=192 tokens | ✓ | Configurable via `mcmc.block_size` |
| **Restrict to last N blocks** | - | ✓ | Cost optimization via `mcmc.restrict_to_last_n` |

Partial regeneration works by sending the prefix as an assistant message and letting the model continue.

### Usage

```bash
# src/ version (API-based, Hydra config)
python src/run_benchmark.py mcmc.enabled=true mcmc.steps=10 mcmc.alpha=4.0

# Only run MCMC (disable other strategies)
python src/run_benchmark.py greedy.enabled=false temperature_sampling.enabled=false

# Experiment version (local models, full algorithm with B=192)
cd llm_experiments/
python power_samp_math.py --model qwen_math --mcmc_steps 10
```

For research reproducing the paper's results, use the experiment code with local models.

### 3. Temperature Sampling
- Standard stochastic sampling
- Configurable via Hydra: `temperature_sampling.temperature`
- Higher T = more diverse outputs

## Adding New Strategies

Inherit from `SamplingStrategy`:

```python
class MyStrategy(SamplingStrategy):
    def __init__(self):
        super().__init__("MyStrategy")

    def generate(self, client, prompt, max_tokens):
        # Your sampling logic here
        response = client.chat.completions.create(...)
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
```

## Configuration (Hydra)

The benchmark uses [Hydra](https://hydra.cc/) for configuration. See `src/conf/config.yaml` for defaults.

### Key Config Options

```yaml
# Model
model.name: grok-2-1212          # LLM model name
model.base_url: https://api.x.ai/v1

# Benchmark
benchmark.name: humaneval        # Benchmark to use
benchmark.num_problems: 10       # Number of problems to test
benchmark.max_tokens: 512        # Max tokens per completion

# MCMC Sampling
mcmc.enabled: true               # Enable MCMC strategy
mcmc.alpha: 1.67                 # Power factor for target distribution
mcmc.steps: 10                   # MCMC refinement steps per block
mcmc.block_size: 192             # Block size B for block-wise generation
mcmc.restrict_to_last_n: null    # Only resample last N blocks (null = all)
mcmc.proposal_temperature: 0.59  # Temperature for proposal distribution
mcmc.debug: false                # Print debug info during MCMC

# Temperature Sampling
temperature_sampling.enabled: true
temperature_sampling.temperature: 0.8

# Greedy Sampling
greedy.enabled: true
```

## Available Benchmarks

### HumanEval (Code Completion)
- 164 Python programming problems
- Function-level code completion
- Default tokens: 512
```bash
python src/run_benchmark.py benchmark.name=humaneval
```

### SWE-bench Lite (Bug Fixing)
- 300 real GitHub issues from popular Python repos
- Repository-level bug fixing and feature addition
- Default tokens: 2048
- Uses heuristic evaluation (see SWEBENCH_USAGE.md)
```bash
python src/run_benchmark.py benchmark.name=swebench benchmark.num_problems=5
```

### SWE-bench Verified
- 500 human-validated issues
- Higher quality, verified problems
```bash
python src/run_benchmark.py benchmark.name=swebench-verified benchmark.num_problems=5
```

## Configuration (Hydra)

The benchmark uses [Hydra](https://hydra.cc/) for configuration. See `src/conf/config.yaml` for defaults.

### Key Config Options

```yaml
# Model
model.name: grok-2-1212          # LLM model name
model.base_url: https://api.x.ai/v1

# Benchmark
benchmark.name: humaneval        # Benchmark to use (humaneval, swebench, swebench-verified)
benchmark.num_problems: 10       # Number of problems to test
benchmark.max_tokens: 3072       # Max tokens per completion

# MCMC Sampling
mcmc.enabled: true               # Enable MCMC strategy
mcmc.alpha: 1.67                 # Power factor for target distribution
mcmc.steps: 10                   # MCMC refinement steps per block
mcmc.block_size: 192             # Block size B for block-wise generation
mcmc.restrict_to_last_n: null    # Only resample last N blocks (null = all)
mcmc.proposal_temperature: 0.59  # Temperature for proposal distribution
mcmc.debug: false                # Print debug info during MCMC

# Temperature Sampling
temperature_sampling.enabled: true
temperature_sampling.temperature: 0.8

# Greedy Sampling
greedy.enabled: true
```

### Override via CLI

```bash
python src/run_benchmark.py mcmc.alpha=4.0 benchmark.num_problems=20
```

### Batch Runs with Hydra Multirun

Use the `-m` flag to run multiple configurations:

```bash
# Compare benchmarks
python src/run_benchmark.py -m benchmark.name=humaneval,swebench

# Sweep MCMC parameters
python src/run_benchmark.py -m mcmc.alpha=1.0,1.67,4.0

# Multiple dimensions
python src/run_benchmark.py -m benchmark.name=humaneval,swebench mcmc.alpha=1.67,4.0
```

## Metrics Tracked

- **Pass Rate (%)**: Percentage of problems solved correctly
- **Avg Time (s)**: Average time per problem
- **Total Tokens**: Total tokens used across all problems
- **Avg Tokens/Problem**: Average tokens per problem
- **Total Cost ($)**: Total API cost in USD
- **Cost/Problem ($)**: Average cost per problem in USD

## Examples

### Compare strategies on first 5 problems
```bash
python src/run_benchmark.py benchmark.num_problems=5
```

### Test only MCMC with higher alpha
```bash
python src/run_benchmark.py \
  greedy.enabled=false \
  temperature_sampling.enabled=false \
  mcmc.alpha=4.0 \
  mcmc.steps=5
```

### Use different model
```bash
python src/run_benchmark.py model.name=grok-beta
```

## Extending the Framework

The framework is designed to be easily extensible:

1. **New Benchmarks**: See `benchmark_template.py` for SWEBench/MATH examples
2. **New Strategies**: Inherit from `SamplingStrategy`
3. **New Metrics**: Extend `BenchmarkMetrics` dataclass
4. **New Models**: Just change `model.name` config (works with any OpenAI-compatible API)

## Troubleshooting

**ImportError**: Make sure all dependencies are installed
```bash
uv pip install -e ".[dev]"
```

**API Key Error**: Set your API key in `.env`
```bash
echo "XAI_API_KEY=your-key" > .env
```

**Execution Errors**: Some benchmarks (like HumanEval) execute code. This is sandboxed but may fail on complex problems.

## License

MIT

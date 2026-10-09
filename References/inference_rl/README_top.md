# Inference RL - Reasoning with Sampling

This repository contains experiments on reasoning with sampling for language models.

## Installation Instructions

### Prerequisites

- Conda/Miniconda installed
- CUDA-capable GPU (recommended for running LLM experiments)
- Git

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd inference_rl
```

## commands
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## vLLM Pass@k Inference

For efficient batched pass@k evaluation using vLLM:

### Run vLLM Pass@k Inference
```bash
cd reasoning-with-sampling/llm_experiments

# Generate k=16 samples per problem with vLLM
python vllm_passk_math.py \
    --model qwen_math \
    --k 16 \
    --temperature 0.8 \
    --save_str results/

# Options:
# --model: qwen, qwen_math, phi, tulu, qwen_math_grpo, phi_grpo
# --k: number of samples per problem (default: 16)
# --temperature: sampling temperature (default: 0.8)
```

### Evaluate Pass@k Results
```bash
# Evaluate and show pass@k curve
python eval_vllm_passk.py results/qwen_math/qwen_math_vllm_passk_16_temp_0.8_seed_0.csv --plot

# Just show numerical results
python eval_vllm_passk.py results/qwen_math/qwen_math_vllm_passk_16_temp_0.8_seed_0.csv
```

This approach generates k samples per problem in a single run using vLLM's efficient batching, unlike the original implementation which requires multiple seed runs.

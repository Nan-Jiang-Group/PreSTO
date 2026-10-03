# Agent Guide

Repo root: `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening`

> Use absolute paths in commands and references throughout this repo.

## Project Overview

Inference for **sharpened LLM distributions** (`p^α`) through temperature
annealing, using Metropolis-Hastings (MH) / MCMC samplers. The same sampling
idea is implemented against three serving backends — HuggingFace
`transformers`, vLLM, and SGLang — and evaluated on reasoning/coding
benchmarks (MATH500, HumanEval, MBPP, GPQA, LiveCodeBench, AIME, etc.).

See `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/README.md`
for the long-form project description and the list of related reference repos.

## Repository Structure

```
src/
  pyproject.toml         # single `power-sharpening` distribution
  power_sharpening/      # the installable package
    common/              #   temperature schedules, proposal tree, sampling stats
    backends/            #   hf/ vllm/ sglang/: wrapper.py + samplers/ per backend
                         #   (backends/vllm/engine_patch/ vendors patched vLLM internals)
    tasks/               #   Benchmark interfaces: MATHBenchmark, constants.MODEL_MAP, grading
    config/              #   dataset.yaml + algorithms.yaml + config_loader / model_context
    runners/             #   CLI entry points: python -m power_sharpening.runners.<backend>.<runner>
  tests/                 # pytest suite (testpaths=src/tests, pythonpath=src in pytest.ini)
experiments/             # Launch scripts: env.sh + clusters/ + <backend>/<benchmark>/*.sh
algorithms/replica-to-reason/  # PowerSMC / Power-SMC replica baselines (gitignored sibling project)
data/                    # Benchmark datasets (JSON/JSONL files)
examples/                # Runnable demos (prefetch DP)
case_studies/            # Figure scripts & notebooks
result/                  # Experiment outputs & statistical analyses (date-stamped)
References/              # Vendored upstream reference implementations (read-only; do not edit)
```

Everything ships as one `power-sharpening` distribution
(`src/pyproject.toml`); the benchmark wrapper is the `power_sharpening.tasks`
subpackage, backends are optional extras.

## Setup

```bash
# from the repository root — uv (preferred; manages src/.venv + src/uv.lock)
uv sync --project src                # core + HF backend + dev tools
uv sync --project src --extra vllm   # + vLLM backend (needs vllm>=0.18.0)
uv sync --project src --extra sglang # + SGLang backend (needs sglang>=0.5.0)
uv run --project src <cmd>           # run inside the env without activating

# or plain pip
pip install -e ./src           # core + benchmark tasks
pip install -e './src[vllm]'   # + vLLM backend
pip install -e './src[sglang]' # + SGLang backend
pip install -e './src[all]'    # everything
```

Base deps (`torch`, `transformers`, `numpy`, `pyyaml`, `pylatexenc`, …) live
in `src/pyproject.toml`; `matplotlib` and `pytest` are in its `dev`
dependency group. vLLM and SGLang are only required for their respective
backends. Manage deps with `uv add/remove --project src` (use `--optional
<extra>` or `--dev` for the groups), not by hand-editing.

## Running Experiments

Launch scripts live under `experiments/<backend>/<benchmark>/` and call the
installed runner modules (`python -m power_sharpening.runners.<backend>.<runner>`).
They need a GPU, so launch them through Slurm:

```bash
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  python -m power_sharpening.runners.hf.run_power_mh \
    --dataset math500 --algorithm power_mcmc --save_str /tmp/out
```

Outputs are written under `result/` (often date-stamped subdirectories).

## Testing

The main test is the batched-vs-scalar sampler equivalence check:
`src/tests/test_low_temp_sampler.py`.

- It imports the installed `power_sharpening` package (`pip install -e './src[hf]'`
  first); `pytest` picks up `src/` automatically via `pythonpath` in
  `pytest.ini`.
- It loads a real model (`qwen-math-small` via `power_sharpening.tasks.constants.MODEL_MAP`)
  in **float32 + eager attention** and uses **greedy** decoding. The batched
  (left-padded) and scalar paths are only exactly equivalent in fp32; bf16 +
  flex_attention + `torch.compile` perturbs borderline argmaxes and the paths
  diverge. Do **not** "optimize" the test to bf16. Needs a GPU.

`examples/prefetch_dp_requests.py` is a small CPU-only demo of the prefetch
dynamic program, useful for sanity-checking without loading a full model.

## Conventions

- `References/` is vendored upstream code kept for comparison — treat as
  read-only; do not modify or "fix" it.
- New samplers should reuse the backend's wrapper
  (`power_sharpening/backends/<backend>/wrapper.py`) and the shared
  `power_sharpening.common` utilities (`temp_scheduler`, `sample_stats`,
  `proposal_tree`) rather than re-implementing them.
- `power_sharpening/backends/vllm/engine_patch/` mirrors vLLM internals
  (pinned to `vllm>=0.18.0`) — change it only when deliberately tracking a
  vLLM version bump.

---

# Markdown Guidelines

- Use **absolute paths** instead of relative paths.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- For tasks requiring a GPU, use the following command: `srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python ...`.

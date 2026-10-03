# Agent Guide

Repo root: `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening`

Use absolute paths in commands, code references, and Markdown.

## Overview

PreSTO: Metropolis–Hastings (MH) samplers for power-sharpened LLM distributions (`p^α`), with predictive subtree prefetching. Implemented for HuggingFace, vLLM, and SGLang; evaluated on MATH500, AIME, MBPP, GPQA, HumanEval, MMLU, and LiveCodeBench. See `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/README.md` for usage.

## Layout

| Path | Purpose |
| --- | --- |
| `src/power_sharpening/common/` | Temperature schedules, proposal tree, prefetch ranking, sampling stats. |
| `src/power_sharpening/backends/{hf,vllm,sglang}/` | `wrapper.py` + `samplers/` per backend. `vllm/engine_patch/` vendors patched vLLM internals. |
| `src/power_sharpening/tasks/` | Benchmarks, `constants.MODEL_MAP`, grading. |
| `src/power_sharpening/config/` | `algorithms.yaml`, `dataset.yaml`, config loader. |
| `src/power_sharpening/runners/` | CLI entry points: `python -m power_sharpening.runners.<backend>.<runner>`. |
| `src/tests/` | pytest suite (`pytest.ini` sets `testpaths` and `pythonpath`). |
| `experiments/` | Launch scripts: `env.sh`, `clusters/`, `<backend>/<benchmark>/*.sh`. |
| `case_studies/` | Launchers (`scripts/`), log parsers (`extract/`), figures (`draw/`), dated logs. |
| `data/`, `result/`, `examples/` | Benchmark inputs; outputs and analyses; CPU-only demos. |
| `website/` | Project website source. |
| `References/` | Vendored upstream code. **Read-only.** |

One `power-sharpening` distribution (`src/pyproject.toml`); vLLM and SGLang are optional extras.

## Setup

```bash
uv sync --project src                # core + HF + dev tools
uv sync --project src --extra vllm   # + vLLM (pinned in pyproject)
uv sync --project src --extra sglang # + SGLang
uv run --project src <cmd>           # run without activating
```

Pip alternative: `pip install -e './src[vllm]'` (extras: `vllm`, `sglang`, `all`). Change dependencies with `uv add/remove --project src` (`--optional <extra>` or `--dev`), not by hand-editing.

## Running

Runs need a GPU; launch through Slurm:

```bash
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  python -m power_sharpening.runners.hf.run_power_mh \
    --dataset math500 --algorithm power_mcmc --save_str /tmp/out
```

## Testing

- Run `pytest` from the repo root with the package installed.
- `src/tests/test_low_temp_sampler.py` needs a GPU. It loads a real model in **float32 + eager attention** with **greedy** decoding; batched and scalar paths match only in fp32. Do **not** switch it to bf16.
- `examples/prefetch_dp_requests.py` is a CPU-only sanity check of the prefetch DP.

## Conventions

- Never edit `References/`.
- New samplers reuse the backend `wrapper.py` and the shared `power_sharpening.common` utilities (`temp_scheduler`, `sample_stats`, `proposal_tree`).
- Edit `backends/vllm/engine_patch/` only when deliberately tracking a vLLM version bump.

## Python Style

- Start every file with a docstring explaining how to run it.
- Read existing code first, reuse existing modules, and match the surrounding style.

# PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling

Metropolis–Hastings (MH) samplers for power-sharpened LLM distributions, `p(x)^α`, on HuggingFace, vLLM, and SGLang backends. Evaluated on MATH500, AIME, MBPP, GPQA, HumanEval, MMLU, and LiveCodeBench.

**Subtree prefetching** batches proposals for the possible future MH states. Each MH step has an accept and a reject branch, so the sampler precomputes a tree of continuations, follows the realized decisions, and discards the rest. This trades speculative computation for fewer sequential model calls and leaves the MH transition rule unchanged.

[Paper](https://arxiv.org/pdf/2610.13171) · [Website](https://nan-jiang-group.github.io/PreSTO/)

## Install

Requires Python ≥ 3.10. Everything ships as one `power-sharpening` distribution (`src/pyproject.toml`); vLLM and SGLang are optional extras.

```bash
export REPO_ROOT=$(pwd)   # the checkout

uv sync --project "$REPO_ROOT/src"                 # core + HF + dev tools
uv sync --project "$REPO_ROOT/src" --extra vllm    # + vLLM (pinned 0.27.1), recommended
uv sync --project "$REPO_ROOT/src" --extra sglang  # + SGLang
```

Plain pip: `python -m pip install -e "$REPO_ROOT/src[vllm]"` (extras: `vllm`, `sglang`, `all`).

Change dependencies with `uv add/remove --project src` (`--optional <extra>` or `--dev`), not by hand-editing `pyproject.toml`.

## Samplers

Runners are `python -m power_sharpening.runners.<backend>.<module>`.

| Method | HF / vLLM module | SGLang module |
| --- | --- | --- |
| PowerMH | `run_power_mh` | `run_power_sample_mh` |
| Subtree-prefetching MH | `run_subtree_prefetching_mh` | `run_subtree_prefetching_mh` |
| Multiple-Try MH (MTM) | `run_multi_try_mh` | — |
| Subtree-prefetching MTM | `run_subtree_prefetching_multi_try_mh` | — |

- HF and vLLM support **uniform** and **entropy** suffix cuts (with the cut-probability correction in the acceptance ratio). SGLang uses `run_entropy_cut_mh` (top-K approximation).
- vLLM also has best-of-N and Power-SMC runners.
- Most runners take `--dataset` and `--override key=value`. HF `run_power_mh` and the SGLang PowerMH/subtree runners use `--task` plus explicit flags.
- **MTM:** `run_multi_try_mh --algorithm multi_try` draws `num_tries` candidates per step. The prefetched variant counts `prefetch_budget` in suffix requests and needs `prefetch_budget >= num_tries`.

## Configuration

Settings merge in this order (later wins): `default` in `algorithms.yaml` → algorithm block → `defaults` in `dataset.yaml` → dataset block → `--override`. Files are in `src/power_sharpening/config/`.

| Setting | Meaning |
| --- | --- |
| `alpha` | Sharpening exponent. |
| `temperature` | Proposal temperature; `-1` means `1/alpha`. |
| `mcmc_steps`, `num_blocks` | MH steps per block; number of blocks. |
| `prefetch_budget` | Proposal budget for subtree methods. |
| `rank_fn` | Expansion priority (default `longest_path_first`; see `common/prefetch_rank.py`). |
| `cut_dist_type` | `uniform` or `entropy`. |
| `cut_power` / `cut_dist_param` | Entropy-cut exponent (PowerMH / subtree MH). |
| `print_tree` | Log proposal trees (needed for tree figures). |
| `max_samples`, `max_new_tokens` | Sample and generation limits. |

Also: `--model_str` (model alias), `--save_str` (output dir), `--run_name` (output prefix).

## Models and benchmarks

- Model aliases: `MODEL_MAP` in `src/power_sharpening/tasks/constants.py` (e.g. `qwen3-4b`, `qwen3-8b`, `qwen3.5-4b`, `gemma-12b-it`).
- Datasets: `math500`, `aime`, `mbpp`, `gpqa`, `human_eval`, `mmlu`, `lcb_v6`. Data lives in `data/`.

## Quick start

Runs need a GPU; launch through Slurm:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  "$REPO_ROOT/src/.venv/bin/python" \
  -m power_sharpening.runners.vllm.run_subtree_prefetching_mh \
  --dataset math500 --algorithm subtree_prefetching_mh --model_str qwen3-4b \
  --save_str /tmp/subtree-prefetch --run_name math500-smoke \
  --override alpha=4.0 temperature=-1 mcmc_steps=10 num_blocks=1 \
  prefetch_budget=20 rank_fn=bfs_accept_first max_samples=2 max_new_tokens=512
```



## Directory layout

```
.
├── src/
│   ├── power_sharpening/
│   │   ├── common/      # temperature schedules, proposal tree, prefetch ranking, sampling stats
│   │   ├── backends/    # hf/, vllm/, sglang/ (wrapper.py + samplers/); vllm/engine_patch/ vendors patched vLLM internals
│   │   ├── tasks/       # benchmarks, MODEL_MAP, grading
│   │   ├── config/      # algorithms.yaml, dataset.yaml, loader
│   │   └── runners/     # CLI entry points per backend
│   ├── tests/           # pytest suite (src/pytest.ini sets testpaths and pythonpath)
│   └── examples/        # CPU-only prefetch DP demo
├── case_studies/        # env.sh, cluster submitters, launchers, log parsers, figure code, dated logs
├── data/                # benchmark inputs
├── References/          # third-party reference code (read-only)
└── website/             # project website source
```

## Contributing

- Never edit `References/`.
- New samplers reuse the backend `wrapper.py` and the shared `power_sharpening.common` utilities (`temp_scheduler`, `sample_stats`, `proposal_tree`).
- Edit `backends/vllm/engine_patch/` only when deliberately tracking a vLLM version bump.
- Start every Python file with a docstring explaining how to run it.
- Read existing code first, reuse existing modules, and match the surrounding style.

# PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling

Metropolis–Hastings (MH) samplers for power-sharpened LLM distributions, `p(x)^α`, on HuggingFace, vLLM, and SGLang backends.

**Subtree prefetching** batches proposals for the possible future MH states. Each MH step has an accept and a reject branch, so the sampler precomputes a tree of continuations, follows the realized decisions, and discards the rest. This trades speculative computation for fewer sequential model calls and leaves the MH transition rule unchanged.

[Paper](https://arxiv.org/xxxx) · [Website](https://step-tree-explorer.jiangnanhugo.chatgpt.site/) · Website source: `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/website`

## Install

Requires Python ≥ 3.10.

```bash
export REPO_ROOT=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening

uv sync --project "$REPO_ROOT/src"                # HF backend, tasks, pytest
uv sync --project "$REPO_ROOT/src" --extra vllm   # + vLLM (pinned 0.27.1)
uv sync --project "$REPO_ROOT/src" --extra sglang # + SGLang (>=0.5.0)
```

Plain pip: `python -m pip install -e "$REPO_ROOT/src[vllm]"` (extras: `vllm`, `sglang`, `all`).

## Samplers

Runners are `python -m power_sharpening.runners.<backend>.<module>`.

| Method | HF / vLLM module | SGLang module |
| --- | --- | --- |
| PowerMH | `run_power_mh` | `run_power_sample_mh` |
| Subtree-prefetching MH | `run_subtree_prefetching_mh` | `run_subtree_prefetching_mh` |
| Multiple-Try MH (MTM) | `run_multi_try_mh` | — |
| Subtree-prefetching MTM | `run_subtree_prefetching_multi_try_mh` | — |

- HF and vLLM support **uniform** and **entropy** suffix cuts (with the cut-probability correction in the acceptance ratio). SGLang uses `run_entropy_cut_mh` (top-K approximation).
- vLLM also has best-of-N, SMC, and Calderhead multiple-proposal MH runners.
- Most runners take `--dataset` and `--override key=value`. HF `run_power_mh` and the SGLang PowerMH/subtree runners use `--task` plus explicit flags.

## Configuration

Settings merge in this order (later wins): `default` in `algorithms.yaml` → algorithm block → `defaults` in `dataset.yaml` → dataset block → `--override`. Files are in `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/config`.

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

## Quick start

GPU runs go through Slurm:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  "$REPO_ROOT/src/.venv/bin/python" \
  -m power_sharpening.runners.vllm.run_subtree_prefetching_mh \
  --dataset math500 --algorithm subtree_prefetching_mh --model_str qwen3-4b \
  --save_str /tmp/subtree-prefetch --run_name math500-smoke \
  --override alpha=4.0 temperature=-1 mcmc_steps=10 num_blocks=1 \
  prefetch_budget=20 rank_fn=bfs_accept_first max_samples=2 max_new_tokens=512
```



**MTM:** `run_multi_try_mh --algorithm multi_try` with `num_tries` candidates per step. The prefetched variant counts `prefetch_budget` in suffix requests and needs `prefetch_budget >= num_tries`.

## Models and benchmarks

- Model aliases: `MODEL_MAP` in `src/power_sharpening/tasks/constants.py` (e.g. `qwen3-4b`, `qwen3-8b`, `qwen3.5-4b`, `gemma-12b-it`).
- Datasets: `math500`, `aime`, `mbpp`, `gpqa`, `human_eval`, `mmlu`, `lcb_v6`. Data lives in `data/`.

## Case studies and figures

Launchers are in `case_studies/scripts/` (see its README). Each has `.interactive.sh`, `.punakha.sbatch.sh`, and `.tacc.sbatch.sh` entry points; pass `DATASETS=`, `MODELS=`, `RANKS=`, `BUDGETS=` after the script name, and `DRY_RUN=1` to preview.

```bash
bash "$REPO_ROOT/case_studies/scripts/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.tacc.sbatch.sh" \
  METHODS=presto BUDGETS=20
```

Outputs per run: `.log`, `.csv` (scores), `.stats.json` (counters, timings), `.resources.json` (resource probe).

Draw proposal-certainty figures (`--all-edges` counts every logged accept/reject edge, including unvisited branches):

```bash
"$REPO_ROOT/src/.venv/bin/python" \
  "$REPO_ROOT/case_studies/draw/draw_dataset_proposal_certainty.py" \
  --log-root "$REPO_ROOT/case_studies/logs" --run-date 2026-09-16 \
  --model qwen3-4b --prefetch-budget 20 --rank-fn bfs_accept_first \
  --alpha 4.0 --mcmc-steps 100 --num-blocks 1 --max-samples 20 \
  --max-new-tokens 1024 --seed 10086 \
  --datasets math500 aime mbpp gpqa human_eval mmlu lcb_v6 --all-edges --hatch
```

Drawing dependencies: `uv pip install --python "$REPO_ROOT/src/.venv/bin/python" -r "$REPO_ROOT/case_studies/draw/requirements.txt"`. PDF export needs `qpdf`; some figures need LaTeX and Graphviz `dot`. All drawing options: `case_studies/draw/README.md`.


## Layout and tests

| Path | Purpose |
| --- | --- |
| `src/power_sharpening` | Package: common utilities, backends, tasks, configs, runners. |
| `src/tests` | Tests. |
| `experiments` | Backend/benchmark launch scripts. |
| `case_studies` | Launchers, log parsers, figure code, dated outputs. |
| `examples` | CPU-only prefetch DP demo. |
| `data`, `result` | Benchmark inputs; results and analyses. |
| `website` | Project website source. |



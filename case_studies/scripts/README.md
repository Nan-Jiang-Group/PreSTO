# Case-study launchers

Slurm and interactive launchers for the vLLM case studies. Most compare a baseline
MH sampler with its PreSTO (subtree-prefetching) version on one grid; a few run a
single method. Three method families share one launcher design: uniform-cut
PowerMH (this folder), `EntropyCut-family/`, and `MultiTry-family/`.

All file names below are relative to
`/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts`.

## Quick start

```bash
S=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts
# Print the sbatch commands for a narrowed grid without submitting anything.
bash $S/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.punakha.sbatch.sh \
  DATASETS=lcb_v6 MODELS=gemma-12b-it DRY_RUN=1
# Equivalent long form.
bash $S/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.punakha.sbatch.sh \
  --datasets lcb_v6 --models gemma-12b-it --dry-run 1
```

The launchers need bash 4 or newer (the cluster nodes have it; the macOS system
bash 3.2 does not).

## Structure

Each experiment is four files sharing one name:

| File | Role |
| --- | --- |
| `<name>.sh` | Runs the grid. Takes the cluster as its first argument. Holds the `d_*` default grid. |
| `<name>.interactive.sh` | Sets the settings block at its top, then runs `<name>.sh interactive`. |
| `<name>.punakha.sbatch.sh` | Same, then `<name>.sh punakha`. |
| `<name>.tacc.sbatch.sh` | Same, then `<name>.sh tacc`. |

The three entry points each start with an editable settings block of `export`
lines (datasets, models, ranks, budgets, time limit, ...). Edit that block to
change what one cluster runs; it replaces any same-named environment variable.
Command-line arguments after the cluster still win over it.

Setting precedence, highest first: arguments (`KEY=value`); an entry point's
settings block; environment variables (only when `<name>.sh` is run directly);
the `d_*` defaults in `<name>.sh`.

```
<name>.<cluster>[.sbatch].sh       calls <name>.sh <cluster>
<name>.sh                          sets family= and d_* defaults, then sources
  _launch_common.sh                parses overrides, applies cluster settings, loops over dataset/model/rank/budget
    uniform     -> run_power_mh_case_study.sh, run_subtree_prefetching_mh_case_study.sh
    entropycut  -> EntropyCut-family/run_entropycut_case_study.sh  (wraps the two uniform job bodies)
    multitry    -> MultiTry-family/run_multitry_case_study.sh      (calls power_sharpening.runners.vllm.run_*multi_try_mh*)
```

Do not rename the `run_*_case_study.sh` job bodies: queued Slurm jobs and
`lcb_v6_power_mh_vs_presto.tacc.sbatch.sh` call them by path.

| Cluster | Behavior |
| --- | --- |
| `interactive` | Runs in the current shell. Needs an allocated GPU and an active Python environment. |
| `punakha` | `sbatch` on partition `dgx` (`PUNAKHA_MACHINE` overrides), Conda env `vllm-cuda130`. Time limit 20 h (family launchers: 24 h). |
| `tacc` | `sbatch -p gh -A CCR25054` (`TACC_PARTITION`, `TACC_ACCOUNT` override), Conda env `cuda130`. Time limit 12 h. |

## Uniform-cut launchers

Defaults unless noted: alpha 4, 100 MH steps, 1 block, 20 samples, 1,024 new
tokens, batch size 1, seed 10086, proposal temperature 1/alpha.

| Launcher (`.sh`) | Methods | Default grid | Jobs |
| --- | --- | --- | --- |
| `subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm` | PowerMH + PreSTO, resource probe | 7 datasets × 4 models (qwen3.5-4b, qwen3-4b, qwen3-8b, gemma-12b-it), `bfs_accept_first`, budgets 10, 20 | 84 |
| `power_mh_and_subtree_prefetch.long_mh_steps+resource_prob_one-dataset.vllm` | PowerMH + PreSTO, resource probe | LCB v6 × qwen3.5-9b, 7 ranks × budgets 4–20 (step 2) | 64 |
| `power_mh.long_mh_steps` | PowerMH | 8 datasets × 8 models | 64 |
| `power_mh.resource_probe` | PowerMH, resource probe | math500, LCB v6 × 5 Qwen models; 8 blocks, 3,072 tokens | 10 |
| `subtree_prefetch.long_mh_steps.vllm` | PreSTO | 8 datasets × 8 models × 7 ranks × budgets 4–20; one job per dataset/model runs its 63 cells | 64 |
| `subtree_prefetch.large_baseLLM.vllm` | PreSTO | LCB v6 × qwen3.5-27b (bf16, `gpu_memory_utilization=0.95`, `max_model_len=8192`), `bfs_reject_first`, budgets 8–20, no tree logging | 7 |
| `lcb_v6_power_mh_vs_presto.tacc.sbatch` | PowerMH then PreSTO in one allocation | LCB V6-only (175 problems) × 6 models; TACC only; takes `--dry-run` instead of a cluster. See `../lcb-v6-matched-comparison-runs.md` | 6 |

The 7 datasets are math500, aime, mbpp, gpqa, human_eval, lcb_v6, mmlu.
The 8 models are qwen, qwen3-4b, qwen3.5-4b, qwen3-8b, qwen3.5-9b,
qwen-math-medium, tulu, phi3.5. The 7 ranks are `accept_first`, `reject_first`,
`longest_first`, `smallest_cut_diff`, `longest_path_first`, `bfs_accept_first`,
`bfs_reject_first`.

## Family launchers

Each family folder has its own README with the full grid, examples, and job body.

| Launcher | Arms | Default grid | Jobs |
| --- | --- | --- | --- |
| `EntropyCut-family/entropycut_mh_and_subtree_prefetch.resource_probe.vllm` | EntropyCut MH + PreSTO + EntropyCutMH | 7 datasets × 5 models, `accept_first`, budgets 10, 20 | 105 |
| `EntropyCut-family/entropycut_mh.resource_probe` | EntropyCut MH | math500 × qwen | 1 |
| `EntropyCut-family/subtree_prefetching_mh.resource_probe` | PreSTO + EntropyCutMH | math500 × qwen, `bfs_accept_first`, budget 10 | 1 |
| `MultiTry-family/multi_try_mh_and_subtree_prefetch.resource_probe.vllm` | MultiTryMH + PreSTO + MultiTryMH | LCB v6 × qwen3.5-9b, `bfs_accept_first`, budgets 4–20 | 10 |
| `MultiTry-family/multi_try_mh_and_subtree_prefetch.rank_sweep.vllm` | MultiTryMH + PreSTO + MultiTryMH | LCB v6 × qwen3.5-9b, 6 ranks × budgets 4, 8, 12 | 19 |
| `MultiTry-family/multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm` | MultiTryMH v2 + PreSTO + MultiTryMH v2 | math500 × qwen, `bfs_accept_first`, budgets 4, 8, 12 | 4 |
| `MultiTry-family/multi_try_mh.resource_probe` | MultiTryMH | math500 × qwen | 1 |
| `MultiTry-family/subtree_prefetching_multi_try_mh.resource_probe` | PreSTO + MultiTryMH | math500 × qwen, `bfs_accept_first`, budget 10 | 1 |
| `MultiTry-family/subtree_prefetching_multi_try_mh_v2.resource_probe` | PreSTO + MultiTryMH v2 | math500 × qwen, `bfs_accept_first`, budget 10 | 1 |

## Overrides

Pass these after the cluster name (or after an entry point's own name) as
`KEY=value` or `--key value` (`--mcmc-steps 50` sets `MCMC_STEPS=50`); unknown
names stop the launch. Exported in the shell, they apply to `<name>.sh <cluster>`
but are overridden by an entry point's settings block. List values are
space-separated.

| Variable | Meaning |
| --- | --- |
| `METHODS` | `baseline`, `presto`, or both (`power_mh`, `subtree_prefetch` are accepted aliases) |
| `DATASETS`, `MODELS` | Dataset keys from `dataset.yaml`; model keys from `MODEL_MAP` |
| `RANKS`, `BUDGETS` | PreSTO rank functions and prefetch budgets (ignored by the baseline) |
| `ALPHA`, `MCMC_STEPS`, `NUM_BLOCKS`, `MAX_SAMPLES`, `MAX_NEW_TOKENS` | Sampler settings; they also appear in run names |
| `TEMPERATURE` | Proposal temperature; `-1` (default) means 1/alpha |
| `BATCH_SIZE`, `PRINT_TREE` | Defaults `1` and `true` (`PRINT_TREE` applies to PreSTO only) |
| `RESOURCE_PROBE` | `on`, `off`, or `auto` (runner default) |
| `KV_CACHE_MODE` | `hooks` (in-process vLLM, exact KV-cache measurements) or `inherit` |
| `EXTRA_OVERRIDES` | Extra `key=value` runner settings, e.g. `"dtype=bfloat16 max_model_len=8192"` |
| `JOB_PER` | `cell`: one job per dataset/model (baseline) and per dataset/model/rank/budget (PreSTO). `model`: one job per dataset/model/arm running all its cells back to back |
| `CUT_POWER` | EntropyCut exponent (default 4.0; EntropyCut only) |
| `NUM_TRIES`, `PROPOSAL_TEMPERATURES`, `SCORING_BATCH_SIZE` | MultiTry candidates per transition, mixture temperatures (e.g. `'[0.25,0.5,1.0]'`), extra-scoring batch size (ignored by v2) |
| `LOG_ROOT` | Uniform and EntropyCut only: write logs under this root instead of the family default (absolute, or relative to the repo root) |
| `RUN_DATE` | Output date folder; default today |
| `TIME` | Slurm time limit |
| `PUNAKHA_MACHINE`, `TACC_PARTITION`, `TACC_ACCOUNT` | Scheduler targets |
| `DRY_RUN=1` | Print the commands instead of running them |

## Outputs

Per-run logs and results go to `../<tree>/<dataset>/<RUN_DATE>/vllm`, where
`<tree>` is `logs` (uniform), `logs-entropycut`, or `logs-MultiTry`. Uniform
Slurm output goes to `../logs/all-datasets/<RUN_DATE>/vllm`; the families write
theirs to the per-dataset `vllm/slurm` folder. The extraction and drawing code
in `../extract` and `../draw` finds runs by these file names.

## Adding an experiment

Copy the smallest launcher of the same family and change its `d_*` lines and
header comment. Copy its three entry points, change the grid name in each, and
add a row to the tables above.

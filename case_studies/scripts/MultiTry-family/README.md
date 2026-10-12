# MultiTryMH case studies

Both standalone MultiTryMH and subtree-prefetching MultiTryMH have HF and vLLM samplers and package runners. The subtree implementation schedules complete candidate bundles while preserving the existing MTM selection and acceptance rule. The [implementation note](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/IMPLEMENTATION.md) explains the connection to Ye and Lu's PMP-MCMC paper and the shared-prefix construction from the accompanying discussion.

| Backend | Standalone runner | Subtree runner |
| --- | --- | --- |
| HF | [run_multi_try_mh.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/runners/hf/run_multi_try_mh.py) | [run_subtree_prefetching_multi_try_mh.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/runners/hf/run_subtree_prefetching_multi_try_mh.py) |
| vLLM | [run_multi_try_mh.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/runners/vllm/run_multi_try_mh.py) | [run_subtree_prefetching_multi_try_mh.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/runners/vllm/run_subtree_prefetching_multi_try_mh.py) |

The runners read `multi_try` and `subtree_prefetching_multi_try_mh`, respectively, from [algorithms.yaml](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/config/algorithms.yaml).

## Run either backend

Install the package and the corresponding backend dependencies in the GPU environment, then run:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  python -m power_sharpening.runners.hf.run_subtree_prefetching_multi_try_mh \
  --dataset math500 --algorithm subtree_prefetching_multi_try_mh --model_str qwen \
  --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry/math500/hf \
  --override num_tries=4 prefetch_budget=16 'proposal_temperatures=[0.25,0.5,1.0]' \
  num_blocks=1 mcmc_steps=100 max_new_tokens=1024 max_samples=20

srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  python -m power_sharpening.runners.vllm.run_subtree_prefetching_multi_try_mh \
  --dataset math500 --algorithm subtree_prefetching_multi_try_mh --model_str qwen \
  --save_str /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry/math500/vllm \
  --override num_tries=4 prefetch_budget=16 'proposal_temperatures=[0.25,0.5,1.0]' \
  num_blocks=1 mcmc_steps=100 max_new_tokens=1024 max_samples=20 scoring_batch_size=128
```

For the standalone baseline, use `run_multi_try_mh`, select `--algorithm multi_try`, and omit `prefetch_budget`. Both subtree runners write a result CSV and per-sample `.stats.json`; the vLLM runner also supports `.resources.json` with GPU and KV-cache probes.

`num_tries=K` controls candidates per transition. `prefetch_budget=B` counts suffix requests, so a batch holds at most `floor(B/K)` complete bundles. `B` must be at least `K`; unused remainder slots are ignored. With `K=4`, budgets 4 and 6 both allow one bundle, budgets 8 and 10 allow two. `B=K` runs the same subtree engine sequentially and is useful for controlled comparisons.

Subtree sampling requires uniform cuts and a constant proposal distribution. An explicit temperature list defines a uniform mixture of whole-suffix proposals; its length is independent of `num_tries`. Omitting the list uses the scalar proposal temperature. The standalone runners additionally allow their existing scalar temperature schedules. `scoring_batch_size` controls vLLM's extra temperature-scoring requests; HF obtains those scores from generation logits.

## vLLM launchers

Each launcher takes the cluster (`interactive`, `punakha`, or `tacc`) and submits
one job per cell, with resource probes and in-process KV-cache hooks. Punakha
jobs request 24 hours. Shared settings: alpha 4, 100 MH steps, one block, 20
samples, 1,024 output tokens, batch size 1, seed 10086, four tries per
transition, proposal temperatures `[0.25,0.5,1.0]`, and scoring batch size 128
(v1 only). Each launcher `<name>.sh` has three entry points that fix the
cluster and carry an editable settings block: `<name>.interactive.sh`,
`<name>.punakha.sbatch.sh`, and `<name>.tacc.sbatch.sh`.

| Launcher | Arms | Default grid | Jobs |
| --- | --- | --- | --- |
| [multi_try_mh_and_subtree_prefetch.resource_probe.vllm.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_and_subtree_prefetch.resource_probe.vllm.sh) | MultiTryMH + PreSTO + MultiTryMH | LCB v6 × qwen3.5-9b, `bfs_accept_first`, budgets 4–20 in steps of 2, no tree logging | 10 |
| [multi_try_mh_and_subtree_prefetch.rank_sweep.vllm.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_and_subtree_prefetch.rank_sweep.vllm.sh) | MultiTryMH + PreSTO + MultiTryMH | LCB v6 × qwen3.5-9b, six rank functions × budgets 4, 8, 12 | 19 |
| [multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.sh) | MultiTryMH v2 + PreSTO + MultiTryMH v2 | Math500 × Qwen, `bfs_accept_first`, budgets 4, 8, 12, no tree logging | 4 |
| [multi_try_mh.resource_probe.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/multi_try_mh.resource_probe.sh) | MultiTryMH | Math500 × Qwen | 1 |
| [subtree_prefetching_multi_try_mh.resource_probe.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/subtree_prefetching_multi_try_mh.resource_probe.sh) | PreSTO + MultiTryMH | Math500 × Qwen, `bfs_accept_first`, budget 10, no tree logging | 1 |
| [subtree_prefetching_multi_try_mh_v2.resource_probe.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/subtree_prefetching_multi_try_mh_v2.resource_probe.sh) | PreSTO + MultiTryMH v2 | Math500 × Qwen, `bfs_accept_first`, budget 10, no tree logging | 1 |

Pass overrides (`METHODS`, `DATASETS`, `MODELS`, `RANKS`, `BUDGETS`, `NUM_TRIES`,
`PROPOSAL_TEMPERATURES`, `SCORING_BATCH_SIZE`, ...) as arguments; the
[scripts README](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/README.md)
lists them and explains their precedence. Each budget must be at least
`NUM_TRIES`. Before anything is launched, the launcher runs `--check-support`
for every selected arm.

```bash
S=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family
# Budget sweep on Punakha.
bash $S/multi_try_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh
# Quick interactive comparison: Qwen3.5-4B, 10 steps, 5 samples, 3 tries.
bash $S/multi_try_mh_and_subtree_prefetch.resource_probe.vllm.interactive.sh \
  MODELS=qwen3.5-4b MCMC_STEPS=10 MAX_SAMPLES=5 NUM_TRIES=3 PROPOSAL_TEMPERATURES='[0.25,0.5,0.75]' \
  RANKS=accept_first BUDGETS="12 18" PRINT_TREE=true
# MultiTryMH only, Math500 / Qwen.
bash $S/multi_try_mh.resource_probe.interactive.sh
# PreSTO + MultiTryMH only: budget 10, bfs_accept_first.
bash $S/subtree_prefetching_multi_try_mh.resource_probe.interactive.sh
# v2 baseline only.
bash $S/multi_try_mh_v2_and_subtree_prefetch.resource_probe.vllm.interactive.sh METHODS=baseline
```

### v1 baseline vs PreSTO + MultiTryMH v2

`multi_try_mh_vs_subtree_prefetching_multi_try_mh_v2.resource_probe.{interactive,punakha.sbatch,tacc.sbatch}.sh`
run the v1 MultiTryMH baseline and the PreSTO + MultiTryMH v2 arm under one
shared settings block. The `punakha` and `interactive` versions run both arms;
the `tacc` version always runs the v2 arm and runs the baseline only with
`RUN_BASELINE=1`. `...tacc-dev.sbatch.sh` is a smoke test of the TACC launcher
on the `gh-dev` queue (one cell per arm, few problems). These four scripts
call `multi_try_mh.resource_probe.sh` and
`subtree_prefetching_multi_try_mh_v2.resource_probe.sh` and are edited in place,
not through overrides.

The job body,
[run_multitry_case_study.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/run_multitry_case_study.sh),
accepts `--method multitry-mh`, `presto-multitry-mh`, `multitry-mh-v2`, or
`presto-multitry-mh-v2`, plus `--num-tries`, `--proposal-temperatures` (a
quoted inline list), and `--scoring-batch-size`. It also takes the standard
case-study job options, and other sampler settings follow `--override`. It
enforces uniform cuts and a constant schedule. `--check-support` confirms that
the selected runner source exists; it does not load a model or check GPU
dependencies. Do not rename the job body, since launchers and queued jobs call
it by path.

Logs, CSV results, and JSON files go under [logs-MultiTry](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry), organized by dataset, date, and `vllm`; batch output paths use its `slurm` subfolder. Run names include tries, proposal temperatures, scoring batch size, and, for the subtree arm, budget and rank. Log suffixes are `.multiTryMH.vllm.log` and `.subtreePrefetchMultiTryMH.vllm.log`, or `.multiTryMHv2.vllm.log` and `.subtreePrefetchMultiTryMHv2.vllm.log` for v2.

The v2 methods keep the same MTM transitions but build the engine with `log_z_temperatures=proposal_temperatures`. Every suffix is then scored at all mixture temperatures during generation, with no one-token scoring requests. The launcher therefore drops `scoring_batch_size`, and v2 run names use `.v2` in place of `.scorebatchN`. v2 requires an explicit proposal-temperature list.

Validation covers the common kernel with a finite-state oracle, backend adapters and runners with mocked inference, and shell dispatch without submitting jobs. Real-model GPU execution, wall-clock speedups, and numerical agreement across batch shapes remain to be measured.

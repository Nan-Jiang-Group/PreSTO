# EntropyCut MH comparisons

Compares EntropyCut MH (sequential PowerMH with entropy-based cuts) against
PreSTO + EntropyCutMH (the same chain with subtree prefetching). Both arms use
entropy cuts with exponent `CUT_POWER` (default 4) and a constant proposal
temperature of `1/alpha`.

## Launchers

[entropycut_mh_and_subtree_prefetch.resource_probe.vllm.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.sh)
takes the cluster as its argument (`interactive`, `punakha`, or
`tacc`) and submits one job per cell, with resource probes and in-process
KV-cache hooks. The default grid is 7 datasets (math500, aime, mbpp, gpqa,
human_eval, lcb_v6, mmlu) × 5 models (qwen3.5-4b, qwen3-4b, qwen3-8b,
gemma-12b-it, qwen3.5-9b). Each dataset/model pair gets one baseline
plus PreSTO with `accept_first` at budgets 10 and 20, so 105 jobs. Other
settings are alpha 4, 100 MH steps, one block, 20 samples, and 1,024 tokens.
Punakha jobs request 24 hours.

Two single-arm launchers run one method on Math500 / Qwen:
[entropycut_mh.resource_probe.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/entropycut_mh.resource_probe.sh)
(EntropyCut MH only, 1 job) and
[subtree_prefetching_mh.resource_probe.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/subtree_prefetching_mh.resource_probe.sh)
(PreSTO + EntropyCutMH only, `bfs_accept_first` at budget 10, no tree
logging, 1 job).

Each launcher has three entry points that fix the cluster and carry an
editable settings block: `<name>.interactive.sh`, `<name>.punakha.sbatch.sh`,
and `<name>.tacc.sbatch.sh`. Pass overrides (`METHODS`, `DATASETS`, `MODELS`,
`RANKS`, `BUDGETS`, `CUT_POWER`, ...) as arguments. The
[scripts README](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/README.md)
lists them and explains their precedence.

```bash
S=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family
# Full grid on Punakha.
bash $S/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh
# Three datasets only.
bash $S/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.punakha.sbatch.sh DATASETS="gpqa human_eval mmlu"
# Interactive comparison on LCB v6 / Qwen3.5-9B at budget 18.
bash $S/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.interactive.sh DATASETS=lcb_v6 MODELS=qwen3.5-9b BUDGETS=18
# EntropyCut MH only, Math500 / Qwen.
bash $S/entropycut_mh.resource_probe.interactive.sh
# PreSTO + EntropyCutMH only: budget 10, bfs_accept_first, no tree logging.
bash $S/subtree_prefetching_mh.resource_probe.interactive.sh
# Short TACC development-queue run.
bash $S/entropycut_mh_and_subtree_prefetch.resource_probe.vllm.tacc.sbatch.sh \
  TACC_PARTITION=gh-dev TIME=2:00:00 DATASETS=lcb_v6 MODELS=qwen3.5-9b BUDGETS="10 16 18"
```

## Job body

[run_entropycut_case_study.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/run_entropycut_case_study.sh)
accepts `--method entropycut-mh` or `--method presto-entropycut-mh` plus
`--cut-power`. It forwards the standard case-study options (`--datasets`,
`--models`, `--run-date`, `--prefetch-budgets`, `--rank-fns`, `--log-mode`,
`--continue-on-error`, …) and the settings after `--override`. Do not rename
it: queued Slurm jobs call it by path.

EntropyCut MH runs through `power_sharpening.runners.vllm.run_power_mh` with
`cut_power`. PreSTO + EntropyCutMH runs through
`power_sharpening.runners.vllm.run_subtree_prefetching_mh` with
`cut_dist_param`. The job body maps `--cut-power` to the right key and
enforces entropy cuts with a constant temperature schedule.

## Outputs

Logs, CSV results, statistics, and resource artifacts go under
[logs-entropycut](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-entropycut)`/<dataset>/<date>/vllm`.
Slurm output goes to that folder's `slurm` subfolder. Run names include
`.cut-entropy4.0` (or the selected exponent). The shared job bodies choose this
tree whenever `cut_dist_type=entropy`; uniform-cut runs stay in
[logs](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs).

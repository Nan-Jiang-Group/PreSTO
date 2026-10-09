# Experiments

Thin launch scripts, organized as `experiments/<backend>/<benchmark>/`.
Python entry points live in the installed package
(`python -m power_sharpening.runners.<backend>.<runner>`); scripts here only
set hyperparameters, output paths, and (for `*.sbatch.sh`) SLURM submission.

```
experiments/
├── env.sh          # shared env (SEED, basepath, HF caches) — sourced by every script
├── clusters/       # cluster-specific generic submitters (punakha, tacc)
└── hf/             # HuggingFace backend: math500/ (one PowerMH and one PreSTO example)
```

`env.sh` and `clusters/` are also used by the launchers in `case_studies/`. The
paper's runs live there: PowerMH and PreSTO grids in `case_studies/scripts/`,
vLLM vs SGLang in `case_studies/compare_vllm_sglang/`, and the low-temperature,
best-of-N, and Power-SMC sweeps in `case_studies/scaling_baselines/`.

Conventions:

- Every launch script sits exactly two levels below this directory and starts
  with `source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"`.
- `*.interactive.sh` (or a plain `*.sh`) runs in the current shell and assumes a GPU allocation.
- Hyperparameters come from `src/power_sharpening/config/{dataset,algorithms}.yaml`;
  scripts override single values with `--override KEY=VALUE`.
- Results are written under `$basepath/result/<BENCH>/<date>/` (see `env.sh`).

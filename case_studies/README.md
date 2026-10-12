# Case studies

Launch scripts, log parsers, figure code, and dated logs for the paper's runs.
Python entry points live in the installed package
(`python -m power_sharpening.runners.<backend>.<runner>`); the shell scripts here
only set hyperparameters, output paths, and (for `*.sbatch.sh`) SLURM submission.

```
case_studies/
├── env.sh                         # shared env (SEED, basepath, HF caches) — sourced by every launcher
├── clusters/                      # cluster-specific generic submitters (punakha, tacc)
├── hf/                            # HuggingFace backend: math500/ (one PowerMH and one PreSTO example)
├── scripts/                       # vLLM PowerMH / PreSTO grids, EntropyCut and MultiTry families
├── compare_vllm_sglang/           # vLLM vs SGLang
├── scaling_baselines/             # low-temperature, best-of-N, and Power-SMC sweeps
├── score_log_likelihood_and_confidence/
├── extract/                       # log parsers
├── draw/                          # figures
└── logs*/                         # dated run logs
```

Conventions:

- Scripts two levels below this directory (e.g. `hf/math500/`) start with
  `source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"`; the other launchers
  source `"$repo_root/case_studies/env.sh"`.
- `*.interactive.sh` (or a plain `*.sh`) runs in the current shell and assumes a GPU allocation.
- Hyperparameters come from `src/power_sharpening/config/{dataset,algorithms}.yaml`;
  scripts override single values with `--override KEY=VALUE`.
- Results are written under `$basepath/result/<BENCH>/<date>/` (see `env.sh`).

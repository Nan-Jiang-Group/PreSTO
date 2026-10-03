# Experiments

Thin launch scripts, organized as `experiments/<backend>/<benchmark>/`.
Python entry points live in the installed package
(`python -m power_sharpening.runners.<backend>.<runner>`); scripts here only
set hyperparameters, output paths, and (for `*.sbatch.sh`) SLURM submission.

```
experiments/
├── env.sh          # shared env (SEED, basepath, HF caches) — sourced by every script
├── clusters/       # cluster-specific generic submitters (punakha, tacc)
├── hf/             # HuggingFace backend: alpaca/ gpqa/ human_eval/ math500/
├── vllm/           # vLLM backend: aime24-25/ gpqa/ human_eval/ lcb/
│                   #   math500/ mbpp/ mmlu/ swebench/
└── sglang/         # SGLang backend: math500/
```

Conventions:

- Every launch script sits exactly two levels below this directory and starts
  with `source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"`.
- `*.interactive.sh` runs in the current shell (assumes a GPU allocation);
  `*.<cluster>.sbatch.sh` submits itself via an `sbatch <<EOT` heredoc.
- Hyperparameters come from `src/power_sharpening/config/{dataset,algorithms}.yaml`;
  scripts override single values with `--override KEY=VALUE`.
- Results are written under `$basepath/result/<BENCH>/<date>/` (see `env.sh`).

Known exceptions (flagged with a `NOT RUNNABLE` header comment):

- `hf/alpaca/power_samp_alpaca.sh` — verbatim from the external
  reasoning-with-sampling repo; no `alpaca` task exists in
  `power_sharpening.tasks`.
- `hf/math500/accuracy.gilbreth.sh` — the `math_experiments.eval_math`
  CSV-accuracy evaluator was dropped in the restructure.
- `vllm/swebench/low_temp.interactive.sh` — migrated to the new CLI, but
  `dataset.yaml`'s `swebench` entry has no benchmark class yet, so the runner
  exits with "unsupported task".

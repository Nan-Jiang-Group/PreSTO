#!/bin/bash
# The entropycut_mh.resource_probe.sh grid on punakha. Submits its jobs with sbatch on Punakha (dgx partition, Conda env vllm-cuda130).
# Edit the settings below, then run:
#   bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family/entropycut_mh.resource_probe.punakha.sbatch.sh

# ---- Settings (edit here) ----
export METHODS=baseline                   # baseline and/or presto
export DATASETS=math500                   # space-separated, e.g. "math500 aime"
export MODELS=qwen                        # keys of power_sharpening.tasks.constants.MODEL_MAP, e.g. "qwen3.5-9b"
export ALPHA=4.0                          # sharpening power p^alpha
export MCMC_STEPS=100                     # total MH steps
export NUM_BLOCKS=1
export MAX_SAMPLES=20                     # number of benchmark problems
export MAX_NEW_TOKENS=1024
export TEMPERATURE=-1                     # -1 uses proposal temperature 1/alpha
export CUT_POWER=4.0
export RESOURCE_PROBE=on                  # on | off | auto
export KV_CACHE_MODE=hooks                # hooks | inherit
export EXTRA_OVERRIDES=""                 # extra Python key=value settings, e.g. "dtype=bfloat16 max_model_len=8192"
export PUNAKHA_MACHINE=dgx
export TIME=1-00:00:00                    # Slurm time limit (QOS max 1-00:00:00)
export RUN_DATE="$(date +%F)"             # log folder date
# ------------------------------

exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/entropycut_mh.resource_probe.sh" punakha "$@"

# env.sh — shared environment for every launch script under experiments/.
# Each script (always two levels below this file) loads it as its first line:
#   source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
set -x
SEED=10086
# The checkout holding this file, so a clone works under any directory name.
basepath=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
if [[ "$(hostname)" == computeinstance-* ]]; then
    # Single-node H100 VM: conda env cuda130 (CUDA 13.0), caches on the 1.3T root disk via ~/SCRATCH.
    source ~/WORK/miniconda3/bin/activate cuda130
    py=~/WORK/miniconda3/envs/cuda130/bin/python
    # Make `uv run --project "$basepath/src"` reuse the conda env instead of creating src/.venv,
    # and skip its exact sync, which would uninstall the extras the script did not name.
    export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
    export UV_NO_SYNC=1

    CACHE_DIR=~/SCRATCH
    export TQDM_DISABLE=1
    export HF_HOME="$CACHE_DIR/huggingface"
    export HF_HUB_CACHE="$CACHE_DIR/hub"
    export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
    mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
else
    CACHE_DIR=~/SCRATCH/
    TQDM_DISABLE=1

    export HF_HOME="$HOME/SCRATCH/huggingface"
    export HF_HUB_CACHE="$CACHE_DIR/hub"
    export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
fi

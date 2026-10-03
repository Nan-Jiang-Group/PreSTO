# Re-score sampled completions with base-model log-likelihood and confidence under p_0 for every clean back-to-back
# pair of a sampler with and without PreSTO. Each job is one dataset/model pair: the baseline run (PowerMH or
# EntropyCut) followed by its PreSTO runs. A pair is clean when both sides match in dataset file, prompt style, alpha,
# MH steps, blocks, cut law, 20 samples, 1024 new tokens, and seed. PowerMH and EntropyCut never log these metrics, and
# uniform-cut PreSTO does not either; PreSTO-EntropyCut does, and its logged values then check the rescoring.
# Writes <run>.rescored.csv beside each input CSV. A failing job does not stop the others; failures are listed at the
# end and the script exits non-zero.
#
# Needs one GPU:
#
#   srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=12:00:00 \
#     zsh /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/score_log_likelihood_and_confidence/score_log_likelihood_and_confidence.sh
#
# CHECK_ONLY=1 runs only the dataset, prompt, and round-trip checks (tokenizers, no model, no GPU) and writes nothing.
# PYTHON overrides the interpreter; ONLY="<dataset> <model>" restricts the run to matching jobs, e.g. ONLY="math500".
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
case_dir=$(cd -- "$script_dir/.." && pwd)

# Environment, taken from experiments/env.sh: only what the scorer needs (interpreter, HF caches, no progress bars).
# Skipped from there: set -x tracing, SEED and basepath (the CSV paths and seeds come from the runs), and the uv
# settings. PYTHON set by the caller wins, and skips the conda activation.
export TQDM_DISABLE=1
if [[ "$(hostname)" == computeinstance-* ]]; then
    # Single-node H100 VM: conda env cuda130 (CUDA 13.0), caches on the root disk via ~/SCRATCH.
    if [[ -z ${PYTHON:-} ]]; then
        source ~/WORK/miniconda3/bin/activate cuda130
        PYTHON=~/WORK/miniconda3/envs/cuda130/bin/python
    fi
    CACHE_DIR=~/SCRATCH
    export HF_HOME="$CACHE_DIR/huggingface"
    export HF_HUB_CACHE="$CACHE_DIR/hub"
    export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
    mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
else
    CACHE_DIR=~/SCRATCH/
    export HF_HOME="$HOME/SCRATCH/huggingface"
    export HF_HUB_CACHE="$CACHE_DIR/hub"
    export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
fi
py=${PYTHON:-python}

jobs=(
    # Uniform cut: PowerMH vs PreSTO-PowerMH (bfs_accept_first, prefetch budget 20; accept_first for Qwen2.5-7B;
    # budget 10 for LCB V6)
    "math500 qwen logs/math500/2026-08-13/vllm/dataset-math500.model-qwen.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/math500/2026-08-13/vllm/dataset-math500.model-qwen.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.samples20.maxnew1024.seed10086.csv"
    "math500 qwen3.5-4b logs/math500/2026-08-13/vllm/dataset-math500.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/math500/2026-09-16/vllm/dataset-math500.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "math500 qwen3.5-9b logs/math500/2026-08-13/vllm/dataset-math500.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/math500/2026-09-04/vllm/dataset-math500.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "aime qwen3.5-4b logs/aime/2026-08-13/vllm/dataset-aime.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/aime/2026-09-16/vllm/dataset-aime.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "aime qwen3.5-9b logs/aime/2026-08-13/vllm/dataset-aime.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/aime/2026-09-04/vllm/dataset-aime.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "gpqa qwen3.5-4b logs/gpqa/2026-08-13/vllm/dataset-gpqa.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/gpqa/2026-09-16/vllm/dataset-gpqa.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "gpqa qwen3.5-9b logs/gpqa/2026-08-13/vllm/dataset-gpqa.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/gpqa/2026-09-05/vllm/dataset-gpqa.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "mmlu qwen3.5-4b logs/mmlu/2026-08-13/vllm/dataset-mmlu.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/mmlu/2026-09-16/vllm/dataset-mmlu.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "mmlu qwen3.5-9b logs/mmlu/2026-08-13/vllm/dataset-mmlu.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/mmlu/2026-09-04/vllm/dataset-mmlu.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "mbpp qwen3.5-4b logs/mbpp/2026-08-13/vllm/dataset-mbpp.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/mbpp/2026-09-16/vllm/dataset-mbpp.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "mbpp qwen3.5-9b logs/mbpp/2026-08-13/vllm/dataset-mbpp.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/mbpp/2026-09-04/vllm/dataset-mbpp.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3-4b logs/human_eval/2026-08-13/vllm/dataset-human_eval.model-qwen3-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/human_eval/2026-09-16/vllm/dataset-human_eval.model-qwen3-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3-8b logs/human_eval/2026-08-13/vllm/dataset-human_eval.model-qwen3-8b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/human_eval/2026-09-16/vllm/dataset-human_eval.model-qwen3-8b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3.5-4b logs/human_eval/2026-08-13/vllm/dataset-human_eval.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/human_eval/2026-09-16/vllm/dataset-human_eval.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3.5-9b logs/human_eval/2026-08-13/vllm/dataset-human_eval.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/human_eval/2026-09-04/vllm/dataset-human_eval.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "human_eval gemma-12b-it logs/human_eval/2026-09-24/vllm/dataset-human_eval.model-gemma-12b-it.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/human_eval/2026-09-24/vllm/dataset-human_eval.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    "lcb_v6 qwen3.5-4b logs/lcb_v6/2026-09-23/vllm/dataset-lcb_v6.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.csv logs/lcb_v6/2026-09-23/vllm/dataset-lcb_v6.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-bfs_accept_first.samples20.maxnew1024.seed10086.csv"
    # EntropyCut vs PreSTO-EntropyCut (accept_first, prefetch budgets 10 and 20)
    "lcb_v6 qwen3.5-4b logs-entropycut/lcb_v6/2026-09-20/vllm/dataset-lcb_v6.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/lcb_v6/2026-09-20/vllm/dataset-lcb_v6.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/lcb_v6/2026-09-20/vllm/dataset-lcb_v6.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "lcb_v6 qwen3.5-9b logs-entropycut/lcb_v6/2026-09-20/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/lcb_v6/2026-09-23/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/lcb_v6/2026-09-20/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "math500 qwen3.5-4b logs-entropycut/math500/2026-09-20/vllm/dataset-math500.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/math500/2026-09-20/vllm/dataset-math500.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/math500/2026-09-20/vllm/dataset-math500.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "math500 qwen3.5-9b logs-entropycut/math500/2026-09-20/vllm/dataset-math500.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/math500/2026-09-20/vllm/dataset-math500.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/math500/2026-09-20/vllm/dataset-math500.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "aime qwen3.5-4b logs-entropycut/aime/2026-09-21/vllm/dataset-aime.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/aime/2026-09-21/vllm/dataset-aime.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/aime/2026-09-21/vllm/dataset-aime.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "aime qwen3.5-9b logs-entropycut/aime/2026-09-21/vllm/dataset-aime.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/aime/2026-09-21/vllm/dataset-aime.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/aime/2026-09-21/vllm/dataset-aime.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "gpqa qwen3.5-4b logs-entropycut/gpqa/2026-09-23/vllm/dataset-gpqa.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/gpqa/2026-09-23/vllm/dataset-gpqa.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/gpqa/2026-09-23/vllm/dataset-gpqa.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "gpqa qwen3.5-9b logs-entropycut/gpqa/2026-09-23/vllm/dataset-gpqa.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/gpqa/2026-09-23/vllm/dataset-gpqa.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/gpqa/2026-09-23/vllm/dataset-gpqa.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "mmlu qwen3.5-4b logs-entropycut/mmlu/2026-09-23/vllm/dataset-mmlu.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mmlu/2026-09-23/vllm/dataset-mmlu.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mmlu/2026-09-23/vllm/dataset-mmlu.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "mbpp qwen3.5-4b logs-entropycut/mbpp/2026-09-21/vllm/dataset-mbpp.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mbpp/2026-09-21/vllm/dataset-mbpp.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mbpp/2026-09-21/vllm/dataset-mbpp.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "mbpp qwen3.5-9b logs-entropycut/mbpp/2026-09-21/vllm/dataset-mbpp.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mbpp/2026-09-21/vllm/dataset-mbpp.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mbpp/2026-09-21/vllm/dataset-mbpp.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3-4b logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3-8b logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3-8b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3-8b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3-8b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3.5-4b logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3.5-4b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "human_eval qwen3.5-9b logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-10.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-09-23/vllm/dataset-human_eval.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    # Gemma4-12B-it EntropyCut vs PreSTO-EntropyCut (accept_first, prefetch budget 20; 2026-10-01 runs)
    "math500 gemma-12b-it logs-entropycut/math500/2026-10-01/vllm/dataset-math500.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/math500/2026-10-01/vllm/dataset-math500.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "aime gemma-12b-it logs-entropycut/aime/2026-10-01/vllm/dataset-aime.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/aime/2026-10-01/vllm/dataset-aime.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "mbpp gemma-12b-it logs-entropycut/mbpp/2026-10-01/vllm/dataset-mbpp.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mbpp/2026-10-01/vllm/dataset-mbpp.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "gpqa gemma-12b-it logs-entropycut/gpqa/2026-10-01/vllm/dataset-gpqa.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/gpqa/2026-10-01/vllm/dataset-gpqa.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "human_eval gemma-12b-it logs-entropycut/human_eval/2026-10-01/vllm/dataset-human_eval.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/human_eval/2026-10-01/vllm/dataset-human_eval.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "mmlu gemma-12b-it logs-entropycut/mmlu/2026-10-01/vllm/dataset-mmlu.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/mmlu/2026-10-01/vllm/dataset-mmlu.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
    "lcb_v6 gemma-12b-it logs-entropycut/lcb_v6/2026-10-01/vllm/dataset-lcb_v6.model-gemma-12b-it.alpha4.0.steps100.blocks-1.cut-entropy4.0.samples20.maxnew1024.seed10086.csv logs-entropycut/lcb_v6/2026-10-01/vllm/dataset-lcb_v6.model-gemma-12b-it.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first.cut-entropy4.0.samples20.maxnew1024.seed10086.csv"
)

failed=()
ran=0
for job in "${jobs[@]}"; do
    parts=(${=job})
    dataset=$parts[1] model=$parts[2]
    [[ -z ${ONLY:-} || " $dataset $model " == *" ${ONLY} "* ]] || continue
    echo "==> $dataset / $model"
    (( ran++ ))
    "$py" -m power_sharpening.runners.hf.score_log_likelihood_and_confidence \
        --dataset "$dataset" --model_str "$model" --dtype bfloat16 ${CHECK_ONLY:+--check-only} \
        --csv "${(@)parts[3,-1]/#/$case_dir/}" || failed+=("$dataset $model")
done

echo "$(( ran - ${#failed[@]} )) of $ran jobs finished"
if (( ${#failed[@]} )); then
    printf 'FAILED: %s\n' "${failed[@]}" >&2
    exit 1
fi

#!/bin/bash
#SBATCH --job-name=psamp_math
#SBATCH -t 0-23:59                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores (MB)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-47               # 6 shards × 8 seeds = 48 tasks

# --- map array id -> (batch_idx, seed) ---
NUM_SHARDS=6
NUM_SEEDS=8
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME={HUGGING_FACE_HOME}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

export PYTHONPATH="$PYTHONPATH:{/path/to/reasoning-with-sampling/llm_experiments}"
export HF_TOKEN={HF_TOKEN}

source activate psamp
cd /path/to/reasoning-with-sampling/llm_experiments

echo "Running shard BATCH_IDX=${BATCH_IDX} with SEED=${SEED} (task ${SLURM_ARRAY_TASK_ID})"
python power_samp_gpqa.py \
  --batch_idx="${BATCH_IDX}" \
  --mcmc_steps=10 \
  --temp=0.25 \
  --seed="${SEED}" \
  --model=qwen_math

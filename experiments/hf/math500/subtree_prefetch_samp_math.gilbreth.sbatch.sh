#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py3=~/data/miniconda3/envs/LLM/bin/python

# Gilbreth-specific scratch (overrides default.sh's ~/SCRATCH cache paths).
CACHE_DIR=/scratch/gilbreth/$USER
export HF_HOME="$CACHE_DIR/.cache/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"

base_model=qwen-math-small

dump_dir=$basepath/result/MATH500/$(date +%F)
if [ ! -d "$dump_dir" ]
then
  echo "create dir: $dump_dir"
  mkdir -p $dump_dir
fi
for alpha in 2.0 4.0 8.0;
do
  for batch_size in 4 8 16;
  do
    echo "alpha = $alpha, prefetch_budget = $batch_size"
    sbatch -A yexiang  --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=6  --partition=a30 -t 06:00:00 --mem 60G <<EOT
#!/bin/bash
#SBATCH --job-name="math Qwen presto-powermh a${alpha} b${batch_size}"
#SBATCH --output=$dump_dir/dataset-MATH500.model-$base_model.alpha${alpha}.batch${batch_size}.out
hostname
which $py3

$py3 -m power_sharpening.runners.hf.run_subtree_prefetching_mh \
  --dataset math500 \
  --algorithm subtree_prefetching_mh \
  --model_str=$base_model \
  --save_str $dump_dir \
  --override seed=${SEED} alpha=$alpha mcmc_steps=10 prefetch_budget=$batch_size \
  > $dump_dir/dataset-MATH500.model-$base_model.alpha${alpha}.batch${batch_size}.log
EOT
  done
done

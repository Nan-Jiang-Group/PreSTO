#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
submitter="$(dirname "${BASH_SOURCE[0]:-$0}")/../../clusters/punakha.sbatch.sh"

# Config-driven low-temperature baseline (run_low_temp) on MATH500 via Punakha SLURM. Hyperparameters come from
# src/power_sharpening/config/*.yaml.

dataset=math500
algorithm=low_temp

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir
echo "${algorithm} on ${dataset}..."

for base_model in qwen-math-medium qwen3-8b;
do
for temp in 0.15 0.2 0.25;
do
run_name=model-${base_model}.algo-${algorithm}.dataset-${dataset}.temp-${temp}.seed-${SEED}

zsh "$submitter" \
  "$dump_dir" \
  "$algorithm" \
  "$run_name" \
  -m power_sharpening.runners.vllm.run_low_temp \
  --dataset $dataset \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED temperature=${temp}
done
done

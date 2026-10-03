#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
submitter="$(dirname "${BASH_SOURCE[0]:-$0}")/../../clusters/tacc.sbatch.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

aime_dataset=aime2024-2025
algorithm=smc
n_particles=16
max_new_tokens=20480
dump_dir=$basepath/result/AIME/${aime_dataset}/$(date +%F)
mkdir -p $dump_dir
echo "${aime_dataset} dataset!..."

for base_model in qwen3-4b qwen3-4b-instruct qwen3-8b;
do
run_name=model-${base_model}.algo-${algorithm}.${aime_dataset}.nparticles-${n_particles}.max_new_tokens-${max_new_tokens}.seed-${SEED}

zsh "$submitter" \
  "$dump_dir" \
  "$algorithm" \
  "$run_name" \
  -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset aime \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} max_new_tokens=${max_new_tokens} aime_dataset=${aime_dataset}
done

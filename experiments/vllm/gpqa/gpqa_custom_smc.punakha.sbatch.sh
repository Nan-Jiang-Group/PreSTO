#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
submitter="$(dirname "${BASH_SOURCE[0]:-$0}")/../../clusters/punakha.sbatch.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

dataset=gpqa
algorithm=smc
max_new_tokens=8192
dump_dir=$basepath/result/GPQA/$(date +%F)
mkdir -p $dump_dir
echo "GPQA dataset!..."

for base_model in qwen3-4b qwen3-8b tulu phi3.5;
do
for n_particles in 2 4 8 16;
do
run_name=model-${base_model}.algo-custom-smc.nparticles-${n_particles}.max_new_tokens-${max_new_tokens}.seed-${SEED}

PUNAKHA_TIME=20:00:00 zsh "$submitter" \
  "$dump_dir" \
  "custom-smc" \
  "$run_name" \
  -m power_sharpening.runners.vllm.run_custom_smc \
  --dataset $dataset \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} max_new_tokens=${max_new_tokens}
done
done

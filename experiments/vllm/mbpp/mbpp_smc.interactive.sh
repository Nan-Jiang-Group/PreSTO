#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

dump_dir=$basepath/result/MBPP/$(date +%F)
mkdir -p $dump_dir
echo "MBPP dataset!..."

for base_model in qwen qwen-math-medium qwen3-4b qwen3-8b tulu phi3.5; #qwen qwen-math-small qwen-math-medium qwen-math-grpo qwen-grpo qwen-coder qwen-coder-instruct qwen2.5-coder-7b qwen2.5-coder-7b-instruct qwen3-4b qwen3-4b-instruct qwen3-4b-thinking qwen3-4b-grpo qwen3-8b qwen3-8b-thinking qwen3-8b-grpo deepseek-math deepseek-math-7b deepseek-math-instruct deepseek-math-grpo phi phi3.5 phi4-mini-instruct llama3.1-8b llama3.1-8b-instruct llama3.1-8b-grpo tulu tulu3-grpo tulu3-sft tulu3-dpo;
do
# for n_particles in 8 16 32 64;
# do
n_particles=16
run_name=model-${base_model}.algo-smc.nparticles-${n_particles}.seed-${SEED}
python -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset mbpp \
  --algorithm smc \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} \
  > $dump_dir/${run_name}.smc.vllm.log 2>&1
done
# done

source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

dump_dir=$basepath/result/human_eval/$(date +%F)
mkdir -p $dump_dir
echo "Human Eval dataset!..."

for base_model in qwen qwen-math-grpo qwen-grpo qwen-coder qwen-coder-instruct qwen2.5-coder-7b qwen2.5-coder-7b-instruct qwen3-4b qwen3-4b-instruct qwen3-4b-thinking qwen3-4b-grpo qwen3-8b qwen3-8b-thinking qwen3-8b-grpo qwen3-30b-a3b phi phi3.5 phi4-mini-instruct llama3.1-8b llama3.1-8b-instruct llama3.1-8b-grpo tulu tulu3-grpo tulu3-sft tulu3-dpo;
do
for n_particles in 8 16 32 64;
do
run_name=model-${base_model}.algo-smc.nparticles-${n_particles}.seed-${SEED}
python -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset human_eval \
  --algorithm smc \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} \
  > $dump_dir/${run_name}.smc.vllm.log 2>&1
done
done

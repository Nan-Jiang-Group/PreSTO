source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

# Sampler hyperparameters (alpha, block_size, ess_threshold, temperature, ...) now live in
# src/power_sharpening/config/{dataset,algorithms}.yaml. This script selects the dataset + model and overrides only the
# values it varies.

aime_dataset=aime2024-2025
dump_dir=$basepath/result/AIME/${aime_dataset}/$(date +%F)
mkdir -p $dump_dir
echo "${aime_dataset} dataset!..."

for base_model in qwen3-4b qwen3-4b-instruct qwen3-8b; 
do
# for n_particles in 8 16 32 64;
# do
n_particles=16
max_new_tokens=20480
run_name=model-${base_model}.algo-smc.${aime_dataset}.nparticles-${n_particles}.max_new_tokens-${max_new_tokens}.seed-${SEED}
python -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset aime \
  --algorithm smc \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} max_new_tokens=${max_new_tokens} aime_dataset=${aime_dataset} \
  > $dump_dir/${run_name}.smc.vllm.log 2>&1
done
# done

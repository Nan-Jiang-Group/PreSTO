#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/cuda130/bin/python

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

dataset=aime
algorithm=smc
aime_dataset=aime2024-2025
max_new_tokens=8192
dump_dir=$basepath/result/AIME/${aime_dataset}/$(date +%F)
mkdir -p $dump_dir
echo "${aime_dataset} dataset!..."

for base_model in qwen3-4b qwen3-4b-instruct qwen3-8b;
do
for n_particles in 2 4 8 16;
do
run_name=model-${base_model}.algo-custom-smc.${aime_dataset}.nparticles-${n_particles}.max_new_tokens-${max_new_tokens}.seed-${SEED}
sbatch -p gh -A CCR25054 -N 1 -n 1 -t 20:00:00 <<EOT
#!/bin/bash
#SBATCH --job-name="${run_name}"
#SBATCH --output=$dump_dir/${run_name}.custom-smc.vllm.out
hostname

$py -m power_sharpening.runners.vllm.run_custom_smc \
  --dataset $dataset \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles} max_new_tokens=${max_new_tokens} aime_dataset=${aime_dataset} \
  > $dump_dir/${run_name}.custom-smc.vllm.log 2>&1
EOT
done
done

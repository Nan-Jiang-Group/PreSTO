#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/cuda130/bin/python

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.

dataset=mbpp
algorithm=smc
max_new_tokens=8192
dump_dir=$basepath/result/MBPP/$(date +%F)
mkdir -p $dump_dir
echo "MBPP dataset!..."

for base_model in qwen qwen-math-medium qwen3-4b qwen3-8b tulu phi3.5;
do
for n_particles in 2 4 8 16;
do
run_name=model-${base_model}.algo-custom-smc.nparticles-${n_particles}.seed-${SEED}
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
  --override seed=$SEED n_particles=${n_particles} \
  > $dump_dir/${run_name}.custom-smc.vllm.log 2>&1
EOT
done
done

#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/cuda130/bin/python

# Sampler hyperparameters live in src/power_sharpening/config/{dataset,algorithms}.yaml; this script selects the dataset
# + model and overrides only what it varies.
#
# NOTE: dataset.yaml defines swebench, but the task is not registered in
# power_sharpening.tasks.registry yet. Submitted jobs will remain unsupported until that registry entry and its
# version-specific construction are added.

dataset=swebench
algorithm=smc
swebench_version=lite
dump_dir=$basepath/result/swebench_${swebench_version}/$(date +%F)
mkdir -p $dump_dir
echo "SWEBench dataset (version=${swebench_version})..."

for base_model in qwen-coder;
do
for n_particles in 2 4 8 16;
do
run_name=model-${base_model}.algo-custom-smc.swebench-${swebench_version}.nparticles-${n_particles}.seed-${SEED}
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
  --override seed=$SEED n_particles=${n_particles} swebench_version=${swebench_version} \
  > $dump_dir/${run_name}.custom-smc.vllm.log 2>&1
EOT
done
done

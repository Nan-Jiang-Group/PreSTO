#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py3=~/WORK/miniconda3/bin/python


algorithm=power_mcmc
base_model=qwen-math-small
mcmc_steps=10
alpha=4.0
temp=0.5
num_blocks=8
temperature_schedule_type=cosine

dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir


sbatch -p gh -N 1 -n 1 -t 01:00:00  <<EOT
#!/bin/bash
#SBATCH --job-name="math-Qwen $algorithm"
#SBATCH --output=$dump_dir/dataset-MATH500.model-$base_model.temp$temp.algo-$algorithm.out
hostname
which $py3

$py3 -m power_sharpening.runners.hf.run_power_mh \
  --task math500 \
  --mcmc_steps=${mcmc_steps} \
   --algorithm ${algorithm} \
  --alpha $alpha \
  --num_blocks=${num_blocks} \
  --seed=${SEED} \
  --model_str=${base_model} \
  --temperature_schedule_type ${temperature_schedule_type} \
  --temperature=${temp} > $dump_dir/model-${base_model}.algo-${algorithm}.temp-${temp}.steps${mcmc_steps}.blocks${num_blocks}.seed${SEED}.temp_sched_type-${temperature_schedule_type}.power-sharp.hf.log
EOT

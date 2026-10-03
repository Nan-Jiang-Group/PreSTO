#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
py=~/WORK/miniconda3/envs/cuda129/bin/python


algorithm=best_of_n
base_model=qwen-math-medium

batch_size=164

dump_dir=$basepath/result/HumanEval/$(date +%F)
mkdir -p $dump_dir

for N in 1 2 4 8 16 32 64 128;
do
for temp in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
sbatch -p gh -N 1 -n 1 -t 02:00:00  <<EOT
#!/bin/bash
#SBATCH --job-name="model-${base_model}.algo-${algorithm}.InitTemp-${temp}.best-of-N${N}.seed-${SEED}"
#SBATCH --output=$dump_dir/model-${base_model}.algo-${algorithm}.InitTemp-${temp}.best-of-N${N}.seed-${SEED}.low_temp.vllm.out
hostname

$py -m power_sharpening.runners.vllm.run_low_temp \
  --dataset human_eval \
  --algorithm ${algorithm} \
  --model_str=${base_model} \
  --save_str $dump_dir \
  --override seed=${SEED} batch_size=${batch_size} temperature=${temp} best_of_N=[${N}] \
  > $dump_dir/model-${base_model}.algo-${algorithm}.InitTemp-${temp}.best-of-N${N}.seed-${SEED}.low_temp.vllm.log 2>&1
EOT
done
done

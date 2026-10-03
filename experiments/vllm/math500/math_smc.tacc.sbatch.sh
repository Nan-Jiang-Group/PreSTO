#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"
submitter="$(dirname "${BASH_SOURCE[0]:-$0}")/../../clusters/tacc.sbatch.sh"

n_particles=16
dump_dir=$basepath/result/MATH500/$(date +%F)
mkdir -p $dump_dir
algorithm=smc

for base_model in qwen3-4b qwen3-4b-instruct qwen3-8b;
do
run_name=model-${base_model}.algo-${algorithm}.nparticles-${n_particles}.seed-${SEED}

zsh "$submitter" \
  "$dump_dir" \
  "$algorithm" \
  "$run_name" \
  -m power_sharpening.runners.vllm.run_standard_smc \
  --dataset math500 \
  --algorithm $algorithm \
  --model_str=$base_model \
  --save_str $dump_dir \
  --run_name "$run_name" \
  --override seed=$SEED n_particles=${n_particles}
done

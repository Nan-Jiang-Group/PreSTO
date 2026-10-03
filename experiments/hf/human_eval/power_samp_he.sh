#!/bin/zsh
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../env.sh"

algorithm=power_mcmc
base_model=qwen-math-medium
mcmc_steps=10
temp=0.25

dump_dir=$basepath/result/HumanEval/$(date +%F)
mkdir -p $dump_dir

python -m power_sharpening.runners.hf.run_power_mh \
  --task human_eval \
  --mcmc_steps=${mcmc_steps} \
  --temperature=${temp} \
  --seed=${SEED} \
  --model_str=${base_model} \
  --algorithm ${algorithm} \
  --save_str ${dump_dir} \
  > $dump_dir/dataset-HumanEval.model-${base_model}.algo-${algorithm}.temp${temp}.steps${mcmc_steps}.seed${SEED}.power-sharp.hf.log

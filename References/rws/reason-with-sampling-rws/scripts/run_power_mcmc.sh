#!/usr/bin/zsh
echo "Power Sampling MCMC reproduction."
set -x

basepath=~/workspace/sym-reason-llm/
py3=~/miniconda3/bin/python

method=power_mcmc
dataname=sample_math
basellm=Qwen2.5-math
datasetpath=$basepath/data/$dataname.jsonl
n_questions=10

dump_dir=$basepath/result/$method/${dataname}/$(date +%F)
if [ ! -d "$dump_dir" ]
then
    echo "create dir: $dump_dir"
    mkdir -p $dump_dir
fi

$py3 $basepath/scripts/run_experiment.py --config $basepath/configs/run_$method.yaml \
    --method $method --dataset $datasetpath \
    --n_questions $n_questions  > $dump_dir/baseLLM-$basellm.dataset-$dataname.method-$method.out

#
#Experiment runner for Power Sampling MCMC reproduction.
#
#Runs experiments with specified configurations and saves wilcoxon_test_mcmc_vs_standard to CSV.
#
#Supports two model modes:
#1. Single model: Use `model_name` for all methods (backward compatible)
#2. Dual model: Use `base_model_name` for power_mcmc, `grpo_model_name` for baselines
#
#Usage:
#    python scripts/run_experiment.py --config configs/exp_nmcmc_sweep.yaml
#    python scripts/run_experiment.py --config configs/main_comparison.yaml
#    python scripts/run_experiment.py --method power_mcmc --nmcmc 5 --n_questions 10
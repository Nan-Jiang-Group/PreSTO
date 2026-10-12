CACHE_DIR=~/SCRATCH/

export HF_HOME="$HOME/SCRATCH/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"

py=~/WORK/miniconda3/envs/cuda129/bin/python
time_limit=${TACC_TIME:-04:00:00}

if (( $# < 4 )); then
  echo "Usage: $0 DUMP_DIR ALGORITHM RUN_NAME PYTHON_SCRIPT [PYTHON_ARGS...]"
  exit 2
fi

dump_dir=$1
algorithm=$2
run_name=$3
script_path=$4
shift 4

mkdir -p "$dump_dir"
output_file=$dump_dir/${run_name}.${algorithm}.vllm.out
log_file=$dump_dir/${run_name}.${algorithm}.vllm.log

command_parts=("$py" "$script_path" "$@")
command_line="${(j: :)${(q)command_parts[@]}}"

sbatch -p gh -A CCR25054 -N 1 -n 1 -t "$time_limit" <<EOT
#!/bin/bash
#SBATCH --job-name="${run_name}"
#SBATCH --output=${output_file}
hostname

${command_line} > ${log_file} 2>&1
EOT

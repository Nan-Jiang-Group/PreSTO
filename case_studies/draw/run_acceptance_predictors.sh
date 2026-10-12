#!/usr/bin/env bash
# Run the acceptance-predictor analysis over the subtree-prefetch logs under one or more directories.
#
# Logs are produced by case_studies/predictive_prefetching_scripts/bfs_vs_predictive_dp.subtree_prefetch.vllm.*.sh (print_tree=true). Only logs
# that carry the sampler's "root log_p" lines are used, so every feature set is fit on the same edges; if no log has
# them, all logs are used and only the cut and tree features are scored. For each LOG_ROOT this writes, under
# OUTPUT_DIR (default: the LOG_ROOT itself):
#
#   pooled.acceptance-edges.csv                            one row per scored edge (extract/acceptance_features.py)
#   pooled.acceptance-predictors.{features,model,coefficients,selection}.csv and .<FIGURE_FORMAT>
#       pooled over all selected logs (draw/draw_acceptance_predictors.py)
#
# Without a LaTeX install on PATH, the figure falls back to Matplotlib's own text rendering.
#
# PER_LOG=1 also writes the same outputs beside each log, named after its run configuration.
#
# Usage, from anywhere:
#
#     bash /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/run_acceptance_predictors.sh [LOG_ROOT ...]
#
# LOG_ROOT defaults to case_studies/logs/lcb_v6, case_studies/logs-entropycut/lcb_v6, and
# case_studies/logs_predictive_prefetching/{Uniform,EntropyCut}-family/lcb_v6 (whichever exist) and is searched recursively for
# *.subtreePrefetch.vllm.log. Settings, as environment variables:
#
#   EPSILON=0.01         easy-case threshold: A >= 1 - EPSILON, A <= EPSILON, and the uncertain rest
#   HELD_ROOT_ONLY=0     1 keeps only edges whose decision node still holds the root state
#   FIGURE_FORMAT=pdf    pdf (needs LaTeX and qpdf) | png
#   MIN_GAIN=0.001       forward selection stops when the best remaining feature lowers held-out log loss by less
#   PER_LOG=0            1 also analyzes every log on its own
#   OUTPUT_DIR=          where the pooled outputs go; defaults to each LOG_ROOT
#   PYTHON=              interpreter to use; defaults to "uv run --project case_studies/draw python"

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
case_studies_dir="${repo_root}/case_studies"

epsilon=${EPSILON:-0.01}
held_root_only=${HELD_ROOT_ONLY:-0}
figure_format=${FIGURE_FORMAT:-pdf}
per_log=${PER_LOG:-0}
output_dir=${OUTPUT_DIR:-}
if [[ -n ${PYTHON:-} ]]; then
    read -r -a python_cmd <<< "${PYTHON}"
else
    python_cmd=(uv run --project "${script_dir}" python)
fi

figure_cache_root="${TMPDIR:-/tmp}/power-sharpening-cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${figure_cache_root}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${figure_cache_root}}"
mkdir -p -- "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"

log_roots=()
for arg in "$@"; do
    case ${arg} in
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]:-$0}"; exit 0 ;;
        -*) echo "unknown option: ${arg}" >&2; exit 2 ;;
        *) log_roots+=("${arg}") ;;
    esac
done
if (( ${#log_roots[@]} == 0 )); then
    for candidate in "${case_studies_dir}/logs/lcb_v6" "${case_studies_dir}/logs-entropycut/lcb_v6" \
            "${case_studies_dir}/logs_predictive_prefetching/Uniform-family/lcb_v6" \
            "${case_studies_dir}/logs_predictive_prefetching/EntropyCut-family/lcb_v6"; do
        [[ -d ${candidate} ]] && log_roots+=("${candidate}")
    done
fi
if (( ${#log_roots[@]} == 0 )); then
    echo "no LOG_ROOT given and no default log directory exists" >&2
    exit 2
fi

case ${figure_format} in
    pdf|png) ;;
    *) echo "FIGURE_FORMAT must be pdf or png, got ${figure_format}" >&2; exit 2 ;;
esac

analysis_args=(--epsilon "${epsilon}" --min-gain "${MIN_GAIN:-0.001}" --figure-format "${figure_format}")
[[ ${held_root_only} == 1 ]] && analysis_args+=(--held-root-only)
command -v latex > /dev/null || analysis_args+=(--no-usetex)

failures=()
for log_root in "${log_roots[@]}"; do
    if [[ ! -d ${log_root} ]]; then
        echo "missing log directory ${log_root}" >&2
        failures+=("${log_root}")
        continue
    fi
    mapfile -t all_logs < <(find -L "${log_root}" -type f -name '*.subtreePrefetch.vllm.log' | sort)
    if (( ${#all_logs[@]} == 0 )); then
        echo "no *.subtreePrefetch.vllm.log under ${log_root}" >&2
        failures+=("${log_root}")
        continue
    fi
    # Logs written with the feature printing carry "root log_p" lines; older ones only have the tree.
    mapfile -t logs < <(grep -l -m1 '^root log_p (len=' "${all_logs[@]}" 2>/dev/null)
    if (( ${#logs[@]} == 0 )); then
        echo "${log_root}: no log has root-score lines; scoring cut and tree features over all ${#all_logs[@]} logs" >&2
        logs=("${all_logs[@]}")
    else
        echo "${log_root}: ${#logs[@]} of ${#all_logs[@]} logs have root-score lines"
    fi

    pooled_dir=${output_dir:-${log_root}}
    mkdir -p -- "${pooled_dir}"
    pooled_prefix="${pooled_dir}/pooled"
    edges_csv="${pooled_prefix}.acceptance-edges.csv"

    echo "== extracting edges -> ${edges_csv}"
    if ! "${python_cmd[@]}" "${case_studies_dir}/extract/acceptance_features.py" \
            --log "${logs[@]}" --output "${edges_csv}" --epsilon "${epsilon}"; then
        failures+=("${log_root} (extract)")
        continue
    fi

    echo "== pooled analysis -> ${pooled_prefix}.acceptance-predictors.*"
    if ! "${python_cmd[@]}" "${script_dir}/draw_acceptance_predictors.py" \
            --edges "${edges_csv}" --output-prefix "${pooled_prefix}" "${analysis_args[@]}"; then
        failures+=("${log_root} (pooled analysis)")
    fi

    if [[ ${per_log} == 1 ]]; then
        for log in "${logs[@]}"; do
            echo "== per-log analysis: ${log}"
            if ! "${python_cmd[@]}" "${script_dir}/draw_acceptance_predictors.py" --log "${log}" "${analysis_args[@]}"; then
                failures+=("${log}")
            fi
        done
    fi
done

if (( ${#failures[@]} > 0 )); then
    echo "failed:" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

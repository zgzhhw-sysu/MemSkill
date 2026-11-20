#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
full="$base/source-compression/factorial"
late="$base/late-pairs"
history=("$work/study-equal-sources-gpu" "$LOCOMO_DATA" "$base")
inputs=("$work/study-equal-sources-gpu" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$base")
runner=(python3 "$guard" run --gpus 0 -- "$MEMSKILL_PYTHON" -m)
prev=''
while true; do
    n=$(find "$full" -mindepth 2 -maxdepth 2 -name measurement.json | wc -l)
    pairs=$(find "$late/outcomes" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    if [[ "$n:$pairs" != "$prev" ]]; then
        stages=(pilot)
        if ((n == 12)); then stages=(main pilot); fi
        for stage in "${stages[@]}"; do
            for family in prefix rank; do
                options=()
                if [[ "$family" == rank ]]; then options=(--rank); fi
                "${runner[@]}" experiments.controller.audit_prefix_return "$stage" "${inputs[@]}" "${options[@]}"
                if [[ "$stage" == pilot && ! -s "$late/$family-return/pilot/historical-budget.json" ]]; then
                    "${runner[@]}" experiments.controller.prefix_history "${history[@]}" "${options[@]}"
                fi
            done
        done
        prev="$n:$pairs"
    fi
    [[ -s "$late/prefix-return/main/summary.json" && -s "$late/prefix-return/pilot/summary.json" && -s "$late/rank-return/main/summary.json" && -s "$late/rank-return/pilot/summary.json" ]] && exit 0
    sleep 45
done

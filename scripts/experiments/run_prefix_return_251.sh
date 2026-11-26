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
prior_cross=-1
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
    cross="$late/cross-window-prefix"
    if [[ -s "$late/rank-return/main/forecast.json" && -s "$cross/manifest.json" ]]; then
        measured=$(find "$cross" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
        if [[ "$measured" != "$prior_cross" ]]; then
            "${runner[@]}" experiments.controller.audit_cross_window "$late"
            prior_cross=$measured
        fi
    fi
    if [[ -s "$late/prefix-return/main/summary.json" && -s "$late/prefix-return/pilot/summary.json" &&
          -s "$late/rank-return/main/summary.json" && -s "$late/rank-return/pilot/summary.json" &&
          ( ! -s "$cross/manifest.json" || -s "$cross/main-crossfit/summary.json" ) ]]; then exit 0; fi
    sleep 45
done

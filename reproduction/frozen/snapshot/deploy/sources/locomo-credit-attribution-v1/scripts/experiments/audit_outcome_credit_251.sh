#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
last=''
while true; do
    n=$(find "$base/source-compression/factorial-repairs" -maxdepth 1 -name 'conv-*_s*.json' | wc -l)
    p=$(find "$base/late-pairs/outcomes" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    if ((n > 0)) && [[ "$n:$p" != "$last" ]]; then
        python3 "$guard" run --gpus 0 -- "$MEMSKILL_PYTHON" -m experiments.controller.outcome_credit "$base"
        last="$n:$p"
        if ((n == 12 && p == 52)); then exit 0; fi
    fi
    sleep 45
done

#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/earlier-terminal-returns"
mkdir -p "$out"
export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
phase=${1:?expected writers, prepared or outcomes}
case $phase in
    writers) parent='' ;;
    prepared) parent=writers ;;
    outcomes) parent=prepared ;;
    *) exit 2 ;;
esac
cmd=("$MEMSKILL_PYTHON" -u -m experiments.controller.earlier_terminal_returns "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$phase")
until [[ -f "$out/$phase-complete.json" ]]; do
    pending=false
    if [[ -z $parent || -f "$out/$parent-complete.json" ]]; then
        pending=true
    else
        for f in "$out/$parent"/conv-*.json; do
            if [[ -f $f && ! -f "$out/$phase/${f##*/}" ]]; then pending=true; break; fi
        done
    fi
    if [[ $pending == true ]]; then
        rc=0
        if [[ $phase == prepared ]]; then
            python3 "$guard" run --gpus 3 -- "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$?
        elif bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
            "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$?
        fi
        if (( rc != 0 && rc != 75 )); then exit "$rc"; fi
    fi
    [[ -f "$out/$phase-complete.json" ]] || sleep 20
done

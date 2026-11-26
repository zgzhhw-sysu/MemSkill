#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/legacy-terminal-symmetry"
phase=${1:?expected writers, prepared, outcomes or root-pivots}
if [[ $phase == root-pivots ]]; then
    out="$base/source-compression/legacy-root-symmetry"
    mkdir -p "$out"
    export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
    until [[ -f "$out/writers-complete.json" ]]; do
        if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
            "$MEMSKILL_PYTHON" -u -m experiments.controller.legacy_root_pivots "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$out/writers.log" 2>&1
        else
            sleep 30
        fi
    done
    exit
fi
mkdir -p "$out"
export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
args=("$MEMSKILL_PYTHON" -u -m experiments.controller.legacy_terminal_symmetry "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$phase")
while [[ ! -f "$out/$phase-complete.json" ]]; do
    rc=0
    case $phase in
        writers)
            if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then "${args[@]}" >>"$out/$phase.log" 2>&1 || rc=$?; fi ;;
        prepared|outcomes)
            earlier=writers
            [[ $phase != outcomes ]] || earlier=prepared
            more=false
            for f in "$out/$earlier/"*.json; do
                if [[ -f $f && ! -f "$out/$phase/${f##*/}" ]]; then more=true; break; fi
            done
            if [[ $more == true || -f "$out/$earlier-complete.json" ]]; then
                if [[ $phase == prepared ]]; then
                    python3 "$guard" run --gpus 3 -- "${args[@]}" >>"$out/$phase.log" 2>&1 || rc=$?
                elif bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
                    "${args[@]}" >>"$out/$phase.log" 2>&1 || rc=$?
                fi
            fi ;;
        *) exit 2 ;;
    esac
    if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    [[ -f "$out/$phase-complete.json" ]] || sleep 20
done
if [[ $phase == outcomes && ! -f "$out/summary.json" ]]; then
    args[-1]=report
    "${args[@]}" >>"$out/$phase.log" 2>&1
fi

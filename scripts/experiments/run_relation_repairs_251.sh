#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
semantic="$base/source-compression/semantic-writer"
baseline=${2:-temporal}
case $baseline in temporal) folder=relation-skill-repairs ;; activity) folder=activity-relation-repairs ;; *) exit 2 ;; esac
root="$base/source-compression/$folder"
phase=${1:?expected prepared or outcomes}
case $phase in prepared|outcomes) ;; *) exit 2 ;; esac
while [[ ! -f "$root/0/$phase-complete.json" || ! -f "$root/1/$phase-complete.json" ]]; do
    for bit in 0 1; do
        out="$root/$bit"
        [[ ! -f "$out/$phase-complete.json" ]] || continue
        inputs="$semantic"
        if ((bit)); then inputs="$semantic/activity-prefix"; fi
        pending=false
        if [[ $phase == prepared ]]; then
            for f in "$inputs"/conv-*/pair.json; do
                name=${f%/*}
                if [[ -f $f && ! -f "$out/prepared/${name##*/}.json" ]]; then pending=true; break; fi
            done
        else
            for f in "$out"/prepared/*.json; do
                if [[ -f $f && ! -f "$out/outcomes/${f##*/}" ]]; then pending=true; break; fi
            done
        fi
        if [[ $pending == false && $(find "$out/$phase" -maxdepth 1 -name 'conv-*.json' 2>/dev/null | wc -l) != 12 ]]; then continue; fi
        mkdir -p "$out"
        export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
        cmd=("$MEMSKILL_PYTHON" -u -m experiments.controller.relation_skill_repairs "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$bit" "$phase" --baseline "$baseline")
        rc=0
        if [[ $phase == prepared ]]; then
            python3 "$guard" run --gpus 3 -- "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$?
        elif bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
            "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$?
        fi
        if ((rc != 0 && rc != 75)); then exit "$rc"; fi
    done
    sleep 25
done

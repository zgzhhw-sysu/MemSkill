#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/legacy-root-symmetry"
export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
phase=${1:?expected root replay phase}
case $phase in
    suffix-inputs) parent=writers ;;
    suffix-writes) parent=suffix-inputs ;;
    reader-inputs) parent=suffix-writes ;;
    outcomes) parent=reader-inputs ;;
    *) exit 2 ;;
esac
args=("$MEMSKILL_PYTHON" -u -m experiments.controller.legacy_root_replay "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$phase")
while [[ ! -f "$out/$phase-complete.json" ]]; do
    more=false
    for f in "$out/$parent/"*.json; do
        if [[ -f $f && ! -f "$out/$phase/${f##*/}" ]]; then more=true; break; fi
    done
    if [[ $more == true || -f "$out/$parent-complete.json" ]]; then
        rc=0
        case $phase in
            *-inputs) python3 "$guard" run --gpus 3 -- "${args[@]}" >>"$out/$phase.log" 2>&1 || rc=$? ;;
            *) if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then "${args[@]}" >>"$out/$phase.log" 2>&1 || rc=$?; fi ;;
        esac
        if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    fi
    [[ -f "$out/$phase-complete.json" ]] || sleep 15
done

#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/partial-skill-transport"
mkdir -p "$out"
export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
cmd=("$MEMSKILL_PYTHON" -u -m experiments.controller.partial_skill_measurement "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA")
"${cmd[@]}" register
until [[ -f "$out/prepared-complete.json" ]]; do
    rc=0
    python3 "$guard" run --gpus 3 -- "${cmd[@]}" prepared >>"$out/prepared.log" 2>&1 || rc=$?
    if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    [[ -f "$out/prepared-complete.json" ]] || sleep 20
done
until [[ -f "$base/late-pairs/semantic-writer/three-writer/transport-total/forecast.json" ]]; do sleep 10; done
until [[ -f "$out/outcomes-complete.json" ]]; do
    if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
        "${cmd[@]}" outcomes >>"$out/outcomes.log" 2>&1
    else
        sleep 20
    fi
done

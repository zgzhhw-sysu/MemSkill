#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen0-shared-b2-duty30
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight/source-compression"
out="$base/turn-keys"
mkdir -p "$out"
if [[ ${1:-} == screen ]]; then
    exec "$MEMSKILL_PYTHON" -u -m experiments.controller.turn_key_credit "$STUDY_DIR" "$LOCOMO_DATA" "$out"
fi
while [[ ! -s "$out/screen.json" ]]; do
    rc=0
    python3 "$guard" run --gpus 0 -- bash "$0" screen >>"$out/screen.log" 2>&1 || rc=$?
    if [[ $rc == 75 ]]; then sleep 30; continue; fi
    if ((rc != 0)); then exit "$rc"; fi
done

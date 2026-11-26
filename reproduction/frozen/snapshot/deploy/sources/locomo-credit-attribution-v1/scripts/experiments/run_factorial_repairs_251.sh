#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen2-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight/source-compression"
out="$base/factorial-repairs"
mkdir -p "$out"
if [[ ${1:-} == client ]]; then
    free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free < 4352)); then exit 75; fi
    bash scripts/experiments/run_cross_prefix_251.sh check-writer || exit 75
    exec bash scripts/experiments/paired_grounding.sh training-factorial-repairs --study-forecast "$base/frozen.json" --log-dir "$out/encoder-logs"
fi
while [[ ! -s "$out/summary.json" ]]; do
    ready=$(find "$base/factorial" -mindepth 2 -maxdepth 2 -name measurement.json | wc -l)
    checked=$(find "$out" -maxdepth 1 -name 'conv-*_s*.json' | wc -l)
    if ((ready <= checked && ready < 12)); then sleep 45; continue; fi
    rc=0
    python3 "$guard" run --gpus 2 -- bash "$0" client >>"$out/client.log" 2>&1 || rc=$?
    if [[ $rc == 75 ]]; then sleep 30; continue; fi
    if ((rc != 0)); then
        if [[ -z "$(ss -H -ltn 'sport = :18774')" ]]; then sleep 30; continue; fi
        exit "$rc"
    fi
done

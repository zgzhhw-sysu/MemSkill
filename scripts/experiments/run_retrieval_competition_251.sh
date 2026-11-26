#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export MEMSKILL_BACKEND_ID=251-qwen1-shared-b2-duty30-retrieval-competition
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/retrieval-competition"
mkdir -p "$out"
if [[ ${1:-} == measure ]]; then
    "$MEMSKILL_PYTHON" -u -m experiments.controller.retrieval_competition "$base" "$work/study-linked-eight-gpu"
    if [[ ! -s "$out/capacity.json" ]]; then
        "$MEMSKILL_PYTHON" -m experiments.controller.two_stage_capacity "$out"
    fi
    "$MEMSKILL_PYTHON" -m experiments.controller.audit_competition "$out"
    exec "$MEMSKILL_PYTHON" -m experiments.controller.paired_design_variance "$out"
fi
while [[ ! -s "$out/summary.json" ]]; do
    ready=$(find "$base/late-pairs/outcomes" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    if [[ -s "$out/through-$(printf '%02d' "$ready").json" ]]; then sleep 60; continue; fi
    rc=0
    python3 "$guard" run --gpus 1 -- bash "$0" measure >>"$out/client.log" 2>&1 || rc=$?
    if ((rc==75)); then sleep 30; continue; fi
    if ((rc!=0)); then exit "$rc"; fi
done

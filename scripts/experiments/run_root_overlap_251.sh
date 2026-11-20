#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight/source-compression"
out="$base/root-overlap"
"$MEMSKILL_PYTHON" -m experiments.controller.root_overlap_credit "$base" --register
while [[ ! -s "$out/summary.json" ]]; do
    if [[ ! -s "$base/root-write-repairs/summary.json" || ! -s "$base/retrieval-competition/calibration.json" ]]; then sleep 45; continue; fi
    count=$(find "$base/retrieval-competition/pairs" -maxdepth 1 -name '*.json' | wc -l)
    if [[ -s "$out/through-$(printf '%02d' "$count").json" ]]; then sleep 45; continue; fi
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.root_overlap_credit "$base" >>"$out/client.log" 2>&1
done

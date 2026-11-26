#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="$work/runs/paired-251-linked-eight/source-compression/semantic-writer"
while [[ ! -s "$out/source-forecast.json" ]]; do sleep 30; done
args=(-m experiments.controller.semantic_writer_balanced "$out")
python3 "$guard" run --gpus 0 -- "$MEMSKILL_PYTHON" "${args[@]}"
exec "$MEMSKILL_PYTHON" "${args[@]}" --watch

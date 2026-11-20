#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/reply-placement"
old="$base/source-compression/writer-block-repair"
mkdir -p "$out"
while [[ ! -s "$out/summary.json" ]]; do
    if [[ ! -e "$out/early.ready" ]]; then sleep 10; continue; fi
    prior=$(find "$old" -maxdepth 1 -name 'through-*.json' -printf '%f\n' | sort | tail -1)
    if [[ -s "$out/calibration.json" ]] && { [[ ! -e "$out/all.ready" ]] || [[ -z "$prior" ]] || [[ -s "$out/$prior" ]]; }; then
        sleep 40; continue
    fi
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.reply_placement "$base" >>"$out/policy.log" 2>&1
done

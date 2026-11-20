#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight/source-compression"
out="$base/writer-block-repair"
mkdir -p "$out"
while [[ ! -s "$out/summary.json" || ! -s "$out/population/through-12.json" ]]; do
    if [[ ! -s "$out/credit/summary.json" ]]; then sleep 25; continue; fi
    if [[ ! -s "$out/continuous-envelope.json" ]]; then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.terminal_credit_envelope "$base" >>"$out/envelope.log" 2>&1
    fi
    previous=$(find "$base/root-overlap" -maxdepth 1 -name 'through-*.json' -printf '%f\n' | sort -V | tail -1)
    windows=$(find "$out/population" -maxdepth 1 -name 'conv-*.json' | wc -l)
    if [[ -s "$out/fit.json" && -n "$previous" && -s "$out/$previous" ]] &&
            { ((windows == 0)) || [[ -s "$out/population/through-$(printf '%02d' "$windows").json" ]]; }; then
        sleep 40; continue
    fi
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.writer_block_policy "$base" >>"$out/policy.log" 2>&1
done

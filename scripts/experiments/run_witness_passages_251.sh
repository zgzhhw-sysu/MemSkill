#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-qwen4-b2-duty30-witness-passages MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
study="$work/study-linked-eight-gpu"
out="$base/source-compression/witness-passages"
mkdir -p "$out"
until [[ -s "$out/encoder.json" ]]; do
    rc=0
    python3 "$guard" run --gpus 4 -- "$MEMSKILL_PYTHON" -m experiments.controller.witness_passages "$base" "$study" "$LOCOMO_DATA" --encode >>"$out/encoder.log" 2>&1 || rc=$?
    if ((rc==75)); then sleep 45; elif ((rc!=0)); then exit "$rc"; fi
done
last=''
until [[ -s "$out/witness/later/summary.json" ]]; do
    pairs=$(find "$base/late-pairs/outcomes" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    state="$pairs-$(test -s "$base/source-compression/reader-slot-credit/single-rank/summary.json" && echo complete || echo partial)"
    if [[ $state != "$last" ]]; then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.witness_passage_policy "$base" "$study" "$LOCOMO_DATA" >>"$out/policy.log" 2>&1
        stage=group
        if [[ -s "$out/witness/forecast.json" ]]; then stage=witness; fi
        if [[ ! -s "$out/$stage/early-label-assignments.json" ]]; then
            python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.witness_passage_factorial "$base" "$study" --early-archive "$LOCOMO_DATA" >>"$out/early-archive.log" 2>&1
        fi
        last=$state
    fi
    sleep 60
done

#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export MEMSKILL_BACKEND_ID=251-qwen1-b2-duty30-reply-probes
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="$work/runs/paired-251-linked-eight/source-compression/reply-placement"
mkdir -p "$out"
for phase in early all; do
    extra=()
    if [[ $phase == early ]]; then extra+=(--early); fi
    while true; do
        rc=0
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.reply_probes \
            "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$out" "${extra[@]}" >>"$out/encoder.log" 2>&1 || rc=$?
        if ((rc == 0)); then break; fi
        if ((rc != 75)); then exit "$rc"; fi
        sleep 30
    done
    touch "$out/$phase.ready"
done

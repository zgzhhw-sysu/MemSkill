#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen1-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
dest="$base/source-compression/temporal-update-writer"
mkdir -p "$dest"
if [[ ${1:-} == client ]]; then
    free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free < 4096)); then exit 75; fi
    bash scripts/experiments/run_cross_prefix_251.sh check-writer || exit 75
    exec bash scripts/experiments/paired_grounding.sh training-temporal-update --study-forecast "$dest/forecast.json" --log-dir "$dest/encoder-logs"
fi
if [[ ! -s "$dest/forecast.json" ]]; then
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.temporal_update_forecast "$base" "$STUDY_DIR" "$LOCOMO_DATA"
fi
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
while [[ ! -s "$dest/early.json" || ! -s "$dest/later.json" ]]; do
    if ! python3 - "$dest" "$base" <<'PY'
import json, sys
from pathlib import Path
out, base=map(Path,sys.argv[1:])
windows=json.loads((out/'forecast.json').read_bytes())['windows']
pending=[w for w in windows if not (out/w['name']/'pair.json').exists()]
for w in pending:
    root=base/('source-compression/factorial' if w['panel']=='early' else 'late-pairs/outcomes')
    if (root/w['name']/(str(w['prefix'])+'0')/'scores.json').exists(): break
else:
    sys.exit(1 if pending else 0)
PY
    then sleep 45; continue; fi
    if [[ -z "$(ss -H -ltn 'sport = :18774')" ]]; then
        if [[ -n "$server" ]] && ! kill -0 "$server" 2>/dev/null; then
            rc=0; wait "$server" || rc=$?; server=''
            if ((rc != 75 && rc != 0)); then exit "$rc"; fi
        fi
        if [[ -z "$server" ]] && ! systemctl --user is-active --quiet memskill-251-{late-pairs,cross-prefix,late-selection-labels,case-credit-writer,donor-factorial}; then
            python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$dest/writer.log" 2>&1 &
            server=$!
        fi
        sleep 25; continue
    fi
    rc=0
    prepared=no
    for job in "$dest"/conv-*/reader-inputs.json; do
        if [[ -f "$job" && ! -f "${job%/*}/pair.json" ]]; then prepared=yes; break; fi
    done
    if [[ "$prepared" == no ]]; then
        python3 "$guard" run --gpus 1 -- bash "$0" client >>"$dest/client.log" 2>&1 || rc=$?
    fi
    if ((rc == 0)) && bash scripts/experiments/run_cross_prefix_251.sh check-writer; then
        "$MEMSKILL_PYTHON" -u -m experiments.controller.temporal_update_writer "$base" "$STUDY_DIR" "$LOCOMO_DATA" >>"$dest/reader.log" 2>&1
    fi
    if ((rc != 0 && rc != 75)); then exit "$rc"; fi
    sleep 45
done

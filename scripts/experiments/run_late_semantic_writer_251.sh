#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen1-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
late="$work/runs/paired-251-linked-eight/late-pairs"
out="$late/semantic-writer"
mkdir -p "$out"
if [[ ${1:-} == client ]]; then
    free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free < 4352)); then exit 75; fi
    bash scripts/experiments/run_cross_prefix_251.sh check-writer || exit 75
    exec bash scripts/experiments/paired_grounding.sh training-late-semantic-writer \
        --study-forecast "$late/frozen.json" --study-pair-limit "$2" --log-dir "$out/encoder-logs"
fi
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
while [[ ! -s "$out/summary.json" ]]; do
    count=$(find "$out" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    next=$(python3 - "$late/cohort.json" "$count" <<'PY'
import json, sys
rows = [r for r in json.load(open(sys.argv[1]))['assignments'] if r['pivot'] == 1]
i = int(sys.argv[2])
if i < len(rows):
    r = rows[i]
    print(f"{r['dialogue']}_s{r['start']}/{r['background']}0/scores.json")
PY
)
    if [[ -n "$next" && -s "$out/forecast.json" && ! -s "$late/outcomes/$next" ]]; then sleep 45; continue; fi
    if [[ -z "$(ss -H -ltn 'sport = :18774')" ]]; then
        if [[ -n "$server" ]] && ! kill -0 "$server" 2>/dev/null; then
            rc=0; wait "$server" || rc=$?; server=''
            if ((rc != 75 && rc != 0)); then exit "$rc"; fi
        fi
        if [[ -z "$server" && -s "$late/outcomes/summary.json" && -s "$late/cross-window-prefix/summary.json" ]] &&
                ! systemctl --user is-active --quiet memskill-251-cross-prefix; then
            python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$out/writer.log" 2>&1 &
            server=$!
        fi
        sleep 30; continue
    fi
    rc=0
    python3 "$guard" run --gpus 1 -- bash "$0" client "$((count+1))" >>"$out/client.log" 2>&1 || rc=$?
    if ((rc != 0 && rc != 75)); then
        if [[ -n "$(ss -H -ltn 'sport = :18774')" ]]; then exit "$rc"; fi
    fi
    if [[ ! -s "$out/summary.json" ]]; then sleep 25; fi
done

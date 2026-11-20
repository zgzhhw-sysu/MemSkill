#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen2-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
late="$base/late-pairs"
dest="$late/donor-factorial"
mkdir -p "$dest"
if [[ ${1:-} == client ]]; then
    free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free < 4352)); then exit 75; fi
    bash scripts/experiments/run_cross_prefix_251.sh check-writer || exit 75
    exec bash scripts/experiments/paired_grounding.sh training-late-pairs --study-forecast "$late/frozen.json" --study-complete-donors --study-pair-limit 1 --log-dir "$dest/encoder-logs"
fi
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
while [[ ! -s "$dest/summary.json" ]]; do
    if ! python3 - "$base" <<'PY'
import json,sys
from pathlib import Path
b=Path(sys.argv[1]); late=b/'late-pairs'; dest=late/'donor-factorial'
names=json.loads((late/'selection-labels/split.json').read_bytes())['windows']['donor']
ready=[n for n in names if (late/'outcomes'/n/'pair.json').exists() and (b/'source-compression/evidence-transport/pilot/predictions'/(n+'.json')).exists()]
sys.exit(0 if any(not (dest/n/'complete.json').exists() for n in ready) or len(ready)==12 else 1)
PY
    then sleep 50; continue; fi
    if [[ -z "$(ss -H -ltn 'sport = :18774')" ]]; then
        if [[ -n "$server" ]] && ! kill -0 "$server" 2>/dev/null; then
            rc=0; wait "$server" || rc=$?; server=''
            if ((rc != 75 && rc != 0)); then exit "$rc"; fi
        fi
        if [[ -z "$server" ]] && ! systemctl --user is-active --quiet memskill-251-{late-pairs,cross-prefix,late-selection-labels,late-semantic-writer,case-credit-writer}; then
            python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$dest/writer.log" 2>&1 &
            server=$!
        fi
        sleep 30; continue
    fi
    rc=0
    python3 "$guard" run --gpus 2 -- bash "$0" client >>"$dest/client.log" 2>&1 || rc=$?
    if ((rc != 0 && rc != 75)); then exit "$rc"; fi
    sleep 45
done

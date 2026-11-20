#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
src="$base/source-compression"
dest="$src/evidence-transport"
last_fit=''
last_audit=''
while [[ ! -s "$dest/expanded/later/summary.json" || ! -s "$dest/pilot/later/summary.json" ]]; do
    if [[ ! -e "$dest/early.ready" ]]; then sleep 15; continue; fi
    donors_ready=$(python3 - "$base/late-pairs/selection-labels" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
names = json.loads((folder / "split.json").read_bytes())["windows"]["donor"]
print(int(len(names) == 12 and all((folder / (name + ".json")).is_file() for name in names)))
PY
)
    ready=early
    if [[ -e "$dest/all.ready" ]]; then ready=all; fi
    if [[ "$last_fit" != "$ready-$donors_ready" ]]; then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.evidence_transport \
          "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$dest/fit.log" 2>&1
        last_fit="$ready-$donors_ready"
    fi
    pairs=$(find "$src/retrieval-competition/pairs" -name '*.json' | wc -l)
    if [[ "$last_audit" != "$last_fit-$pairs" ]]; then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.audit_evidence_transport "$base" >>"$dest/audit.log" 2>&1
        last_audit="$last_fit-$pairs"
    fi
    sleep 35
done

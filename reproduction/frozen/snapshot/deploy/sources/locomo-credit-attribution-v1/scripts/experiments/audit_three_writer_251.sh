#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
semantic="$base/late-pairs/semantic-writer"
out="$semantic/three-writer"
mkdir -p "$out"
while [[ ! -s "$out/summary.json" ]]; do
    if [[ ! -s "$semantic/forecast.json" ]]; then sleep 30; continue; fi
    if [[ -f "$out/registration.json" ]]; then
        if ! python3 - "$semantic" "$base/source-compression/retrieval-competition/pairs" <<'PY'
from pathlib import Path
import json, sys
root, activity = map(Path, sys.argv[1:])
for w in json.loads((root / "panel.json").read_bytes())["windows"]:
    name = f"{w['dialogue']}_s{w['start']}"
    if not (root / "three-writer" / (name + ".json")).is_file():
        pair = root / name / "pair.json"
        ready = pair.is_file() and (not json.loads(pair.read_bytes())["matched"] or (activity / (name + ".json")).is_file())
        sys.exit(0 if ready else 1)
PY
        then sleep 30; continue; fi
    fi
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.three_writer_credit "$base" "$LOCOMO_DATA" >>"$out/client.log" 2>&1
done

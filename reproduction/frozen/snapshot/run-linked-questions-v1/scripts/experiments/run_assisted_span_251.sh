#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
base="$work/runs/paired-251-linked-eight"
src="$base/source-compression"
dest="$src/assisted-span-backup"
late="$base/late-pairs"
mkdir -p "$dest"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_gpu() {
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.assisted_trial "$1" "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$dest/$1.log" 2>&1
}
last_fit=''
last_audit=''
until [[ -s "$dest/resolved/later/summary.json" && -s "$dest/expanded/later/summary.json" && -s "$dest/interim/later/summary.json" ]]; do
    if [[ ! -e "$src/evidence-transport/all.ready" ]]; then sleep 30; continue; fi
    ready=$(python3 - "$late" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); names=json.loads((p/'selection-labels/split.json').read_bytes())['windows']['donor']
ready=[n for n in names if (p/'selection-labels'/(n+'.json')).is_file()]
print(sum(n.endswith('_s9') for n in ready), len(ready), sum((p/'donor-factorial'/n/'complete.json').is_file() for n in names))
PY
)
    if [[ "$ready" != "$last_fit" ]]; then run_gpu fit; last_fit="$ready"; fi
    pairs=$(find "$src/retrieval-competition/pairs" -name '*.json' | wc -l)
    if [[ "$ready $pairs" != "$last_audit" ]]; then run_gpu audit; last_audit="$ready $pairs"; fi
    sleep 45
done

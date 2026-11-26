#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
out="$base/late-pairs/selection-labels"
mkdir -p "$out"
while [[ ! -s "$out/expanded/summary.json" ]]; do
    if [[ ! -s "$out/split.json" ]]; then sleep 25; continue; fi
    if ! python3 - "$out" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
split=json.loads((root/'split.json').read_bytes())['windows']
for label,group in (('pilot','donor'),('expanded','evaluation')):
    if label=='expanded' and any(not (root/(k+'.json')).is_file() for k in split['donor']):
        continue
    dest=root/label
    n=sum((root.parent/'outcomes'/k/'pair.json').is_file() for k in split[group])
    if not (dest/'fit.json').exists() or (n and not (dest/f'through-{n:02d}.json').is_file()):
        sys.exit(0)
sys.exit(1)
PY
    then sleep 35; continue; fi
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.late_credit_kernel "$base" >>"$out/kernel.log" 2>&1
done

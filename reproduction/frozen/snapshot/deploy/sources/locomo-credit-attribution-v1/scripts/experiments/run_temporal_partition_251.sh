#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
src="$work/runs/paired-251-linked-eight/source-compression"
parent="$src/temporal-update-writer"
audit="$src/temporal-update-audit"
dest="$audit/controller"
mkdir -p "$dest"
last=''
until [[ -s "$dest/twelve/summary.json" ]]; do
    ready=$(python3 - "$parent" "$audit" <<'PY'
from pathlib import Path
import sys
p,a=map(Path,sys.argv[1:])
print(sum(1 for _ in p.glob('conv-*/pair.json')),sum(1 for _ in a.glob('conv-*/result.json')))
PY
)
    if [[ -s "$parent/forecast.json" && "$ready" != "$last" ]]; then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.temporal_partition_policy "$parent/forecast.json" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$dest/fit.log" 2>&1
        last="$ready"
    fi
    sleep 45
done

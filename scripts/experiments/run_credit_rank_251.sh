#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
capped="$base/source-compression/capped-source-credit"
out="$capped/rank-allocation"
mkdir -p "$out"
last=''
until [[ -s "$out/joint-repaired/later/summary.json" && -s "$out/joint-repaired/long-prefix/summary.json" ]]; do
    state=$(python3 - "$base" "$capped" <<'PY'
from pathlib import Path
import sys
b,c=map(Path,sys.argv[1:])
print(*[int((c/'policy'/s/'fit.json').is_file()) for s in ('joint-mask','joint-repaired')],len(list((b/'late-pairs/outcomes').glob('conv-*/pair.json'))),len(list((b/'late-pairs/cross-window-prefix').glob('conv-*_to*/pair.json'))))
PY
)
    if [[ "$state" != "$last" && -s "$capped/policy/joint-mask/fit.json" ]]; then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.credit_rank_allocation "$base" >>"$out/policy.log" 2>&1
        last="$state"
    fi
    sleep 45
done
